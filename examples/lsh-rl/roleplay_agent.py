# Copyright (c) Microsoft. All rights reserved.
# pyright: reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

"""PersonaArena-style roleplay rollout for Agent-Lightning.

This module keeps the training integration stable while restoring key simulator behaviors:
- Environment agent and evaluator agent are separated objects.
- Per-round environment updates include scene update before characters act.
- Per-character self/env belief updates and synopsis generation before actions.
- Ordered full-character turns with configurable action/dialogue phases.
- Optional FAISS + GenerativeAgentMemory backend with graceful fallback.
- Detailed record dumps (persona/round/simulation) per scene rollout.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from openai import AsyncOpenAI

import agentlightning as agl

from reward_evaluator import (
    LONG_TERM_WEIGHT,
    SHORT_TERM_WEIGHT,
    TrajectoryEvaluator,
    TurnEvaluator,
    TurnVerifier,
    combine_short_long_rewards,
    is_long_term_bonus_eligible,
)


class CharacterSpec(TypedDict, total=False):
    """Character description from a scene file."""

    id: int
    name: str
    description: str
    position: str
    states: str
    is_npc: bool


class SceneTask(TypedDict, total=False):
    """Task schema for PersonaArena-style single-scene rollouts."""

    task_id: str
    scene_file: str
    scene_id: int
    title: str
    event: str
    time: str
    location: str
    description: str
    plot: str
    social_purpose: str
    max_rounds: int
    characters: List[CharacterSpec]
    actions: List[Dict[str, Any]]

    # environment agent routing
    environment_model: str
    environment_base_url: str
    environment_api_key: str

    # final evaluator routing
    evaluator_model: str
    evaluator_base_url: str
    evaluator_api_key: str


def _normalize_base_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("Empty base_url is not allowed.")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _strip_json_fence(text: str) -> str:
    content = text.strip()
    content = re.sub(r"^```(?:json)?\\s*", "", content)
    content = re.sub(r"\\s*```$", "", content)
    return content.strip()


def _extract_text_from_chat_response(response: Any) -> str:
    choice = response.choices[0]
    message = choice.message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                pieces.append(str(part.get("text", "")))
        return "\n".join(pieces).strip()
    return str(content).strip()


def _training_phase_from_env() -> str:
    """Return the rollout interaction phase used for training."""
    phase = os.environ.get("AGL_ROLEPLAY_TRAIN_PHASE", "full").strip().lower() or "full"
    if phase not in {"full", "dialogue_only", "action_only"}:
        raise ValueError(f"Unsupported AGL_ROLEPLAY_TRAIN_PHASE={phase!r}")
    return phase


TRAIN_PHASE = _training_phase_from_env()
ENABLE_ACTION_TURNS = TRAIN_PHASE in {"full", "action_only"}
ENABLE_DIALOGUE_TURNS = TRAIN_PHASE in {"full", "dialogue_only"}


def _strip_think_and_speaker(text: str, speaker: str) -> str:
    clean = text.strip()
    lower = clean.lower()
    idx = lower.rfind("</think>")
    if idx != -1:
        clean = clean[idx + len("</think>") :].strip()

    speaker_patterns = [
        rf"^{re.escape(speaker)}\\s*[:：\-–—]\\s*",
        rf"^{re.escape(speaker)}\\s*,\\s*",
    ]
    for pattern in speaker_patterns:
        clean = re.sub(pattern, "", clean, count=1)

    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    return lines[0] if lines else clean


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return token or "unknown"


def _history_to_text(history: List[Dict[str, Any]], *, max_chars: int = 0) -> str:
    lines: List[str] = []
    for turn in history:
        turn_type = turn.get("type", "dialogue")
        speaker = turn.get("speaker", "")
        text = turn.get("utterance", "")
        if turn_type == "action":
            lines.append(f"{speaker} [ACTION]: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    text = "\n".join(lines)
    if max_chars > 0 and len(text) > max_chars:
        return text[-max_chars:]
    return text


def _current_round_prefix_text(
    history: List[Dict[str, Any]],
    *,
    round_idx: int,
    max_turns: int = 4,
    max_chars: int = 1200,
) -> str:
    lines: List[str] = []
    for turn in history:
        if int(turn.get("round", -1)) != round_idx:
            continue
        turn_type = turn.get("type", "dialogue")
        speaker = turn.get("speaker", "")
        text = turn.get("utterance", "")
        if turn_type == "action":
            lines.append(f"{speaker} [ACTION]: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    text = "\n".join(lines[-max_turns:])
    if max_chars > 0 and len(text) > max_chars:
        return text[-max_chars:]
    return text


def _percentile(values: List[float], q: float) -> float:
    """Linear-interpolated percentile for small rollout reward batches."""
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    data = sorted(float(v) for v in values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return data[lo]
    frac = pos - lo
    return data[lo] * (1.0 - frac) + data[hi] * frac


async def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    enable_thinking: bool,
    timeout_seconds: float = 180.0,
) -> str:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout_seconds,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            )
            return _extract_text_from_chat_response(response)
        except Exception as exc:
            last_error = exc
            text = str(exc)
            transient = ("502" in text) or ("BadGateway" in text) or ("timeout" in text.lower())
            if transient and attempt < 3:
                backoff = (0.6 * (2**attempt)) + random.uniform(0.0, 0.4)
                await asyncio.sleep(backoff)
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("_chat_completion failed unexpectedly without an exception")


def _parse_json_dict(raw: str) -> Dict[str, Any]:
    content = _strip_json_fence(raw)
    try:
        loaded = json.loads(content)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass
    return {}


class CharacterMemory:
    """Hybrid memory wrapper.

    Prefers FAISS + GenerativeAgentMemory when dependencies are available.
    Falls back to an in-process rolling memory list otherwise.
    """

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self._fallback: List[str] = []
        self._memory_backend: Any = None
        self.enabled = False
        self.backend_name = "fallback"

        try:
            import faiss  # type: ignore
            from langchain.retrievers import TimeWeightedVectorStoreRetriever  # type: ignore
            from langchain_community.docstore import InMemoryDocstore  # type: ignore
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
            from langchain_community.vectorstores.faiss import FAISS  # type: ignore
            from langchain_experimental.generative_agents import GenerativeAgentMemory  # type: ignore

            chat_llm = None
            try:
                from langchain_openai import ChatOpenAI  # type: ignore

                chat_llm = ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    base_url=base_url,
                    temperature=0.0,
                    timeout=60,
                )
            except Exception:
                # GenerativeAgentMemory can still be initialized with a lightweight fallback object.
                chat_llm = None

            embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={"device": "cpu", "local_files_only": True},
            )
            dim = len(embeddings_model.embed_query("memory_probe"))
            index = faiss.IndexFlatL2(dim)
            vectorstore = FAISS(
                embeddings_model.embed_query,
                index,
                InMemoryDocstore({}),
                {},
                relevance_score_fn=lambda score: 1.0 - score / (2.0**0.5),
            )
            retriever = TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=5)
            self._memory_backend = GenerativeAgentMemory(
                llm=chat_llm,
                memory_retriever=retriever,
                verbose=False,
                reflection_threshold=10,
                max_tokens_limit=2500,
                importance_weight=0.1,
            )
            self.enabled = True
            self.backend_name = "faiss+generative"
        except Exception:
            self.enabled = False
            self.backend_name = "fallback"

    def add(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned:
            return
        self._fallback.append(cleaned)
        if len(self._fallback) > 400:
            self._fallback = self._fallback[-400:]
        if self.enabled and self._memory_backend is not None:
            try:
                self._memory_backend.add_memory(cleaned, now=datetime.now())
            except Exception:
                # Keep rollout stable even if memory backend has runtime issues.
                self.enabled = False
                self.backend_name = "fallback"

    def recent(self, max_items: int = 8) -> str:
        if max_items <= 0:
            return ""
        return "\n".join(self._fallback[-max_items:])


class CharacterRuntime:
    """Character-side runtime with internal reaction chain, aligned with original Character flow."""

    def __init__(
        self,
        *,
        spec: CharacterSpec,
        scene: SceneTask,
        memory: CharacterMemory,
        client: Any,
        model: str,
    ) -> None:
        self.spec = spec
        self.scene = scene
        self.memory = memory
        self.client = client
        self.model = model
        self.self_belief = ""
        self.env_belief = ""

    def _recent_memories(self) -> str:
        return self.memory.recent(max_items=12)

    async def _generate_reaction(
        self,
        *,
        observation: str,
        suffix: str,
        temperature: float = 0.7,
        trace_type: str = "unknown",
        history_text: str = "",
    ) -> str:
        system_prompt = "Act as the character in a realistic social scene and stay in role."
        history_block = f"[History]\n{history_text or '(empty)'}\n" if history_text else ""
        user_prompt = (
            f"[TraceType] {trace_type}\n"
            f"[Scene] Event: {self.scene.get('event', '')} | Time: {self.scene.get('time', '')} | "
            f"Location: {self.scene.get('location', '')} | Description: {self.scene.get('description', '')}\n"
            f"[Role] Name: {self.spec.get('name', '')} | Description: {self.spec.get('description', '')} | "
            f"Position: {self.spec.get('position', '')} | State: {self.spec.get('states', '')}\n"
            f"[Goal] {self.scene.get('social_purpose', '')}\n"
            f"[Observation]\n{observation or '(none)'}\n"
            f"{history_block}"
            f"[Self Belief] {self.self_belief}\n"
            f"[Env Belief] {self.env_belief}\n\n"
            f"{suffix}"
        )
        raw = await _chat_completion(
            self.client,
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=temperature,
            enable_thinking=False,
        )
        return _strip_think_and_speaker(raw, str(self.spec.get("name", "")))

    async def update_self_belief(self, *, observation: str, current_state: str) -> str:
        self.spec["states"] = current_state
        suffix = (
            "Assume you are this character and describe self-belief from first-person perspective.\n"
            "1. Belief: How do I perceive current situation and myself?\n"
            "2. Desire: What are my short-term and long-term goals?\n"
            "3. Intention: How do I intend to act next?\n"
            "Answer in a few concise sentences. Keep time-space-causality consistency."
        )
        self.self_belief = await self._generate_reaction(
            observation=observation,
            suffix=suffix,
            temperature=0.0,
            trace_type="self_belief",
        )
        return self.self_belief

    async def update_env_belief(self, *, observation: str, other_characters: List[str], current_state: str) -> str:
        self.spec["states"] = current_state
        suffix = (
            f"Other characters: {', '.join(other_characters) or '(none)'}\n"
            "Act as the character and describe environmental beliefs.\n"
            "1. View of others: intentions/relations/potential impact on me.\n"
            "2. Understanding of scene: contextual factors/challenges/opportunities.\n"
            "Provide <=3 concise sentences. Keep time-space-causality consistency."
        )
        self.env_belief = await self._generate_reaction(
            observation=observation,
            suffix=suffix,
            temperature=0.0,
            trace_type="env_belief",
        )
        return self.env_belief

    async def take_action(self, *, observation: str, plot: str, current_state: str) -> str:
        self.spec["states"] = current_state
        suffix = (
            f"Action hint from current plot beat (use only as a weak hint, not a script): {plot}\n"
            "Generate exactly one short, visible physical action in character.\n"
            "Hard constraints: the action must be directly observable; it must be a single action unit, not a mini-scene; "
            "it should create one concrete interaction affordance for the next turn by changing attention, distance, object state, access, or immediate social pressure; "
            "it must not duplicate a recent action pattern without a clear new purpose; "
            "do not include dialogue, inner thoughts, motives, symbolism, scene summary, or implied future consequences; "
            "do not convert the action hint into staged prose; keep it causally consistent and grounded in the immediate situation."
        )
        act = await self._generate_reaction(
            observation=observation,
            suffix=suffix,
            temperature=0.7,
            trace_type="action",
        )
        self.memory.add(f"{self.spec.get('name', '')} [ACTION]: {act}")
        return act

    async def generate_dialogue(
        self,
        *,
        observation: str,
        plot: str,
        current_state: str,
        action: str = "",
        history: List[Dict[str, Any]],
        round_idx: int,
    ) -> str:
        self.spec["states"] = current_state
        history_text = _current_round_prefix_text(history, round_idx=round_idx, max_turns=4, max_chars=1200)
        action_context = ""
        if action.strip():
            action_context += f"[Current Action] {action}\n"
        if plot.strip():
            action_context += f"[Current Turn Hint] {plot}\n"
        suffix = (
            f"{action_context}"
            "Generate exactly one concise utterance in character that responds naturally to the current situation.\n"
            "Hard constraints: one sentence; make one meaningful conversational contribution such as a statement, "
            "clarification, proposal, commitment, boundary, reveal, topic shift, or a context-grounded question; "
            "move the scene forward with a concrete new development, new angle, or clear next step; "
            "ground the line in the current-round interaction prefix when available; "
            "do not circle around the same object, detail, or fact pattern; "
            "do not reuse wording or the same interaction shell from the last one or two rounds unless there is a clear new consequence, decision, reveal, or shift in relationship; "
            "do not use trivia-style facts or generic probing questions just to sound informative; "
            "no action description/inner thoughts; causally consistent."
        )
        line = await self._generate_reaction(
            observation=observation,
            suffix=suffix,
            temperature=0.7,
            trace_type="dialogue",
            history_text=history_text,
        )
        self.memory.add(f"{self.spec.get('name', '')}: {line}")
        return line

    async def update_character(self, *, observation: str, current_state: str) -> tuple[str, str]:
        self.spec["states"] = current_state
        suffix = (
            "Based on scene and observation, summarize current position and state.\n"
            "Output strictly:\n"
            "Position: <text>\n"
            "State: <text>\n"
            "Keep consistency and reflect new developments."
        )
        raw = await self._generate_reaction(
            observation=observation,
            suffix=suffix,
            temperature=0.0,
            trace_type="update_character",
        )
        pos_match = re.search(r"(?:^|\n)\s*Position\s*[:：]\s*(.+)", raw)
        state_match = re.search(r"(?:^|\n)\s*State\s*[:：]\s*(.+)", raw)
        position = (pos_match.group(1).strip() if pos_match else str(self.spec.get("position", "")).strip()) or str(
            self.spec.get("position", "")
        ).strip()
        state = (state_match.group(1).strip() if state_match else current_state).strip() or current_state
        self.spec["position"] = position
        self.spec["states"] = state
        return position, state


class NarratorRuntime:
    """Narrator-side runtime aligned with original synopsis/scene/plot chain."""

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(base_url=_normalize_base_url(base_url), api_key=api_key)

    async def generate_synopsis(
        self,
        *,
        scene: SceneTask,
        actions: str,
        sequence: List[str],
        history: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        system_prompt = "Act as a realistic drama screenwriter and generate current plot per character."
        user_prompt = (
            "[TraceType] synopsis\n"
            "Story history:\n"
            f"{json.dumps(history, ensure_ascii=False)}\n\n"
            f"Past actions:\n{actions or '(empty)'}\n"
            f"Character action order: {sequence}\n"
            f"Goal: {scene.get('social_purpose', '')}\n"
            "For each character, write the unresolved next dramatic pressure they should respond to now, not a recap.\n"
            "Do not repeat or paraphrase a proposal, object, plan, joke, question, or topic already stated in the past actions "
            "unless it now creates a new decision, consequence, conflict, refusal, reveal, or relationship shift.\n"
            "Avoid keeping the same concrete object or plan in focus across rounds when it has already been proposed; "
            "advance to the next pressure, reaction, complication, or choice.\n"
            "The synopsis is a weak context hint for the next turn, not wording for the character to reuse.\n"
            "Output format must be exactly one line each: [Character Name]: [unresolved pressure and next beat]. "
            "No extra text."
        )
        raw = await _chat_completion(
            self.client,
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            enable_thinking=False,
        )
        synopsis: Dict[str, str] = {}
        matches = re.findall(r"([^:\n]+)[：:]\s*([^\n]+)", raw)
        for name, beat in matches:
            synopsis[name.strip(" []'\"")] = beat.strip(" []'\"")
        for name in sequence:
            synopsis.setdefault(name, "")
        return synopsis

    async def update_scene(
        self,
        *,
        time_text: str,
        location: str,
        description: str,
        observation: str,
    ) -> tuple[str, str, str]:
        system_prompt = "Update scene physical environment only."
        user_prompt = (
            "[TraceType] scene_update\n"
            "Given initial scene and observation, update only direct physical environment changes.\n"
            "If no major physical change, keep original values.\n"
            "Output strictly:\n"
            "Time: <text>\n"
            "Location: <text>\n"
            "Environment Description: <text>\n\n"
            f"Time: {time_text}\n"
            f"Location: {location}\n"
            f"Environment Description: {description}\n"
            f"Observation: {observation}\n"
        )
        raw = await _chat_completion(
            self.client,
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            enable_thinking=False,
        )
        time_match = re.search(r"(?:^|\n)\s*Time\s*[:：]\s*(.+)", raw)
        loc_match = re.search(r"(?:^|\n)\s*Location\s*[:：]\s*(.+)", raw)
        desc_match = re.search(r"(?:^|\n)\s*(?:Environment Description|Description)\s*[:：]\s*([\s\S]+)", raw)
        new_time = (time_match.group(1).strip() if time_match else time_text) or time_text
        new_loc = (loc_match.group(1).strip() if loc_match else location) or location
        new_desc = (desc_match.group(1).strip() if desc_match else description) or description
        return new_time, new_loc, new_desc

    async def summary_plot(self, *, actions: str, goal: str) -> str:
        system_prompt = "Summarize round plot concisely."
        user_prompt = (
            "[TraceType] summary_plot\n"
            f"Goal: {goal}\n"
            f"Round actions:\n{actions or '(empty)'}\n"
            "Provide one concise paragraph summarizing key actions and direct implications."
        )
        raw = await _chat_completion(
            self.client,
            model=self.model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            enable_thinking=False,
        )
        return raw.strip()


class EvaluatorAgentRuntime:
    """Final trajectory evaluator object (separate from environment agent object)."""

    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.client = AsyncOpenAI(base_url=_normalize_base_url(base_url), api_key=api_key)

    async def evaluate(self, *, scene: SceneTask, history: List[Dict[str, str]]) -> tuple[float, str]:
        transcript = _history_to_text(history)
        system_prompt = (
            "You are the environment evaluator for a roleplay RL trajectory. "
            "Assess whether the dialogue progresses toward the scene social purpose. "
            "Output strict JSON only."
        )
        user_prompt = (
            "[TraceType] evaluator\n"
            "Return JSON with keys: reward (0~1 float), reason (short string).\n\n"
            f"[Scene Event] {scene.get('event', '')}\n"
            f"[Social Purpose] {scene.get('social_purpose', '')}\n"
            f"[Scene Plot] {scene.get('plot', '')}\n\n"
            f"[Transcript]\n{transcript}\n"
        )

        raw = await _chat_completion(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            enable_thinking=False,
        )

        reward = 0.0
        reason = ""
        try:
            parsed = json.loads(_strip_json_fence(raw))
            reward = float(parsed.get("reward", 0.0))
            reason = str(parsed.get("reason", ""))
        except Exception:
            match = re.search(r"-?\\d+(?:\\.\\d+)?", raw)
            if match:
                reward = float(match.group(0))
            reason = raw[:300]

        reward = max(0.0, min(1.0, reward))
        return reward, reason


async def _generate_character_action(
    *,
    client: Any,
    model: str,
    scene: SceneTask,
    scene_description: str,
    character: CharacterSpec,
    history: List[Dict[str, str]],
    observation: str,
    current_state: str,
    self_belief: str,
    env_belief: str,
    synopsis: str,
    memory_excerpt: str,
) -> str:
    history_text = _history_to_text(history, max_chars=8000)
    system_prompt = (
        "Act as the character in a realistic social scene. "
        "Output one concise, visible physical action only."
    )
    user_prompt = (
        f"[Scene] Event: {scene.get('event', '')} | Time: {scene.get('time', '')} | Location: {scene.get('location', '')}\n"
        f"[Scene Description] {scene_description}\n"
        f"[Goal] Social objective: {scene.get('social_purpose', '')}\n"
        f"[Character] {character.get('name', '')}\n"
        f"[Character Description] {character.get('description', '')}\n"
        f"[Character Position] {character.get('position', '')}\n"
        f"[Character State] {current_state}\n"
        f"[Self Belief] {self_belief}\n"
        f"[Env Belief] {env_belief}\n"
        f"[Action Hint] {synopsis}\n"
        f"[Memory Excerpt]\n{memory_excerpt or '(none)'}\n"
        f"[Current Observation] {observation or '(none)'}\n\n"
        f"[History]\n{history_text or '(empty)'}\n\n"
        "Hard constraints:\n"
        "1. Output exactly one short, visible physical action.\n"
        "2. The action must be contextually logical and clearly observable.\n"
        "3. The action must function as a single action unit, not a mini-scene or narrated sequence.\n"
        "4. The action should create one concrete interaction affordance for the next turn by changing attention, distance, object state, access, or immediate social pressure.\n"
        "5. The action must not duplicate recent behavior without a clear new purpose.\n"
        "6. Do not include dialogue, thoughts, motives, symbolism, scene summary, or implied future consequences.\n"
        "7. Treat the action hint only as a weak guide, not a script to paraphrase.\n"
        "8. Keep the action causally consistent and grounded in the immediate situation.\n\n"
        "Generate one natural action phrase for this character."
    )
    raw = await _chat_completion(
        client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        enable_thinking=False,
    )
    return _strip_think_and_speaker(raw, str(character.get("name", "")))


async def _generate_character_dialogue(
    *,
    client: Any,
    model: str,
    scene: SceneTask,
    scene_description: str,
    character: CharacterSpec,
    history: List[Dict[str, str]],
    action: str,
    observation: str,
    current_state: str,
    self_belief: str,
    env_belief: str,
    synopsis: str,
    memory_excerpt: str,
) -> str:
    history_text = _history_to_text(history, max_chars=8000)
    system_prompt = (
        "Act as the character in a realistic social scene. "
        "Stay in character and output exactly one utterance."
    )
    user_prompt = (
        f"[Scene] Event: {scene.get('event', '')} | Time: {scene.get('time', '')} | Location: {scene.get('location', '')}\n"
        f"[Scene Description] {scene_description}\n"
        f"[Goal] Social objective: {scene.get('social_purpose', '')}\n"
        f"[Character] {character.get('name', '')}\n"
        f"[Character Description] {character.get('description', '')}\n"
        f"[Character Position] {character.get('position', '')}\n"
        f"[Character State] {current_state}\n"
        f"[Current Action] {action}\n"
        f"[Current Action Reference] {synopsis}\n"
        f"[Self Belief] {self_belief}\n"
        f"[Env Belief] {env_belief}\n"
        f"[Memory Excerpt]\n{memory_excerpt or '(none)'}\n"
        f"[Current Observation] {observation or '(none)'}\n\n"
        f"[History]\n{history_text or '(empty)'}\n\n"
        "Hard constraints:\n"
        "1. Output exactly one sentence.\n"
        "2. Respond naturally and make one meaningful conversational contribution such as a statement, clarification, "
        "proposal, commitment, boundary, reveal, topic shift, or a context-grounded question.\n"
        "3. Move the scene forward with a concrete new development, new angle, or clear next step.\n"
        "4. Do not keep circling the same object, detail, or fact pattern, and do not use trivia-style facts or generic probing questions just to sound informative.\n"
        "5. Do not include action descriptions or inner thoughts.\n"
        "6. Keep consistency with time, space, and causal flow.\n\n"
        "Generate exactly one natural utterance for this character, grounded in scene and history."
    )
    raw = await _chat_completion(
        client,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        enable_thinking=False,
    )
    return _strip_think_and_speaker(raw, str(character.get("name", "")))


def _write_records(
    *,
    task: SceneTask,
    round_records: List[Dict[str, Any]],
    environment_records: List[Dict[str, Any]],
    final_reward: float,
    final_reason: str,
) -> None:
    scene_file = str(task.get("scene_file", "scene_unknown"))
    title = Path(scene_file).stem if scene_file else _safe_token(str(task.get("title", "scene")))

    base_dir = Path(__file__).resolve().parent / "records" / title
    persona_dir = base_dir / "persona_detail"
    round_dir = base_dir / "round_detail"
    sim_dir = base_dir / "simulation"
    persona_dir.mkdir(parents=True, exist_ok=True)
    round_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    env_model = _safe_token(str(task.get("environment_model", task.get("evaluator_model", "env"))))
    role_model = _safe_token(str(task.get("task_id", "roleplay")))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    persona_path = persona_dir / f"{env_model}_{role_model}_{stamp}_character.jsonl"
    round_path = round_dir / f"{env_model}_{role_model}_{stamp}_round.jsonl"
    sim_path = sim_dir / f"{env_model}_{role_model}_{stamp}_simulation.txt"

    persona_payload = {
        "title": task.get("title", title),
        "scene_file": scene_file,
        "scene_id": task.get("scene_id", -1),
        "characters": task.get("characters", []),
    }
    with persona_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(persona_payload, ensure_ascii=False) + "\n")

    round_payload = {
        "task_id": task.get("task_id", ""),
        "scene_id": task.get("scene_id", -1),
        "event": task.get("event", ""),
        "social_purpose": task.get("social_purpose", ""),
        "round_records": round_records,
        "environment_records": environment_records,
        "final_reward": final_reward,
        "final_reason": final_reason,
    }
    with round_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(round_payload, ensure_ascii=False) + "\n")

    env_by_round: Dict[int, List[Dict[str, Any]]] = {}
    for item in environment_records:
        env_by_round.setdefault(int(item.get("round", -1)), []).append(item)

    with sim_path.open("a", encoding="utf-8") as f:
        f.write(f"task_id={task.get('task_id', '')} scene_id={task.get('scene_id', -1)}\n")
        max_round = max([int(r.get("round", 0)) for r in round_records] + [0])
        for round_idx in range(1, max_round + 1):
            f.write(f"round {round_idx}:\n")
            for env_item in env_by_round.get(round_idx, []):
                detail = env_item.get("detail", {})
                f.write(f"  [Environment] observation={detail.get('observation', '')}\n")
                f.write(f"  [Environment] scene_update={detail.get('scene_update', '')}\n")
            for item in round_records:
                if int(item.get("round", -1)) != round_idx:
                    continue
                c = item.get("character_name", "")
                t = item.get("type", "")
                text = item.get("text", "")
                f.write(f"  {c} ({t}): {text}\n")
            f.write("\n")
        f.write(f"final_reward={final_reward:.4f}\n")
        f.write(f"final_reason={final_reason}\n")
        f.write("-" * 80 + "\n")


@agl.rollout
async def roleplay_persona_agent(task: SceneTask, llm: agl.LLM) -> None:
    """Run one full-scene multi-character roleplay and emit a terminal reward."""
    print(f"\n{'='*80}")
    print(f"[SCENE START] task_id={task.get('task_id')} scene_id={task.get('scene_id')} title={task.get('title')}")
    print(f"[SCENE INFO] event={task.get('event')} location={task.get('location')}")
    print(f"[SCENE PURPOSE] {task.get('social_purpose')}")
    print(f"[CHARACTERS] {len(task.get('characters', []))} characters")
    print(f"{'='*80}\n")

    roleplay_client = AsyncOpenAI(
        base_url=_normalize_base_url(llm.get_base_url()),
        api_key=llm.api_key or os.environ.get("OPENAI_API_KEY", "xxx"),
    )

    env_base_url = (
        task.get("environment_base_url")
        or os.environ.get("ROLEPLAY_ENV_BASE_URL")
        or task.get("evaluator_base_url")
        or llm.endpoint
    )
    env_api_key = (
        task.get("environment_api_key")
        or os.environ.get("ROLEPLAY_ENV_API_KEY")
        or task.get("evaluator_api_key")
        or llm.api_key
    )
    env_model = (
        task.get("environment_model")
        or os.environ.get("ROLEPLAY_ENV_MODEL")
        or task.get("evaluator_model")
        or llm.model
    )

    evaluator_base_url = task.get("evaluator_base_url") or env_base_url
    evaluator_api_key = task.get("evaluator_api_key") or env_api_key
    evaluator_model = task.get("evaluator_model") or env_model

    narrator_agent = NarratorRuntime(
        base_url=str(env_base_url),
        api_key=str(env_api_key or "xxx"),
        model=str(env_model),
    )
    evaluator_agent = EvaluatorAgentRuntime(
        base_url=str(evaluator_base_url),
        api_key=str(evaluator_api_key or "xxx"),
        model=str(evaluator_model),
    )

    # Initialize reward evaluators
    enable_long_short_reward = os.environ.get("AGL_ENABLE_LONG_SHORT_REWARD", "true").lower() == "true"
    enable_short_term_reward = os.environ.get("AGL_ENABLE_SHORT_TERM_REWARD", "true").lower() == "true"
    if enable_long_short_reward:
        if enable_short_term_reward:
            turn_evaluator = TurnEvaluator(
                base_url=str(evaluator_base_url),
                api_key=str(evaluator_api_key or "xxx"),
                model=str(evaluator_model),
                enable_action_evaluation=True,
                enable_dialogue_evaluation=True,
            )
        else:
            turn_evaluator = None
        trajectory_evaluator = TrajectoryEvaluator(
            base_url=str(evaluator_base_url),
            api_key=str(evaluator_api_key or "xxx"),
            model=str(evaluator_model),
            enable_evaluation=True,
        )
        print("[REWARD SYSTEM] Long-Short Term Reward ENABLED")
        print(f"[REWARD SYSTEM] Short-term evaluator: {'ENABLED' if enable_short_term_reward else 'DISABLED'}")
        print(
            "[REWARD SYSTEM] Training reward = triplet credit assignment "
            f"(short-term base={SHORT_TERM_WEIGHT:.2f}, long-term budget={LONG_TERM_WEIGHT:.2f})"
        )
        print(f"[REWARD SYSTEM] Train phase = {TRAIN_PHASE}")
    else:
        turn_evaluator = None
        trajectory_evaluator = None
        print(f"[REWARD SYSTEM] DISABLED (using neutral rewards only)")

    enable_turn_verifier = (
        os.environ.get("AGL_ENABLE_TURN_VERIFIER", "true").lower() == "true"
        and turn_evaluator is not None
    )
    verifier_base_url = (
        os.environ.get("AGL_VERIFIER_BASE_URL")
        or os.environ.get("ROLEPLAY_VERIFIER_BASE_URL")
        or str(evaluator_base_url)
    )
    verifier_api_key = (
        os.environ.get("AGL_VERIFIER_API_KEY")
        or os.environ.get("ROLEPLAY_VERIFIER_API_KEY")
        or str(evaluator_api_key or "xxx")
    )
    verifier_model = (
        os.environ.get("AGL_VERIFIER_MODEL")
        or os.environ.get("ROLEPLAY_VERIFIER_MODEL")
        or str(evaluator_model)
    )
    turn_verifier = TurnVerifier(
        base_url=verifier_base_url,
        api_key=verifier_api_key,
        model=verifier_model,
        enabled=enable_turn_verifier,
    )
    print(
        f"[VERIFIER] {'ENABLED' if enable_turn_verifier else 'DISABLED'} "
        f"(model={verifier_model})"
    )

    characters = task.get("characters", [])
    character_states: Dict[str, str] = {}
    memories: Dict[str, CharacterMemory] = {}
    scene_context: SceneTask = {
        "event": str(task.get("event", "")),
        "time": str(task.get("time", "")),
        "location": str(task.get("location", "")),
        "description": str(task.get("description", "")),
        "social_purpose": str(task.get("social_purpose", "")),
    }
    for character in characters:
        name = str(character.get("name", "")).strip()
        if not name:
            continue
        character_states[name] = str(character.get("states", "")).strip()
        memory = CharacterMemory(
            base_url=_normalize_base_url(llm.get_base_url()),
            api_key=llm.api_key or os.environ.get("OPENAI_API_KEY", "xxx"),
            model=llm.model,
        )
        memories[name] = memory
        print(f"[MEMORY] {name}: backend={memory.backend_name}")

    character_agents: Dict[str, CharacterRuntime] = {}
    for character in characters:
        name = str(character.get("name", "")).strip()
        if not name:
            continue
        character_agents[name] = CharacterRuntime(
            spec=character,
            scene=scene_context,
            memory=memories[name],
            client=roleplay_client,
            model=llm.model,
        )

    history: List[Dict[str, Any]] = []
    environment_history: List[Dict[str, Any]] = []
    round_records: List[Dict[str, Any]] = []
    environment_records: List[Dict[str, Any]] = []

    # Store trajectories for long-term reward evaluation (per character)
    character_trajectories: Dict[str, List[Dict[str, Any]]] = {
        str(character.get("name", "")).strip(): [] for character in characters if character.get("name")
    }

    # Collect short-term rewards for each triplet (for final reward calculation)
    triplet_rewards: List[Dict[str, Any]] = []  # per turn: base reward + verifier-gated reward

    initial_actions = task.get("actions", [])
    for action in initial_actions:
        dialogue = str(action.get("dialogue", "")).strip()
        speaker = str(action.get("character", "")).strip()
        if dialogue and speaker:
            history.append({"speaker": speaker, "type": "dialogue", "utterance": dialogue, "round": 0})
            if speaker in memories:
                memories[speaker].add(f"{speaker}: {dialogue}")

    current_scene_description = str(scene_context.get("description", ""))
    max_rounds = 5
    last_round_observation = _history_to_text(history, max_chars=8000)
    synopsis_history: List[Dict[str, Any]] = []

    for round_idx in range(1, max_rounds + 1):
        print(f"\n{'─'*80}", flush=True)
        print(f"[ROUND {round_idx}/{max_rounds}]", flush=True)
        print(f"{'─'*80}\n", flush=True)

        # 1) belief updates + synopsis generation before character actions
        self_beliefs: Dict[str, str] = {}
        env_beliefs: Dict[str, str] = {}
        sequence: List[str] = [str(c.get("name", "")).strip() for c in characters if str(c.get("name", "")).strip()]

        for character in characters:
            name = str(character.get("name", "")).strip()
            if not name:
                continue
            runtime = character_agents[name]
            others = [
                str(c.get("name", "")).strip()
                for c in characters
                if str(c.get("name", "")).strip() and str(c.get("name", "")).strip() != name
            ]
            try:
                self_belief = await runtime.update_self_belief(
                    observation=last_round_observation,
                    current_state=character_states.get(name, ""),
                )
            except Exception as exc:
                self_belief = f"(self_belief_failed:{type(exc).__name__})"
            try:
                env_belief = await runtime.update_env_belief(
                    observation=last_round_observation,
                    other_characters=others,
                    current_state=character_states.get(name, ""),
                )
            except Exception as exc:
                env_belief = f"(env_belief_failed:{type(exc).__name__})"

            self_beliefs[name] = self_belief
            env_beliefs[name] = env_belief
            if name in memories:
                memories[name].add(f"SelfBelief: {self_belief}")
                memories[name].add(f"EnvBelief: {env_belief}")

        try:
            synopsis_map = await narrator_agent.generate_synopsis(
                scene=scene_context,
                actions=last_round_observation,
                sequence=sequence,
                history=synopsis_history,
            )
        except Exception as exc:
            synopsis_map = {name: f"(synopsis_failed:{type(exc).__name__})" for name in sequence}

        print("[BELIEF+SYNOPSIS]")
        for name in sequence:
            print(f"  {name} | self={self_beliefs.get(name, '')}")
            print(f"  {name} | env={env_beliefs.get(name, '')}")
            print(f"  {name} | synopsis={synopsis_map.get(name, '')}")

        # 2) fixed-order full character turns with configurable action/dialogue phases
        print("\n[CHARACTER TURNS]")
        round_lines: List[str] = []
        for character in characters:
            speaker = str(character.get("name", "")).strip()
            if not speaker:
                continue
            runtime = character_agents[speaker]
            current_state = character_states.get(speaker, str(character.get("states", "")))
            self_belief = self_beliefs.get(speaker, "")
            env_belief = env_beliefs.get(speaker, "")
            synopsis = synopsis_map.get(speaker, "")

            if ENABLE_ACTION_TURNS:
                try:
                    action = await runtime.take_action(
                        observation=last_round_observation,
                        plot=synopsis,
                        current_state=current_state,
                    )
                except Exception as exc:
                    action = f"(action_failed:{type(exc).__name__})"

                eval_history = list(history)
                history.append({"speaker": speaker, "type": "action", "utterance": action, "round": round_idx})
                round_lines.append(f"{speaker} [ACTION]: {action}")

                if speaker in character_trajectories:
                    character_trajectories[speaker].append(
                        {
                            "round": round_idx,
                            "type": "action",
                            "text": action,
                            "state": current_state,
                            "self_belief": self_belief,
                            "env_belief": env_belief,
                            "synopsis": synopsis,
                        }
                    )

                round_records.append(
                    {
                        "round": round_idx,
                        "character_name": speaker,
                        "character_id": character.get("id", -1),
                        "type": "action",
                        "text": action,
                        "state": current_state,
                        "self_belief": self_belief,
                        "env_belief": env_belief,
                        "synopsis": synopsis,
                    }
                )

                if turn_evaluator is not None:
                    try:
                        action_reward = await turn_evaluator.evaluate_action(
                            character=character,
                            action=action,
                            round_idx=round_idx,
                            history=eval_history,
                            scene=scene_context,
                        )
                        verifier_result = await turn_verifier.verify_turn(
                            scene=scene_context,
                            character=character,
                            history=eval_history,
                            turn_type="action",
                            text=action,
                            round_idx=round_idx,
                        )
                        gate_factor = float(verifier_result.get("gate_factor", 1.0))
                        gated_reward = max(0.0, min(1.0, action_reward * gate_factor))
                        triplet_rewards.append(
                            {
                                "character": speaker,
                                "type": "action",
                                "text": action,
                                "base_reward": action_reward,
                                "reward": gated_reward,
                                "verifier": verifier_result,
                            }
                        )
                        print(
                            f"  [{speaker}] ACTION REWARD: base={action_reward:.3f} "
                            f"gate={gate_factor:.2f} final={gated_reward:.3f}"
                        )
                    except Exception as exc:
                        print(f"  [{speaker}] ACTION EVAL FAILED: {exc}")
                        triplet_rewards.append(
                            {
                                "character": speaker,
                                "type": "action",
                                "text": action,
                                "base_reward": 0.5,
                                "reward": 0.5,
                                "verifier": {
                                    "severity": "none",
                                    "gate_factor": 1.0,
                                    "violations": [],
                                    "reason": "action_eval_failed",
                                },
                            }
                        )
                else:
                    triplet_rewards.append(
                        {
                            "character": speaker,
                            "type": "action",
                            "text": action,
                            "base_reward": 0.5,
                            "reward": 0.5,
                            "verifier": {
                                "severity": "none",
                                "gate_factor": 1.0,
                                "violations": [],
                                "reason": "turn_eval_disabled",
                            },
                        }
                    )

                print(f"  [{speaker}] ACTION: {action}")

            if ENABLE_DIALOGUE_TURNS:
                try:
                    dialogue = await runtime.generate_dialogue(
                        observation=last_round_observation,
                        plot=synopsis,
                        current_state=current_state,
                        action=action if ENABLE_ACTION_TURNS else "",
                        history=list(history),
                        round_idx=round_idx,
                    )
                except Exception as exc:
                    dialogue = f"(generation_failed:{type(exc).__name__})"

                eval_history = list(history)
                history.append({"speaker": speaker, "type": "dialogue", "utterance": dialogue, "round": round_idx})
                round_lines.append(f"{speaker}: {dialogue}")

                if speaker in character_trajectories:
                    character_trajectories[speaker].append(
                        {
                            "round": round_idx,
                            "type": "dialogue",
                            "text": dialogue,
                            "state": current_state,
                            "self_belief": self_belief,
                            "env_belief": env_belief,
                            "synopsis": synopsis,
                        }
                    )

                round_records.append(
                    {
                        "round": round_idx,
                        "character_name": speaker,
                        "character_id": character.get("id", -1),
                        "type": "dialogue",
                        "text": dialogue,
                        "state": current_state,
                        "self_belief": self_belief,
                        "env_belief": env_belief,
                        "synopsis": synopsis,
                    }
                )

                if turn_evaluator is not None:
                    try:
                        dialogue_reward = await turn_evaluator.evaluate_dialogue(
                            character=character,
                            dialogue=dialogue,
                            round_idx=round_idx,
                            history=eval_history,
                            scene=scene_context,
                        )
                        verifier_result = await turn_verifier.verify_turn(
                            scene=scene_context,
                            character=character,
                            history=eval_history,
                            turn_type="dialogue",
                            text=dialogue,
                            round_idx=round_idx,
                        )
                        gate_factor = float(verifier_result.get("gate_factor", 1.0))
                        gated_reward = max(0.0, min(1.0, dialogue_reward * gate_factor))
                        triplet_rewards.append(
                            {
                                "character": speaker,
                                "type": "dialogue",
                                "text": dialogue,
                                "base_reward": dialogue_reward,
                                "reward": gated_reward,
                                "verifier": verifier_result,
                            }
                        )
                        print(
                            f"  [{speaker}] DIALOGUE REWARD: base={dialogue_reward:.3f} "
                            f"gate={gate_factor:.2f} final={gated_reward:.3f}"
                        )
                    except Exception as exc:
                        print(f"  [{speaker}] DIALOGUE EVAL FAILED: {exc}")
                        triplet_rewards.append(
                            {
                                "character": speaker,
                                "type": "dialogue",
                                "text": dialogue,
                                "base_reward": 0.5,
                                "reward": 0.5,
                                "verifier": {
                                    "severity": "none",
                                    "gate_factor": 1.0,
                                    "violations": [],
                                    "reason": "dialogue_eval_failed",
                                },
                            }
                        )
                else:
                    triplet_rewards.append(
                        {
                            "character": speaker,
                            "type": "dialogue",
                            "text": dialogue,
                            "base_reward": 0.5,
                            "reward": 0.5,
                            "verifier": {
                                "severity": "none",
                                "gate_factor": 1.0,
                                "violations": [],
                                "reason": "turn_eval_disabled",
                            },
                        }
                    )

                print(f"  [{speaker}] DIALOGUE: {dialogue}")

        # 3) post-round narrator processing: update_scene + summary_plot + update_character
        round_observation = "\n".join(round_lines).strip()
        last_round_observation = round_observation or "(empty)"
        environment_history.append({"speaker": "Round", "type": "observation", "utterance": last_round_observation})

        prev_description = str(scene_context.get("description", ""))
        try:
            new_time, new_location, new_description = await narrator_agent.update_scene(
                time_text=str(scene_context.get("time", "")),
                location=str(scene_context.get("location", "")),
                description=prev_description,
                observation=last_round_observation,
            )
        except Exception as exc:
            print(f"[NARRATOR UPDATE_SCENE ERROR] {exc}")
            new_time = str(scene_context.get("time", ""))
            new_location = str(scene_context.get("location", ""))
            new_description = prev_description

        scene_context["time"] = new_time
        scene_context["location"] = new_location
        scene_context["description"] = new_description
        current_scene_description = new_description

        state_updates: Dict[str, str] = {}
        position_updates: Dict[str, str] = {}
        for character in characters:
            speaker = str(character.get("name", "")).strip()
            if not speaker:
                continue
            runtime = character_agents[speaker]
            try:
                new_pos, new_state = await runtime.update_character(
                    observation=last_round_observation,
                    current_state=character_states.get(speaker, ""),
                )
            except Exception as exc:
                print(f"[NARRATOR UPDATE_CHARACTER ERROR] {speaker}: {exc}")
                new_pos = str(character.get("position", ""))
                new_state = character_states.get(speaker, "")
            character["position"] = new_pos
            character["states"] = new_state
            character_states[speaker] = new_state
            state_updates[speaker] = new_state
            position_updates[speaker] = new_pos

        try:
            round_plot = await narrator_agent.summary_plot(
                actions=last_round_observation, goal=str(scene_context.get("social_purpose", ""))
            )
        except Exception as exc:
            round_plot = f"(summary_plot_failed:{type(exc).__name__})"
            print(f"[NARRATOR SUMMARY_PLOT ERROR] {exc}")

        synopsis_history.append({"round": round_idx, "synopsis": synopsis_map})
        environment_records.append(
            {
                "round": round_idx,
                "type": "post_round_update",
                "detail": {
                    "observation": last_round_observation,
                    "scene_update": new_description,
                    "prev_scene": prev_description,
                    "time": new_time,
                    "location": new_location,
                    "state_updates": state_updates,
                    "position_updates": position_updates,
                    "round_plot": round_plot,
                },
            }
        )
        print("[NARRATOR POST-ROUND]")
        print(f"  scene_update: {new_description[:220] if len(new_description) > 220 else new_description}")
        print(f"  round_plot: {round_plot[:220] if len(round_plot) > 220 else round_plot}")

    print(f"\n{'='*80}")
    print("[REWARD CALCULATION AND EMISSION]")

    # Step 1: Calculate long-term rewards for each character
    character_long_rewards: Dict[str, float] = {}

    if trajectory_evaluator is not None:
        for character in characters:
            name = str(character.get("name", "")).strip()
            if not name or name not in character_trajectories:
                continue

            trajectory = character_trajectories[name]
            if not trajectory:
                continue

            try:
                long_term_reward = await trajectory_evaluator.evaluate_trajectory(
                    character=character,
                    trajectory=trajectory,
                    scene=scene_context,
                )
                character_long_rewards[name] = long_term_reward
                print(f"[{name}] LONG-TERM REWARD: {long_term_reward:.4f}")
            except Exception as exc:
                print(f"[{name}] TRAJECTORY EVAL FAILED: {exc}")
                character_long_rewards[name] = 0.5  # Default neutral reward
    else:
        print("[LONG-TERM] Skipped (long-term reward set to 0.5 for all characters)")
        for character in characters:
            name = str(character.get("name", "")).strip()
            if name:
                character_long_rewards[name] = 0.5

    # Step 2: Calculate and emit combined rewards. Trainer will further perform
    # triplet-level long-term credit assignment before token-level PPO.
    print("\n[EMITTING FINAL REWARDS]")
    emitted_final_rewards: List[float] = []
    for i, triplet_data in enumerate(triplet_rewards):
        character = triplet_data["character"]
        short_reward = triplet_data["reward"]
        base_short_reward = triplet_data.get("base_reward", short_reward)
        verifier_result = triplet_data.get("verifier", {})
        long_reward = character_long_rewards.get(character, 0.5)
        long_bonus_eligible = is_long_term_bonus_eligible(short_reward)
        final_reward = combine_short_long_rewards(
            short_reward,
            long_reward,
            eligible_for_long_term_bonus=long_bonus_eligible,
        )
        emitted_final_rewards.append(float(final_reward))

        agl.emit_reward(
            final_reward,
            attributes={
                "agl.roleplay.character_name": character,
                "agl.roleplay.turn_type": str(triplet_data.get("type", "")),
                "agl.roleplay.base_short_term_reward": float(base_short_reward),
                "agl.roleplay.short_term_reward": float(short_reward),
                "agl.roleplay.long_term_reward": float(long_reward),
                "agl.roleplay.combined_reward": float(final_reward),
                "agl.roleplay.short_term_weight": float(SHORT_TERM_WEIGHT),
                "agl.roleplay.long_term_weight": float(LONG_TERM_WEIGHT),
                "agl.roleplay.long_term_bonus_eligible": bool(long_bonus_eligible),
                "agl.roleplay.verifier_gate_factor": float(verifier_result.get("gate_factor", 1.0)),
                "agl.roleplay.verifier_severity": str(verifier_result.get("severity", "none")),
                "agl.roleplay.verifier_violations": json.dumps(verifier_result.get("violations", []), ensure_ascii=False),
            },
        )

        # Debug: print first few and last few rewards
        if i < 3 or i >= len(triplet_rewards) - 3:
            print(
                f"  [{i:2d}] {character:10s} | base_short={base_short_reward:.3f} "
                f"gated_short={short_reward:.3f} long={long_reward:.3f} "
                f"bonus={'Y' if long_bonus_eligible else 'N'} → final={final_reward:.3f}"
            )
        elif i == 3:
            print(f"  ...")

    print(f"\n[TOTAL] Emitted {len(triplet_rewards)} final rewards")
    if emitted_final_rewards:
        n = len(emitted_final_rewards)
        mean_reward = sum(emitted_final_rewards) / n
        min_reward = min(emitted_final_rewards)
        max_reward = max(emitted_final_rewards)
        p25 = _percentile(emitted_final_rewards, 25)
        p50 = _percentile(emitted_final_rewards, 50)
        p75 = _percentile(emitted_final_rewards, 75)
        variance = sum((r - mean_reward) ** 2 for r in emitted_final_rewards) / n
        std_reward = math.sqrt(variance)
        print(
            "[ROLLOUT REWARD STATS] "
            f"n={n} mean={mean_reward:.3f} std={std_reward:.3f} "
            f"p25={p25:.3f} p50={p50:.3f} p75={p75:.3f} "
            f"min={min_reward:.3f} max={max_reward:.3f}"
        )

    # Step 3: Legacy terminal reward evaluation (for logging only, not emitted)
    print("\n[LEGACY TERMINAL REWARD EVALUATION (for comparison)]")
    try:
        eval_scene = dict(task)
        eval_scene["description"] = current_scene_description
        eval_scene["time"] = str(scene_context.get("time", eval_scene.get("time", "")))
        eval_scene["location"] = str(scene_context.get("location", eval_scene.get("location", "")))
        reward, reason = await evaluator_agent.evaluate(scene=eval_scene, history=environment_history + history)
        print(f"[LEGACY] reward={reward:.4f}")
        print(f"[LEGACY REASON] {reason}")
    except Exception as exc:
        reward, reason = 0.0, f"reward_eval_failed:{type(exc).__name__}"
        print(f"[ERROR] {exc}")

    try:
        _write_records(
            task={**task, "environment_model": str(env_model)},
            round_records=round_records,
            environment_records=environment_records,
            final_reward=character_long_rewards.get(list(character_long_rewards.keys())[0], 0.5),  # First character's long-term reward
            final_reason=f"Long-Short Term Reward System: {len(triplet_rewards)} triplets with final rewards emitted",
        )
    except Exception as exc:
        print(f"[RECORD ERROR] {exc}")

    print(f"\n{'='*80}")
    print(
        f"[SCENE COMPLETE] task_id={task.get('task_id', 'unknown')} "
        f"scene_file={task.get('scene_file', 'unknown')} scene_id={task.get('scene_id', -1)}"
    )
    print(f"[REWARD SUMMARY]")
    for character, long_reward in character_long_rewards.items():
        avg_short = sum(t["reward"] for t in triplet_rewards if t["character"] == character) / max(1, sum(1 for t in triplet_rewards if t["character"] == character))
        avg_base_short = sum(float(t.get("base_reward", t["reward"])) for t in triplet_rewards if t["character"] == character) / max(
            1, sum(1 for t in triplet_rewards if t["character"] == character)
        )
        avg_gate = (avg_short / avg_base_short) if avg_base_short > 1e-6 else 1.0
        avg_bonus_eligible = is_long_term_bonus_eligible(avg_short)
        avg_final = combine_short_long_rewards(
            avg_short,
            long_reward,
            eligible_for_long_term_bonus=avg_bonus_eligible,
        )
        print(
            f"  {character:10s}: avg_base_short={avg_base_short:.3f} avg_gated_short={avg_short:.3f} "
            f"gate={avg_gate:.3f} long={long_reward:.3f} bonus={'Y' if avg_bonus_eligible else 'N'} → avg_final={avg_final:.3f}"
        )
    print(f"{'='*80}\n")


async def debug() -> None:
    """Run one tiny in-process debug task without the training loop."""
    tracer = agl.OtelTracer()
    runner = agl.LitAgentRunner[SceneTask](tracer)
    store = agl.InMemoryLightningStore()

    resource = agl.LLM(
        endpoint=os.environ.get("ROLEPLAY_BASE_URL", "xxx"),
        model=os.environ.get("ROLEPLAY_MODEL", "xxx"),
        api_key=os.environ.get("ROLEPLAY_API_KEY", "xxx"),
        sampling_parameters={"temperature": 0.7},
    )

    debug_task: SceneTask = {
        "task_id": "debug-0",
        "scene_file": "debug",
        "scene_id": 0,
        "event": "Coffee chat before a meetup",
        "time": "Afternoon",
        "location": "Community center",
        "description": "Two old friends reconnect while preparing for a meetup.",
        "plot": "One person wants to repair trust after an argument.",
        "social_purpose": "Reveal values and past experiences naturally.",
        "max_rounds": 2,
        "characters": [
            {
                "id": 0,
                "name": "Alex",
                "description": "A thoughtful open source maintainer.",
                "position": "Near the coffee machine",
                "states": "Nervous but sincere",
                "is_npc": False,
            },
            {
                "id": 1,
                "name": "Jordan",
                "description": "A pragmatic engineer who values direct communication.",
                "position": "At the table",
                "states": "Calm and guarded",
                "is_npc": True,
            },
        ],
        "environment_base_url": os.environ.get("ROLEPLAY_ENV_BASE_URL", "xxx"),
        "environment_api_key": os.environ.get("ROLEPLAY_ENV_API_KEY", "xxx"),
        "environment_model": os.environ.get("ROLEPLAY_ENV_MODEL", "xxx"),
        "evaluator_base_url": os.environ.get("ROLEPLAY_ENV_BASE_URL", "xxx"),
        "evaluator_api_key": os.environ.get("ROLEPLAY_ENV_API_KEY", "xxx"),
        "evaluator_model": os.environ.get("ROLEPLAY_ENV_MODEL", "xxx"),
    }

    with runner.run_context(agent=roleplay_persona_agent, store=store):
        await runner.step(debug_task, resources={"roleplay_llm": resource})


if __name__ == "__main__":
    asyncio.run(debug())
