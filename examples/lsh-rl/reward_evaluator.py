# # Copyright (c) Microsoft. All rights reserved.

# """Long-Short Term Reward Evaluators for Roleplay Persona Training.

# This module implements:
# 1. TurnEvaluator: Evaluates short-term reward after each action/dialogue
# 2. TrajectoryEvaluator: Evaluates long-term reward after complete scene

# Based on Persona Arena evaluation criteria and RolePlay-RL reward design.
# """

# from __future__ import annotations

# import json
# import logging
# import re
# from typing import Any, Dict, List

# from openai import AsyncOpenAI

# logger = logging.getLogger(__name__)


# # Weights for action evaluation (6 dimensions)
# ACTION_WEIGHTS = {
#     "Correctness": 0.12,
#     "Character Consistency": 0.15,
#     "Logical Coherence": 0.12,
#     "Naturalness": 0.10,
#     "Anti-template": 0.06,
#     "Anti-dramatic": 0.05,
# }

# # Weights for dialogue evaluation (10 dimensions)
# DIALOGUE_WEIGHTS = {
#     "Correctness": 0.12,
#     "Character Consistency": 0.15,
#     "Logical Coherence": 0.12,
#     "Content Non-redundancy": 0.08,
#     "Emotional Expression": 0.12,
#     "Interaction Adaptability": 0.12,
#     "Creativity": 0.08,
#     "Naturalness": 0.10,
#     "Anti-template": 0.06,
#     "Anti-dramatic": 0.05,
# }

# # Weights for trajectory evaluation (5 dimensions)
# TRAJECTORY_WEIGHTS = {
#     "Persona Consistency": 0.25,
#     "Immersion": 0.20,
#     "Behavioral Coherence": 0.20,
#     "Adaptability": 0.20,
#     "Interaction Richness": 0.15,
# }


# def _normalize_base_url(base_url: str) -> str:
#     """Normalize base URL to ensure it ends with /v1."""
#     base = (base_url or "").strip().rstrip("/")
#     if not base:
#         raise ValueError("Empty base_url is not allowed.")
#     if base.endswith("/v1"):
#         return base
#     return f"{base}/v1"


# def _strip_json_fence(text: str) -> str:
#     """Strip JSON code fences from LLM response."""
#     content = text.strip()
#     content = re.sub(r"^```(?:json)?\s*", "", content)
#     content = re.sub(r"\s*```$", "", content)
#     return content.strip()


# def _extract_json_scores(response: str, expected_keys: List[str]) -> Dict[str, int]:
#     """Extract scores from JSON response with fallback parsing."""
#     try:
#         obj = json.loads(_strip_json_fence(response))
#         scores = {}
#         for key in expected_keys:
#             if key in obj:
#                 scores[key] = max(1, min(5, int(obj[key])))
#             else:
#                 # Try fuzzy matching
#                 for obj_key in obj.keys():
#                     if key.lower() in obj_key.lower() or obj_key.lower() in key.lower():
#                         scores[key] = max(1, min(5, int(obj[obj_key])))
#                         break
#                 if key not in scores:
#                     scores[key] = 3  # Default to neutral score
#         return scores
#     except Exception as e:
#         logger.warning(f"Failed to parse JSON response: {e}, using default scores")
#         return {key: 3 for key in expected_keys}


# def _compute_weighted_average(scores: Dict[str, int], weights: Dict[str, float]) -> float:
#     """Compute weighted average of scores and normalize to [0, 1]."""
#     total_weight = sum(weights.values())
#     weighted_sum = sum(scores.get(key, 3) * weight for key, weight in weights.items())
#     weighted_avg = weighted_sum / total_weight if total_weight > 0 else 3.0
#     # Normalize from [1, 5] to [0, 1]
#     return (weighted_avg - 1.0) / 4.0


# def _history_to_text(history: List[Dict[str, str]], max_chars: int = 2000) -> str:
#     """Convert history to text format for evaluation context."""
#     lines = []
#     total_chars = 0
#     for turn in reversed(history):  # Most recent first
#         turn_type = turn.get("type", "dialogue")
#         speaker = turn.get("speaker", "")
#         text = turn.get("utterance", "")
#         if turn_type == "action":
#             line = f"{speaker} [ACTION]: {text}"
#         else:
#             line = f"{speaker}: {text}"
#         lines.insert(0, line)  # Keep chronological order
#         total_chars += len(line)
#         if max_chars > 0 and total_chars > max_chars:
#             break
#     return "\n".join(lines[-10:])  # Keep last 10 turns max


# class TurnEvaluator:
#     """Evaluates short-term reward immediately after each action/dialogue.

#     This evaluator is called after every character turn to provide immediate
#     feedback for PPO training.
#     """

#     def __init__(
#         self,
#         *,
#         base_url: str,
#         api_key: str,
#         model: str,
#         enable_action_evaluation: bool = True,
#         enable_dialogue_evaluation: bool = True,
#     ):
#         self.base_url = _normalize_base_url(base_url)
#         self.api_key = api_key or "xxx"
#         self.model = model
#         self.enable_action_evaluation = enable_action_evaluation
#         self.enable_dialogue_evaluation = enable_dialogue_evaluation
#         self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

#     async def evaluate_action(
#         self,
#         *,
#         character: Any,  # Accept CharacterSpec or Dict
#         action: str,
#         round_idx: int,
#         history: List[Dict[str, str]],
#         scene: Any,  # Accept SceneTask or Dict
#     ) -> float:
#         """Evaluate a single action and return reward in [0, 1].

#         Args:
#             character: Character specification (CharacterSpec or dict)
#             action: The action text to evaluate
#             round_idx: Current round number (1-indexed)
#             history: Conversation history so far
#             scene: Scene/task information (SceneTask or dict)

#         Returns:
#             Reward value in range [0, 1]
#         """
#         if not self.enable_action_evaluation:
#             return 0.5  # Neutral reward if disabled

#         # Convert to dict if needed
#         char_dict = dict(character) if not isinstance(character, dict) else character
#         scene_dict = dict(scene) if not isinstance(scene, dict) else scene

#         try:
#             response = await self._call_action_evaluator(
#                 character=char_dict,
#                 action=action,
#                 round_idx=round_idx,
#                 history=history,
#                 scene=scene_dict,
#             )
#             scores = _extract_json_scores(response, list(ACTION_WEIGHTS.keys()))
#             reward = _compute_weighted_average(scores, ACTION_WEIGHTS)
#             logger.debug(f"[Action Eval] Round {round_idx} {char_dict.get('name')}: {reward:.3f}")
#             return reward
#         except Exception as e:
#             logger.warning(f"[Action Eval] Failed for round {round_idx}: {e}")
#             return 0.5

#     async def evaluate_dialogue(
#         self,
#         *,
#         character: Any,  # Accept CharacterSpec or Dict
#         dialogue: str,
#         round_idx: int,
#         history: List[Dict[str, str]],
#         scene: Any,  # Accept SceneTask or Dict
#     ) -> float:
#         """Evaluate a single dialogue and return reward in [0, 1].

#         Args:
#             character: Character specification (CharacterSpec or dict)
#             dialogue: The dialogue text to evaluate
#             round_idx: Current round number (1-indexed)
#             history: Conversation history so far
#             scene: Scene/task information (SceneTask or dict)

#         Returns:
#             Reward value in range [0, 1]
#         """
#         if not self.enable_dialogue_evaluation:
#             return 0.5  # Neutral reward if disabled

#         # Convert to dict if needed
#         char_dict = dict(character) if not isinstance(character, dict) else character
#         scene_dict = dict(scene) if not isinstance(scene, dict) else scene

#         try:
#             response = await self._call_dialogue_evaluator(
#                 character=char_dict,
#                 dialogue=dialogue,
#                 round_idx=round_idx,
#                 history=history,
#                 scene=scene_dict,
#             )
#             scores = _extract_json_scores(response, list(DIALOGUE_WEIGHTS.keys()))
#             reward = _compute_weighted_average(scores, DIALOGUE_WEIGHTS)
#             logger.debug(f"[Dialogue Eval] Round {round_idx} {char_dict.get('name')}: {reward:.3f}")
#             return reward
#         except Exception as e:
#             logger.warning(f"[Dialogue Eval] Failed for round {round_idx}: {e}")
#             return 0.5

#     async def _call_action_evaluator(
#         self,
#         *,
#         character: Dict[str, Any],
#         action: str,
#         round_idx: int,
#         history: List[Dict[str, str]],
#         scene: Dict[str, Any],
#     ) -> str:
#         """Call LLM to evaluate action quality."""
#         history_text = _history_to_text(history, max_chars=2000)

#         system_prompt = "You are an expert role-play evaluator. Rate character actions objectively."
#         user_prompt = f"""[TraceType] action_evaluation

# Evaluate the quality of this character's action based on the scene and context.

# [Scene]
# Event: {scene.get('event', '')}
# Time: {scene.get('time', '')}
# Location: {scene.get('location', '')}
# Description: {scene.get('description', '')}
# Social Purpose: {scene.get('social_purpose', '')}

# [Character]
# Name: {character.get('name', '')}
# Description: {character.get('description', '')}
# Position: {character.get('position', '')}
# State: {character.get('states', '')}

# [Context - Recent Actions/Dialogue]
# {history_text if history_text else '(This is the first action)'}

# [Current Action to Evaluate - Round {round_idx}]
# {action}

# [Evaluation Dimensions]
# Rate each dimension on a scale of 1-5:

# 1. Correctness (12%): Is the action factually accurate and consistent with the scene?
# 2. Character Consistency (15%): Does it match the character's traits and goals?
# 3. Logical Coherence (12%): Is the action logically consistent with context?
# 4. Naturalness (10%): Is it natural and daily-life style, not performative?
# 5. Anti-template (6%): Does it avoid template/cliché expressions?
# 6. Anti-dramatic (5%): Is it restrained, not overly dramatic?

# Output ONLY JSON with no extra text:
# {{
#   "Correctness": <1-5>,
#   "Character Consistency": <1-5>,
#   "Logical Coherence": <1-5>,
#   "Naturalness": <1-5>,
#   "Anti-template": <1-5>,
#   "Anti-dramatic": <1-5>
# }}
# """

#         response = await self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             temperature=0.0,
#             timeout=60,
#         )
#         return response.choices[0].message.content or ""

#     async def _call_dialogue_evaluator(
#         self,
#         *,
#         character: Dict[str, Any],
#         dialogue: str,
#         round_idx: int,
#         history: List[Dict[str, str]],
#         scene: Dict[str, Any],
#     ) -> str:
#         """Call LLM to evaluate dialogue quality."""
#         history_text = _history_to_text(history, max_chars=2000)

#         system_prompt = "You are an expert role-play evaluator. Rate character dialogue objectively."
#         user_prompt = f"""[TraceType] dialogue_evaluation

# Evaluate the quality of this character's dialogue based on the scene and context.

# [Scene]
# Event: {scene.get('event', '')}
# Time: {scene.get('time', '')}
# Location: {scene.get('location', '')}
# Description: {scene.get('description', '')}
# Social Purpose: {scene.get('social_purpose', '')}

# [Character]
# Name: {character.get('name', '')}
# Description: {character.get('description', '')}
# Position: {character.get('position', '')}
# State: {character.get('states', '')}

# [Context - Recent Actions/Dialogue]
# {history_text if history_text else '(This is the first dialogue)'}

# [Current Dialogue to Evaluate - Round {round_idx}]
# {character.get('name', '')}: {dialogue}

# [Evaluation Dimensions]
# Rate each dimension on a scale of 1-5:

# 1. Correctness (12%): Is the dialogue factually accurate?
# 2. Character Consistency (15%): Does it match character's personality?
# 3. Logical Coherence (12%): Is it coherent with the conversation flow?
# 4. Content Non-redundancy (8%): Does it add new information, not repetitive?
# 5. Emotional Expression (12%): Is emotion expressed appropriately?
# 6. Interaction Adaptability (12%): Does it respond naturally to others?
# 7. Creativity (8%): Is it original and creative?
# 8. Naturalness (10%): Is it conversational and natural?
# 9. Anti-template (6%): Does it avoid template phrases?
# 10. Anti-dramatic (5%): Is it restrained, not overly dramatic?

# Output ONLY JSON with no extra text:
# {{
#   "Correctness": <1-5>,
#   "Character Consistency": <1-5>,
#   "Logical Coherence": <1-5>,
#   "Content Non-redundancy": <1-5>,
#   "Emotional Expression": <1-5>,
#   "Interaction Adaptability": <1-5>,
#   "Creativity": <1-5>,
#   "Naturalness": <1-5>,
#   "Anti-template": <1-5>,
#   "Anti-dramatic": <1-5>
# }}
# """

#         response = await self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             temperature=0.0,
#             timeout=60,
#         )
#         return response.choices[0].message.content or ""


# class TrajectoryEvaluator:
#     """Evaluates long-term reward after complete scene trajectory.

#     This evaluator is called once per character after the entire scene
#     is complete to assess overall performance.
#     """

#     def __init__(
#         self,
#         *,
#         base_url: str,
#         api_key: str,
#         model: str,
#         enable_evaluation: bool = True,
#     ):
#         self.base_url = _normalize_base_url(base_url)
#         self.api_key = api_key or "xxx"
#         self.model = model
#         self.enable_evaluation = enable_evaluation
#         self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

#     async def evaluate_trajectory(
#         self,
#         *,
#         character: Any,  # Accept CharacterSpec or Dict
#         trajectory: List[Dict[str, Any]],
#         scene: Any,  # Accept SceneTask or Dict
#     ) -> float:
#         """Evaluate a complete character trajectory and return reward in [0, 1].

#         Args:
#             character: Character specification (CharacterSpec or dict)
#             trajectory: List of all turns (action + dialogue) for this character
#             scene: Scene/task information (SceneTask or dict)

#         Returns:
#             Reward value in range [0, 1]
#         """
#         if not self.enable_evaluation:
#             return 0.5  # Neutral reward if disabled

#         # Convert to dict if needed
#         char_dict = dict(character) if not isinstance(character, dict) else character
#         scene_dict = dict(scene) if not isinstance(scene, dict) else scene

#         try:
#             response = await self._call_trajectory_evaluator(
#                 character=char_dict,
#                 trajectory=trajectory,
#                 scene=scene_dict,
#             )
#             scores = _extract_json_scores(response, list(TRAJECTORY_WEIGHTS.keys()))
#             reward = _compute_weighted_average(scores, TRAJECTORY_WEIGHTS)
#             logger.info(f"[Trajectory Eval] {char_dict.get('name')}: {reward:.3f}")
#             return reward
#         except Exception as e:
#             logger.warning(f"[Trajectory Eval] Failed for {char_dict.get('name')}: {e}")
#             return 0.5

#     async def _call_trajectory_evaluator(
#         self,
#         *,
#         character: Dict[str, Any],
#         trajectory: List[Dict[str, Any]],
#         scene: Dict[str, Any],
#     ) -> str:
#         """Call LLM to evaluate trajectory quality."""
#         # Build trajectory text
#         trajectory_lines = []
#         for turn in trajectory:
#             round_idx = turn.get("round", 0)
#             turn_type = turn.get("type", "")
#             text = turn.get("text", "")
#             if turn_type == "action":
#                 trajectory_lines.append(f"Round {round_idx} [ACTION]: {text}")
#             else:
#                 trajectory_lines.append(f"Round {round_idx} [Dialogue]: {text}")

#         trajectory_text = "\n".join(trajectory_lines)

#         system_prompt = "You are an expert role-play evaluator. Assess overall character performance across the entire scene."
#         user_prompt = f"""[TraceType] trajectory_evaluation

# Evaluate the overall quality of this character's performance across the entire scene.

# [Scene]
# Event: {scene.get('event', '')}
# Time: {scene.get('time', '')}
# Location: {scene.get('location', '')}
# Description: {scene.get('description', '')}
# Social Purpose: {scene.get('social_purpose', '')}

# [Character]
# Name: {character.get('name', '')}
# Description: {character.get('description', '')}

# [Complete Character Trajectory]
# {trajectory_text}

# [Evaluation Dimensions]
# Rate each dimension on a scale of 1-5:

# 1. Persona Consistency (25%): Does the character maintain consistent traits throughout?
# 2. Immersion (20%): Is the portrayal immersive and consistently in-character?
# 3. Behavioral Coherence (20%): Are actions/dialogue coherent across turns and plot progression?
# 4. Adaptability (20%): Does the character adapt well to context changes?
# 5. Interaction Richness (15%): Is the interaction varied and does it advance conversation?

# Output ONLY JSON with no extra text:
# {{
#   "Persona Consistency": <1-5>,
#   "Immersion": <1-5>,
#   "Behavioral Coherence": <1-5>,
#   "Adaptability": <1-5>,
#   "Interaction Richness": <1-5>
# }}
# """

#         response = await self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt},
#             ],
#             temperature=0.0,
#             timeout=120,
#         )
#         return response.choices[0].message.content or ""










# Copyright (c) Microsoft. All rights reserved.

"""Long-short term reward evaluators for roleplay persona training.

This version is tuned for RL objectives that prioritize:
- richer and more varied interaction
- grounded, natural expression
- coherence and persona consistency
- lower tolerance for template phrasing and melodrama

Main changes:
1. Short-term action/dialogue reward uses stricter checklist scoring.
2. Short-term weights emphasize richness, non-redundancy, adaptability,
   naturalness, and anti-template behavior.
3. Long-term trajectory reward emphasizes sustained interaction richness while
   preserving coherence and persona consistency.
4. Checklist items are scored as 1 / 0 / -1 and mapped conservatively to [0, 1].
5. Rule-based penalties catch narrator-style dialogue, repetitive openers,
   stock roleplay gestures, template questions, and over-dramatic wording.
6. Final reward combines short-term and long-term signals with a short-term bias.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Mapping, Sequence

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Weight configuration
# -----------------------------------------------------------------------------

REWARD_PROFILE = os.environ.get("AGL_REWARD_PROFILE", "lsh_rl").strip().lower() or "lsh_rl"

# Short-term action evaluation. Prioritize naturalness and anti-template to avoid
# repetitive, mechanical expressions. Anti-template is most critical.
ACTION_WEIGHTS: Dict[str, float] = {
    "Character Consistency": 0.24,
    "Logical Coherence": 0.24,
    "Interaction Richness": 0.22,
    "Naturalness": 0.16,
    "Anti-template": 0.14,
}

# Short-term dialogue evaluation. Heavily penalize repetition, template expressions,
# and reward interaction adaptability and naturalness.
DIALOGUE_WEIGHTS: Dict[str, float] = {
    "Character Consistency": 0.18,
    "Logical Coherence": 0.20,
    "Content Non-redundancy": 0.24,
    "Interaction Adaptability": 0.22,
    "Naturalness": 0.16,
}

# Long-term trajectory evaluation. Prioritize interaction richness and adaptability
# to ensure varied, engaging trajectories rather than repetitive patterns.
TRAJECTORY_WEIGHTS: Dict[str, float] = {
    "Persona Consistency": 0.18,
    "Immersion": 0.10,
    "Behavioral Coherence": 0.22,
    "Adaptability": 0.22,
    "Interaction Richness": 0.28,
}

PENALTY_CONFIG: Dict[str, float] = {
    "dialogue_repetitive_penalty": 0.20,
    "dialogue_repeated_starter_penalty": 0.10,
    "dialogue_question_template_penalty": 0.08,
    "dialogue_generic_question_penalty": 0.12,
    "dialogue_recent_template_reuse_penalty": 0.14,
    "dialogue_template_penalty": 0.12,
    "dialogue_overdramatic_penalty": 0.08,
    "action_similarity_penalty": 0.16,
    "action_generic_penalty": 0.10,
    "action_template_penalty": 0.12,
    "action_overdramatic_penalty": 0.10,
    "trajectory_repeat_pair_penalty": 0.18,
    "trajectory_repeat_starter_penalty": 0.12,
    "trajectory_question_heavy_penalty": 0.10,
    "trajectory_generic_question_penalty": 0.12,
    "trajectory_template_penalty": 0.12,
    "trajectory_overdramatic_penalty": 0.10,
    "trajectory_generic_action_penalty": 0.08,
    "trajectory_narrator_dialogue_penalty": 0.16,
    "trajectory_repeat_threshold": 0.62,
    "trajectory_action_repeat_threshold": 0.70,
}

if REWARD_PROFILE == "lsh_rl":
    ACTION_WEIGHTS = {
        "Character Consistency": 0.24,
        "Naturalness": 0.36,
        "Content Non-redundancy": 0.40,
    }
    DIALOGUE_WEIGHTS = {
        "Character Consistency": 0.24,
        "Content Non-redundancy": 0.40,
        "Naturalness": 0.36,
    }
    TRAJECTORY_WEIGHTS = {
        "Persona Consistency": 0.30,
        "Behavioral Coherence": 0.40,
        "Interaction Richness": 0.30,
    }
    PENALTY_CONFIG.update(
        {
            "dialogue_repetitive_penalty": 0.28,
            "dialogue_repeated_starter_penalty": 0.22,
            "dialogue_question_template_penalty": 0.14,
            "dialogue_generic_template_penalty": 0.26,
            "dialogue_recent_template_reuse_penalty": 0.24,
            "dialogue_template_penalty": 0.24,
            "dialogue_shell_repeat_penalty": 0.34,
            "dialogue_low_increment_penalty": 0.34,
            "action_similarity_penalty": 0.22,
            "action_generic_penalty": 0.16,
            "action_template_penalty": 0.18,
            "action_shell_repeat_penalty": 0.26,
            "action_low_increment_penalty": 0.30,
            "trajectory_repeat_pair_penalty": 0.22,
            "trajectory_repeat_starter_penalty": 0.22,
            "trajectory_question_heavy_penalty": 0.18,
            "trajectory_generic_template_penalty": 0.24,
            "trajectory_template_penalty": 0.24,
            "trajectory_shell_repeat_penalty": 0.34,
            "trajectory_low_increment_penalty": 0.28,
            "trajectory_repeat_threshold": 0.58,
            "trajectory_action_repeat_threshold": 0.64,
        }
    )
    logger.info("Reward profile enabled: %s", REWARD_PROFILE)

SHORT_TERM_WEIGHT = float(os.environ.get("AGL_SHORT_TERM_WEIGHT", "0.50"))
LONG_TERM_WEIGHT = float(os.environ.get("AGL_LONG_TERM_WEIGHT", "0.50"))
LONG_TERM_BONUS_MIN_SHORT = float(os.environ.get("AGL_LONG_TERM_BONUS_MIN_SHORT", "0.0"))
LONG_TERM_BONUS_GATE_BAD_TURNS = os.environ.get("AGL_LONG_TERM_GATE_BAD_TURNS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


# -----------------------------------------------------------------------------
# Checklist definitions
# Each item should be judged as:
# 1  = clearly satisfied / exhibited
# 0  = not mentioned, unclear, or not applicable
# -1 = clearly violated
# -----------------------------------------------------------------------------

ACTION_CHECKLISTS: Dict[str, List[str]] = {
    "Character Consistency": [
        "The action demonstrates the character's unique personality traits.",
        "The action reflects the character's current emotional state.",
        "The action showcases the character's expertise or skills.",
        "The action aligns with the character's goals and motivations.",
        "The action maintains the character's social position and authority.",
        "The action uses behavior patterns specific to this character.",
        "The action is distinguishable from other characters' behaviors.",
        "The action shows character development or depth.",
        "The action fits the character's established behavior patterns.",
        "The action reveals something new about the character.",
    ],
    "Logical Coherence": [
        "The action has a clear and explicit motivation.",
        "The action creates meaningful interaction potential with others.",
        "The action follows naturally from the immediate context.",
        "The action contributes to the ongoing conversation or plot.",
        "The action bridges to the next beat in the scene.",
        "The action responds to scene pressure or other characters.",
        "The action is purposeful, not random or ornamental.",
        "The action is easily understood without explanation.",
        "The action maintains causal chain integrity.",
        "The action sets up the accompanying dialogue effectively.",
    ],
    "Interaction Richness": [
        "The action changes the social or physical state of the scene in a concrete way.",
        "The action creates a specific opening for another character to react.",
        "The action introduces a fresh angle instead of repeating a recent beat.",
        "The action uses the environment, props, or spatial relation meaningfully.",
        "The action adds new information, pressure, or opportunity to the interaction.",
        "The action contributes to scene progression rather than decorative motion.",
        "The action helps differentiate this turn from the character's previous turns.",
        "The action supports richer follow-up dialogue or conflict.",
        "The action creates believable consequences in the shared scene.",
        "The action avoids empty filler gestures that do not change anything.",
    ],
    "Naturalness": [
        "The action feels like a believable in-the-moment behavior rather than staged prose.",
        "The action uses concrete physical detail without turning into cinematic flourish.",
        "The action shows natural body language or object interaction tied to the current beat.",
        "The action balances concision with enough specificity to feel embodied.",
        "The action reads like something a person would actually do in this context.",
        "The action avoids inflated symbolic gestures that overshadow practical interaction value.",
        "The action stays grounded in everyday motion, pressure, or spatial relation.",
        "The action avoids mechanical or robotic description patterns.",
        "The action has a natural rhythm and does not feel assembled from stock roleplay parts.",
        "The action avoids generic flourish that could be pasted into many unrelated scenes.",
    ],
    "Content Non-redundancy": [
        "The action introduces a fresh interaction beat rather than replaying recent motion.",
        "The action adds new state, pressure, access, or opportunity for the next response.",
        "The action changes relation, distance, attention, or scene affordance in a meaningful way.",
        "The action avoids recycling near-identical gestures from the same speaker's recent turns.",
        "The action contributes unique context-specific detail rather than a generic filler motion.",
        "The action pushes the scene forward instead of maintaining a static loop.",
        "The action differs structurally from the character's immediately prior action shell.",
        "The action avoids repeating the same prop interaction unless it has a clear new purpose.",
        "The action creates non-trivial follow-up possibilities for other characters.",
        "The action adds a real incremental change, not just a different wording of the same move.",
    ],
    "Anti-template": [
        "The action demonstrates context-specific behavior choices, not generic roleplay defaults.",
        "The action contains unique details specific to this moment.",
        "The action varies in structure from previous actions.",
        "The action avoids generic filler gestures (nodding, looking, gazing).",
        "The action is not interchangeable with any other character.",
        "The action avoids repeated physical patterns from recent turns.",
        "The action shows fresh behavioral choices within context.",
        "The action avoids stock emotional cues without justification.",
        "The action does not rely on cinematic blocking defaults.",
        "The action uses scene-specific props or environmental features.",
    ],
}

DIALOGUE_CHECKLISTS: Dict[str, List[str]] = {
    "Character Consistency": [
        "The dialogue demonstrates the character's unique voice and style.",
        "The dialogue reflects the character's current emotional state.",
        "The dialogue showcases the character's expertise or knowledge base.",
        "The dialogue advances the character's goals in the scene.",
        "The dialogue is distinguishable from other characters' speech patterns.",
        "The dialogue maintains continuity with the character's earlier turns.",
        "The dialogue uses tone, vocabulary, and syntax fitting for this character.",
        "The dialogue reveals new facets of the character's personality.",
        "The dialogue aligns with the character's social role and position.",
        "The dialogue feels authentic to this character's background rather than a generic probing template.",
    ],
    "Logical Coherence": [
        "The dialogue responds directly and meaningfully to the immediate context.",
        "The dialogue addresses another character, event, or discovered fact.",
        "The dialogue has a clear conversational purpose and intent.",
        "The dialogue connects smoothly to the accompanying action.",
        "The dialogue advances the conversation or plot state.",
        "The dialogue maintains causal chain with previous turns.",
        "The dialogue is easily understood without extra explanation.",
        "The dialogue creates setup for future interaction or beats with concrete next-step content.",
        "The dialogue builds on what was just said or done.",
        "The dialogue contributes to scene progression through new information, decision, or constraint.",
    ],
    "Content Non-redundancy": [
        "The dialogue introduces incremental new information, perspective, or stance beyond the last two turns.",
        "The dialogue offers fresh insights rather than restating known facts with light paraphrase.",
        "The dialogue contributes novel inferences, requests, or decisions tied to current scene details.",
        "The dialogue varies from recent utterances in structure and content, not just nouns or names.",
        "The dialogue pushes the conversation forward rather than stalling in rhetorical loops.",
        "The dialogue avoids circular or repetitive conversation patterns, including repeated sentence starters or reusable stock shells.",
        "The dialogue adds unique value to the ongoing interaction that other lines did not already provide.",
        "The dialogue explores new implications grounded in prior turns, not generic side facts.",
        "The dialogue responds with originality rather than echoing the same intent in a new phrasing shell or stock statement frame.",
        "Question-form or statement-form dialogue only counts as non-redundant when it adds specific new information, commitment, boundary, reveal, choice pressure, explanation pressure, or a concrete next move.",
    ],
    "Interaction Adaptability": [
        "The dialogue responds dynamically to new information or changes with specific adaptation signals.",
        "The dialogue acknowledges and builds on other characters' contributions with explicit references.",
        "The dialogue creates opportunities for others to respond through concrete choices or unresolved tension.",
        "The dialogue adjusts to conflict, tension, or uncertainty instead of repeating a fixed interaction frame.",
        "The dialogue coordinates with the group's shared task or goal through actionable next moves.",
        "The dialogue maintains awareness of the social context and updates tone/strategy accordingly.",
        "The dialogue shifts strategy when the situation evolves, rather than staying on a single question template.",
        "The dialogue actively participates in the collaborative dynamic with role-appropriate initiative.",
        "The dialogue engages with others' concerns or questions by answering or reframing them concretely.",
        "The dialogue helps drive collaborative or conflictual progression via specific commitments or boundaries.",
    ],
    "Naturalness": [
        "The dialogue sounds like authentic spoken language.",
        "The dialogue uses conversational rhythm and pacing.",
        "The dialogue includes natural pauses, hesitations, or intensifiers.",
        "The dialogue has realistic length and complexity for the moment.",
        "The dialogue demonstrates grammatical flow and coherence.",
        "The dialogue feels like a character genuinely speaking to others.",
        "The dialogue uses appropriate colloquialisms or informal language.",
        "The dialogue captures the spontaneity of real conversation.",
        "The dialogue avoids stiff, formal, or written-style phrasing and avoids trivia-dump behavior.",
        "The dialogue fits naturally into the back-and-forth flow without relying on repetitive or low-value template turns, whether phrased as questions or statements.",
    ],
}

TRAJECTORY_CHECKLISTS: Dict[str, List[str]] = {
    "Persona Consistency": [
        "The character demonstrates stable core traits throughout the scene.",
        "The character shows believable evolution in beliefs and attitudes.",
        "The character maintains coherent and achievable goals.",
        "The character's emotional journey follows a logical arc.",
        "The character exhibits consistent expertise and competence level.",
        "The character demonstrates social behavior fitting their persona.",
        "The character remains distinguishable from other characters.",
        "The character's behavior builds on earlier turns.",
        "The character's final state emerges naturally from their journey.",
        "The character shows depth beyond a simple stereotype.",
    ],
    "Immersion": [
        "The character remains fully embodied in the scene environment.",
        "The character's dialogue and actions feel physically grounded.",
        "The character responds naturally to concrete scene details.",
        "The character maintains the scene's atmosphere and tone.",
        "The character demonstrates authentic presence in the moment.",
        "The character avoids breaking the fourth wall or meta-commentary.",
        "The character's language flows naturally and spontaneously.",
        "The character stays in-character without evaluator-style phrasing.",
        "The sequence feels like a continuous, lived experience.",
        "The character creates believable emotional and physical reality.",
    ],
    "Behavioral Coherence": [
        "The character's actions and dialogue reinforce each other.",
        "Each turn builds naturally on previous turns.",
        "The character demonstrates clear and plausible motivations.",
        "The character maintains causal chain integrity throughout.",
        "The character remembers and references important events.",
        "The character shows adaptive but consistent behavior patterns.",
        "The character's decisions lead to logical consequences.",
        "The trajectory demonstrates meaningful progression and development.",
        "The character avoids contradictory or confusing behavior.",
        "The overall behavior supports the scene's narrative arc.",
    ],
    "Adaptability": [
        "The character responds dynamically to new information.",
        "The character adjusts strategies based on others' actions.",
        "The character adapts to changing environment or circumstances.",
        "The character evolves their approach while staying in persona.",
        "The character engages with conflict and tension constructively.",
        "The character acknowledges and incorporates discoveries.",
        "The character demonstrates context-sensitive responses.",
        "The character varies their behavior based on the situation.",
        "The character helps advance the group dynamic or plot.",
        "The character handles challenges in character-appropriate ways.",
    ],
    "Interaction Richness": [
        "The character contributes varied response strategies instead of repeating one interaction shell.",
        "The character introduces meaningful new information over time.",
        "The character engages authentically with others' concerns.",
        "The interaction demonstrates progression and development.",
        "The character varies how they open, respond, or pressure the conversation across turns.",
        "The character creates opportunities for rich responses.",
        "The trajectory balances task progress with relationship dynamics.",
        "The character adds concrete scene-specific details.",
        "The character's participation enriches the scene naturally.",
        "Question-led turns only count as rich interaction when they introduce concrete new information, constraint, decision pressure, or actionable follow-up value.",
    ],
}

TRAJECTORY_HOLISTIC_METRICS: tuple[str, ...] = (
    "Personality Traits",
    "Behavioral Coherence",
    "Interaction Richness",
)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def _normalize_base_url(base_url: str) -> str:
    """Normalize base URL to ensure it ends with /v1."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("Empty base_url is not allowed.")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def is_long_term_bonus_eligible(short_reward: float) -> bool:
    """Return whether a turn is eligible to receive long-term reward bonus."""
    if not LONG_TERM_BONUS_GATE_BAD_TURNS:
        return True
    return _clip01(short_reward) >= LONG_TERM_BONUS_MIN_SHORT


def combine_short_long_rewards(
    short_reward: float,
    long_reward: float,
    *,
    eligible_for_long_term_bonus: bool | None = None,
) -> float:
    """Combine short-term and long-term rewards with optional long-term bonus gating."""
    short = _clip01(short_reward)
    long = _clip01(long_reward)
    eligible = is_long_term_bonus_eligible(short) if eligible_for_long_term_bonus is None else eligible_for_long_term_bonus
    if not eligible:
        return short
    return _clip01(SHORT_TERM_WEIGHT * short + LONG_TERM_WEIGHT * long)


def _score_item_guidance(scope: str) -> str:
    """Return profile-specific checklist scoring guidance."""
    if REWARD_PROFILE != "lsh_rl":
        return (
            f"Scoring rule for EACH checklist item:\n"
            f"- Use 1 only if the current {scope} gives explicit, concrete evidence for that item.\n"
            f"- Use 0 if the evidence is weak, ambiguous, missing, or not applicable.\n"
            f"- Use -1 if the {scope} clearly violates the item or shows the opposite tendency.\n"
            "Important: JSON does not allow +1. Positive scores must be written as 1, not +1."
        )
    return (
        f"Scoring rule for EACH checklist item:\n"
        f"- Use 1 only if the current {scope} clearly changes the social state, decision state, relationship state, "
        "or interaction options in a way that is directly observable in the text.\n"
        f"- Use 0 if support is partial, weak, generic, or present only as named details without meaningful interaction effect.\n"
        f"- Use -1 if the {scope} clearly violates the item, falls into a repeated shell, or reads like evaluator-facing filler.\n"
        "Important: JSON does not allow +1. Positive scores must be written as 1, not +1.\n"
        "For LSH-RL, prefer social effect, commitment, boundary, reveal, decision pressure, or grounded response value over "
        "evidence-style detail accumulation."
    )


def _strip_json_fence(text: str) -> str:
    """Strip common JSON code fences from LLM response."""
    content = (text or "").strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    return content.strip()


def _extract_json_object(text: str) -> str:
    """Extract and lightly repair the first JSON object from an LLM response."""
    content = _strip_json_fence(text)
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start : end + 1]

    # JSON does not allow +1, but LLMs often emit it when the rubric says +1.
    content = re.sub(r"(?<=[:\[,])\s*\+1\b", " 1", content)
    # Remove trailing commas.
    content = re.sub(r",\s*([}\]])", r"\1", content)
    # Sometimes models use Unicode minus.
    content = content.replace("−1", "-1")
    return content.strip()


def _history_to_text(history: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    """Convert history to text format for evaluation context."""
    lines: List[str] = []
    total_chars = 0
    for turn in reversed(history):
        turn_type = turn.get("type", "dialogue")
        speaker = turn.get("speaker", "")
        text = turn.get("utterance", turn.get("text", ""))
        if turn_type == "action":
            line = f"{speaker} [ACTION]: {text}"
        else:
            line = f"{speaker}: {text}"
        lines.insert(0, line)
        total_chars += len(line)
        if max_chars > 0 and total_chars > max_chars:
            break
    return "\n".join(lines[-10:])


def _safe_character_name(character: Mapping[str, Any]) -> str:
    return str(character.get("name", "")).strip()


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"^[\"'“”]+|[\"'“”]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


CONTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "too",
    "up",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


# -----------------------------------------------------------------------------
# Checklist parsing and scoring
# -----------------------------------------------------------------------------

def _neutral_checklist_scores(checklists: Mapping[str, Sequence[str]]) -> Dict[str, List[int]]:
    return {key: [0 for _ in items] for key, items in checklists.items()}


def _active_checklists(
    checklists: Mapping[str, Sequence[str]],
    weights: Mapping[str, float],
) -> Dict[str, Sequence[str]]:
    """Return only the checklist dimensions that participate in the current profile."""
    return {key: list(checklists[key]) for key in checklists if key in weights}


def _coerce_checklist_values(value: Any, expected_len: int) -> List[int]:
    """Coerce model output into a fixed-length list of -1/0/1 values."""
    raw_values: List[Any]

    if isinstance(value, list):
        raw_values = value
    elif isinstance(value, dict):
        def sort_key(k: Any) -> Any:
            m = re.search(r"\d+", str(k))
            return int(m.group(0)) if m else str(k)
        raw_values = [value[k] for k in sorted(value.keys(), key=sort_key)]
    elif isinstance(value, str):
        # Accept strings like "1, 0, -1, ..." as fallback.
        raw_values = re.findall(r"[-+]?1|0", value)
    else:
        raw_values = []

    coerced: List[int] = []
    for item in raw_values[:expected_len]:
        if isinstance(item, dict):
            item = item.get("score", item.get("value", item.get("judgment", 0)))
        if isinstance(item, str):
            lowered = item.strip().lower()
            if lowered in {"+1", "1", "satisfied", "yes", "true", "met"}:
                score = 1
            elif lowered in {"-1", "violated", "no", "false", "unmet"}:
                score = -1
            else:
                score = 0
        else:
            try:
                number = int(float(item))
                score = 1 if number > 0 else -1 if number < 0 else 0
            except Exception:
                score = 0
        coerced.append(max(-1, min(1, score)))

    while len(coerced) < expected_len:
        coerced.append(0)
    return coerced


def _parse_checklist_response(response: str, checklists: Mapping[str, Sequence[str]]) -> Dict[str, List[int]]:
    """Parse checklist JSON with robust fallbacks.

    Expected primary format:
    {
      "Dimension": [1, 0, -1, ...]
    }

    Also accepts:
    {
      "Dimension": {"1": 1, "2": 0, ...}
    }
    """
    try:
        obj = json.loads(_extract_json_object(response))
        if not isinstance(obj, dict):
            raise ValueError("Checklist response root is not a JSON object")

        parsed: Dict[str, List[int]] = {}
        for key, items in checklists.items():
            source_key = key
            if source_key not in obj:
                source_key = ""
                for obj_key in obj.keys():
                    if key.lower() == str(obj_key).lower():
                        source_key = str(obj_key)
                        break
                    if key.lower() in str(obj_key).lower() or str(obj_key).lower() in key.lower():
                        source_key = str(obj_key)
                        break
            parsed[key] = _coerce_checklist_values(obj.get(source_key, []), len(items)) if source_key else [0] * len(items)
        return parsed
    except Exception as e:
        logger.warning(
            "Failed to parse checklist JSON response: %s; raw response:\n%s",
            e,
            response,
        )
        return _neutral_checklist_scores(checklists)


def _coerce_1to5(value: Any) -> int:
    try:
        if isinstance(value, bool):
            score = int(value)
        else:
            score = int(float(value))
    except Exception:
        return 3
    return max(1, min(5, score))


def _parse_holistic_metric_scores(response: str, metrics: Sequence[str]) -> Dict[str, int]:
    """Parse holistic 1-5 trajectory scores with robust fallbacks."""
    fallback = {metric: 3 for metric in metrics}
    try:
        obj = json.loads(_extract_json_object(response))
        if not isinstance(obj, dict):
            return fallback
        out: Dict[str, int] = {}
        for metric in metrics:
            source_key = metric
            if source_key not in obj:
                source_key = ""
                for obj_key in obj.keys():
                    if metric.lower() == str(obj_key).lower():
                        source_key = str(obj_key)
                        break
            out[metric] = _coerce_1to5(obj.get(source_key, 3)) if source_key else 3
        return out
    except Exception as e:
        logger.warning(
            "Failed to parse holistic trajectory JSON response: %s; raw response:\n%s",
            e,
            response,
        )
        return fallback


def _normalize_holistic_metric_scores(scores: Mapping[str, int]) -> float:
    """Normalize 1-5 holistic scores into [0, 1]."""
    values = [_coerce_1to5(scores.get(metric, 3)) for metric in TRAJECTORY_HOLISTIC_METRICS]
    if not values:
        return 0.5
    normalized = [(_coerce_1to5(value) - 1) / 4 for value in values]
    return _clip01(float(sum(normalized) / len(normalized)))


def _score_checklists(parsed: Mapping[str, Sequence[int]], weights: Mapping[str, float]) -> float:
    """Map checklist scores to [0, 1] and aggregate by weights.

    The mapping is intentionally conservative:
    - 1 means there is explicit evidence in the text
    - 0 means the evidence is weak, missing, or ambiguous
    - -1 means the text clearly violates the criterion

    For a dimension with 10 checklist items:
    - All +1 → score = 1.0
    - All 0 → score = 0.05
    - All -1 → score = 0.0

    Negative evidence is weighted more strongly than positive evidence to avoid
    reward saturation and to make repetitive or immersion-breaking behavior
    materially costly during RL.
    """
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.05

    weighted = 0.0
    for key, weight in weights.items():
        values = list(parsed.get(key, []))
        if not values:
            dim_score = 0.05
        else:
            n = len(values)
            n_positive = sum(1 for v in values if v == 1)
            n_negative = sum(1 for v in values if v == -1)
            n_neutral = n - n_positive - n_negative
            dim_score = (1.00 * n_positive + 0.05 * n_neutral - 1.25 * n_negative) / n
            dim_score = _clip01(dim_score)
        weighted += dim_score * weight
    return _clip01(weighted / total_weight)


# -----------------------------------------------------------------------------
# Rule-based penalties and fallback reward
# -----------------------------------------------------------------------------

def _contains_unexpected_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _has_character_name_prefix(character_name: str, dialogue: str) -> bool:
    text = (dialogue or "").strip().strip('"“”')
    if not character_name:
        return False
    full = re.escape(character_name)
    first = re.escape(character_name.split()[0])
    return bool(re.match(rf"^({full}|{first})\s*[:：]", text, flags=re.IGNORECASE))


def _is_third_person_narration_dialogue(character_name: str, dialogue: str) -> bool:
    text = _normalize_text(dialogue)
    if not text:
        return False

    first = re.escape(character_name.split()[0].lower()) if character_name else r"[a-z]+"
    verbs = (
        "notices|realizes|thinks|feels|prepares|sets|starts|begins|continues|"
        "moves|reaches|adjusts|looks|turns|steps|walks|pulls|types|configures|"
        "discovers|decides|tries|attempts|offers|asks|says"
    )
    patterns = [
        rf"^{first}\s+({verbs})\b",
        rf"^(he|she|they)\s+({verbs})\b",
        r"\bnext,\s+(he|she|they)\b",
        r"\bthe logs show\b.*\bnext,\b",
        r"\bsuggesting that\b.*\bnext,\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _has_template_expression(text: str) -> bool:
    normalized = _normalize_text(text)
    if _dialogue_shell(text) in {
        "you_question",
        "have_you_ever",
        "what_if",
        "i_wonder",
        "you_know",
        "wh_question",
        "formal_request",
        "fact_update",
        "note_logging",
    }:
        return True
    patterns = [
        r"\bdid you know\b",
        r"\bdo you know why\b",
        r"\bdo you remember\b",
        r"\bdid you check\b",
        r"\bdid you notice\b",
        r"\bhave you tried\b",
        r"\bi understand how you feel\b",
        r"\bthank you for sharing\b",
        r"\bas an ai\b",
        r"\bi'm here to help\b",
        r"\bwe need to work together\b",
        r"\bnext,\s+(he|she|they)\b",
        r"\bgaze lingers\b",
        r"\bbreath hitch(?:es|ing)\b",
        r"\bslow, measured breath\b",
        r"\bfingers? brush(?:ing)? the edge\b",
        r"\bknuckles? tighten(?:ing)?\b",
    ]
    return any(re.search(p, normalized) for p in patterns)


def _is_overdramatic(text: str) -> bool:
    normalized = _normalize_text(text)
    patterns = [
        r"\beverything i've built\b",
        r"\bthey're watching\b",
        r"\bgetting closer\b",
        r"\bin the silence\b",
        r"\bin the dark\b",
        r"\bmy life\b",
        r"\bdoomed\b|\bcatastrophic\b|\bterrifying\b",
    ]
    return any(re.search(p, normalized) for p in patterns)


def _recent_dialogue_texts(history: Sequence[Mapping[str, Any]], limit: int = 8) -> List[str]:
    out: List[str] = []
    for turn in list(history)[-limit:]:
        turn_type = str(turn.get("type", "dialogue"))
        if turn_type == "action":
            continue
        text = str(turn.get("utterance", turn.get("text", ""))).strip()
        if text:
            out.append(text)
    return out


def _recent_action_texts(history: Sequence[Mapping[str, Any]], limit: int = 8) -> List[str]:
    out: List[str] = []
    for turn in list(history)[-limit:]:
        turn_type = str(turn.get("type", "dialogue"))
        if turn_type != "action":
            continue
        text = str(turn.get("utterance", turn.get("text", ""))).strip()
        if text:
            out.append(text)
    return out


def _recent_turn_texts(
    history: Sequence[Mapping[str, Any]],
    *,
    turn_type: str,
    limit: int = 8,
    speaker: str = "",
) -> List[str]:
    out: List[str] = []
    speaker_norm = _normalize_text(speaker)
    for turn in list(history)[-max(limit * 3, limit):]:
        curr_type = str(turn.get("type", "dialogue"))
        if curr_type != turn_type:
            continue
        if speaker_norm:
            turn_speaker = _normalize_text(str(turn.get("speaker", turn.get("name", ""))))
            if turn_speaker and turn_speaker != speaker_norm:
                continue
        text = str(turn.get("utterance", turn.get("text", ""))).strip()
        if text:
            out.append(text)
    return out[-limit:]


def _history_round_index(turn: Mapping[str, Any]) -> int | None:
    raw = turn.get("round")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _recent_round_turn_texts(
    history: Sequence[Mapping[str, Any]],
    *,
    turn_type: str,
    rounds: int = 2,
    limit: int = 8,
) -> List[str]:
    round_ids = [
        round_idx
        for turn in history
        if str(turn.get("type", "dialogue")) == turn_type and (round_idx := _history_round_index(turn)) is not None
    ]
    if not round_ids:
        if turn_type == "action":
            return _recent_action_texts(history, limit=limit)
        return _recent_dialogue_texts(history, limit=limit)

    recent_rounds = set(sorted(set(round_ids))[-max(rounds, 1) :])
    out: List[str] = []
    for turn in history:
        if str(turn.get("type", "dialogue")) != turn_type:
            continue
        if _history_round_index(turn) not in recent_rounds:
            continue
        text = str(turn.get("utterance", turn.get("text", ""))).strip()
        if text:
            out.append(text)
    return out[-limit:]


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}", _normalize_text(text))
    return {token for token in tokens if token not in CONTENT_STOPWORDS}


def _turn_has_value_signal(text: str, previous_texts: Sequence[str]) -> bool:
    normalized = _normalize_text(text)
    current_tokens = _content_tokens(text)
    if not normalized or not current_tokens:
        return False

    prior_tokens: set[str] = set()
    for prev in previous_texts:
        prior_tokens |= _content_tokens(prev)

    context_anchor = len(current_tokens & prior_tokens) >= 2
    explicit_choice = " or " in normalized and len(current_tokens) >= 5
    explicit_constraint = any(
        marker in normalized
        for marker in (" if ", " unless ", " instead ", " before ", " after ", " because ", " so that ")
    )
    explicit_action_pressure = bool(
        re.search(
            r"\b(tell|show|explain|confirm|admit|bring|call|meet|leave|stay|sign|pay|help|stop|start|change|fix|prove|choose|decide|remember|hide|wait|check)\b",
            normalized,
        )
    )
    explicit_commitment_or_reveal = bool(
        re.search(
            r"\b(i will|i won't|i can|i can't|i need|i need you to|i want|i don't want|i'm going to|i have to|"
            r"i should|i shouldn't|let's|we should|we need to|the truth is|i need to tell you|i should tell you|"
            r"i admit|i realized|i've decided|i'm afraid|i'm not ready|i'm ready)\b",
            normalized,
        )
    )
    return context_anchor or explicit_choice or explicit_constraint or explicit_action_pressure or explicit_commitment_or_reveal


def _is_low_value_question(text: str, previous_texts: Sequence[str]) -> bool:
    normalized = _normalize_text(text)
    if "?" not in text or not normalized:
        return False

    shell = _dialogue_shell(text)
    if shell not in {"you_question", "have_you_ever", "what_if", "i_wonder", "you_know", "wh_question"}:
        return False

    if _turn_has_value_signal(text, previous_texts):
        if shell == "wh_question":
            return False
        if shell in {"what_if", "i_wonder"} and len(_content_tokens(text)) >= 5:
            return False
        if shell in {"you_question", "have_you_ever", "you_know"} and any(
            marker in normalized for marker in (" if ", " because ", " instead ", " before ", " after ", " or ")
        ):
            return False

    current_tokens = _content_tokens(text)
    if len(current_tokens) <= 3:
        return True

    if shell in {"you_question", "have_you_ever", "you_know"}:
        return True

    return not _turn_has_value_signal(text, previous_texts)


def _is_low_value_template_turn(text: str, previous_texts: Sequence[str]) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    shell = _dialogue_shell(text)
    current_tokens = _content_tokens(text)
    repeated_starter_hit = _repeated_starter_count([*previous_texts, text]) >= 1
    shell_repeat_hit = bool(shell) and _repeated_shell_count([*previous_texts, text], _dialogue_shell) >= 1
    low_increment_hit = _low_increment_against_history(text, previous_texts, shell_fn=_dialogue_shell)

    if len(current_tokens) <= 3:
        return True

    if shell in {"formal_request", "fact_update", "note_logging"}:
        if not _turn_has_value_signal(text, previous_texts):
            return True
        current_overlap = 0.0
        if current_tokens:
            prior_tokens: set[str] = set()
            for prev in previous_texts:
                prior_tokens |= _content_tokens(prev)
            current_overlap = len(current_tokens & prior_tokens) / max(len(current_tokens), 1)
        return low_increment_hit or current_overlap >= 0.6

    if _is_low_value_question(text, previous_texts):
        return True

    if (shell_repeat_hit or repeated_starter_hit) and low_increment_hit:
        return not _turn_has_value_signal(text, previous_texts)

    if repeated_starter_hit and not _turn_has_value_signal(text, previous_texts):
        return True

    return False


def _dialogue_shell(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    shell_patterns = [
        (r"^(did|do|have|are|can|could|would|will|should)\s+you\b", "you_question"),
        (r"^have you ever\b", "have_you_ever"),
        (r"^what if\b", "what_if"),
        (r"^i wonder(?: if)?\b", "i_wonder"),
        (r"^you know\b", "you_know"),
        (r"^(why|how|what|when|where)\b.*\?$", "wh_question"),
        (r"^i(?:'d| would) like to (?:formally )?request\b", "formal_request"),
        (r"^i just (checked|called|found out|learned|confirmed)\b", "fact_update"),
        (r"^(?:[a-z]+ [a-z]+ )?(writes|adds)\b.*\b(notebook|notes)\b", "note_logging"),
    ]
    for pattern, label in shell_patterns:
        if re.search(pattern, normalized):
            return label
    return ""


def _dialogue_template_family(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    shell = _dialogue_shell(text)
    if shell:
        return shell
    family_patterns = [
        (
            r"^(?:[a-z]+,\s+)?if\s+(?:you're|you are|we're|we are)\s+going\s+to\b.*\b(?:need to|have to|let's)\b",
            "conditional_guidance",
        ),
        (r"^(?:[a-z]+,\s+)?(?:i|we)\s+need\s+you\s+to\b", "directive_need"),
        (r"^(?:[a-z]+,\s+)?let'?s\s+make\s+sure\b", "alignment_check"),
        (r"^(?:[a-z]+,\s+)?i\s+need\s+to\s+make\s+sure\b", "alignment_check"),
        (r"^it(?:'s| is)\s+not\s+just\s+about\b", "reframing_statement"),
        (r"\bon\s+the\s+same\s+page\b", "alignment_check"),
        (r"\bstay\s+calm\s+and\s+listen\b", "soothing_directive"),
        (r"\btake\s+a\s+moment\s+to\b", "soothing_directive"),
        (r"\brest\s+up\b", "soothing_directive"),
        (r"\bbe\s+present\b", "soothing_directive"),
    ]
    for pattern, label in family_patterns:
        if re.search(pattern, normalized):
            return label
    return ""


def _is_recent_semantic_template_reuse(
    text: str,
    history: Sequence[Mapping[str, Any]],
    *,
    rounds: int = 2,
) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False

    recent_texts = _recent_round_turn_texts(history, turn_type="dialogue", rounds=rounds, limit=10)
    if not recent_texts:
        return False

    current_shell = _dialogue_shell(text)
    current_family = _dialogue_template_family(text)
    current_tokens = _content_tokens(text)
    current_template_like = (
        bool(current_family)
        or _is_question_heavy_template(text)
        or _has_template_expression(text)
        or _is_low_value_template_turn(text, recent_texts)
    )
    if not current_template_like:
        return False

    for prev in recent_texts:
        prev_norm = _normalize_text(prev)
        if not prev_norm:
            continue
        prev_shell = _dialogue_shell(prev)
        prev_family = _dialogue_template_family(prev)
        prev_tokens = _content_tokens(prev)
        prev_template_like = bool(prev_family) or _is_question_heavy_template(prev) or _has_template_expression(prev)
        if not prev_template_like:
            continue

        same_shell = bool(current_shell) and current_shell == prev_shell
        same_family = bool(current_family) and current_family == prev_family
        lexical_overlap = _jaccard_similarity(normalized, prev_norm)
        low_new_content = bool(current_tokens) and (
            len(current_tokens - prev_tokens) / max(len(current_tokens), 1) <= 0.45
        )
        if same_family and (lexical_overlap >= 0.32 or low_new_content):
            return True
        if same_shell and (lexical_overlap >= 0.38 or low_new_content):
            return True
        if (same_shell or same_family) and _low_increment_against_history(text, [prev], shell_fn=_dialogue_shell):
            return True
    return False


def _action_shell(text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""
    shell_patterns = [
        (r"\b(looks?|glances?|gazes?|stares?)\b", "look_gesture"),
        (r"\b(nods?|shrugs?|sighs?|exhales?|pauses?)\b", "micro_reaction"),
        (r"\b(steps?|walks?|turns?|leans?|shifts?)\b", "reposition"),
        (r"\b(reaches?|grabs?|picks?|places?|sets?)\b", "reach_and_place"),
        (r"\b(brushes?|rests?|tightens?|taps?)\b", "touch_gesture"),
        (r"\b(smiles?|frowns?)\b", "facial_cue"),
    ]
    for pattern, label in shell_patterns:
        if re.search(pattern, normalized):
            return label
    return ""


def _sentence_starter(text: str, max_words: int = 3) -> str:
    tokens = re.findall(r"[a-zA-Z']+", _normalize_text(text))
    return " ".join(tokens[:max_words])


def _repeated_starter_count(texts: Sequence[str]) -> int:
    seen: Dict[str, int] = {}
    repeated = 0
    for text in texts:
        starter = _sentence_starter(text)
        if not starter:
            continue
        seen[starter] = seen.get(starter, 0) + 1
        if seen[starter] >= 2:
            repeated += 1
    return repeated


def _repeated_shell_count(texts: Sequence[str], shell_fn: Any) -> int:
    seen: Dict[str, int] = {}
    repeated = 0
    for text in texts:
        shell = str(shell_fn(text))
        if not shell:
            continue
        seen[shell] = seen.get(shell, 0) + 1
        if seen[shell] >= 2:
            repeated += 1
    return repeated


def _low_increment_against_history(
    text: str,
    previous_texts: Sequence[str],
    *,
    shell_fn: Any,
) -> bool:
    current_norm = _normalize_text(text)
    current_tokens = _content_tokens(text)
    current_shell = str(shell_fn(text))
    if not current_norm or not previous_texts:
        return False

    for prev in previous_texts:
        prev_norm = _normalize_text(prev)
        prev_tokens = _content_tokens(prev)
        same_shell = bool(current_shell) and current_shell == str(shell_fn(prev))
        if current_norm == prev_norm:
            return True
        if same_shell and len(current_tokens) <= 2:
            return True
        if same_shell and current_tokens:
            new_ratio = len(current_tokens - prev_tokens) / max(len(current_tokens), 1)
            overlap_ratio = len(current_tokens & prev_tokens) / max(len(current_tokens), 1)
            if new_ratio <= 0.45 or overlap_ratio >= 0.65:
                return True
        if same_shell and _jaccard_similarity(current_norm, prev_norm) >= 0.45:
            return True
    return False


def _is_question_heavy_template(text: str) -> bool:
    normalized = _normalize_text(text)
    if "?" not in text:
        return False
    if _dialogue_shell(text) in {"you_question", "have_you_ever", "what_if", "i_wonder", "you_know", "wh_question"}:
        return True
    patterns = [
        r"^(did|do|have|are|can|will)\s+you\b",
        r"^(why|how|what|when)\b",
    ]
    return any(re.search(p, normalized) for p in patterns)


def _is_generic_action(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    generic_patterns = [
        r"\b(nods?|glances?|looks?|gazes?|smiles?|sighs?|shrugs?|pauses?|exhales?|frowns?)\b",
        r"\b(steps|turns|walks)\s+(forward|back|away|toward|towards)\b",
        r"\b(brushes?|rests?|tightens?)\b",
    ]
    has_generic = any(re.search(p, normalized) for p in generic_patterns)
    has_scene_anchor = bool(
        re.search(
            r"\b(table|door|screen|monitor|phone|cup|mug|chair|window|bag|file|log|keyboard|counter|hallway|sofa)\b",
            normalized,
        )
    )
    return has_generic and not has_scene_anchor


def _jaccard_similarity(a: str, b: str) -> float:
    at = set(re.findall(r"[a-zA-Z0-9_']+", _normalize_text(a)))
    bt = set(re.findall(r"[a-zA-Z0-9_']+", _normalize_text(b)))
    if not at or not bt:
        return 0.0
    return len(at & bt) / len(at | bt)


def _is_repetitive(
    text: str,
    history: Sequence[Mapping[str, Any]],
    *,
    speaker: str = "",
    turn_type: str = "dialogue",
) -> bool:
    """Check if text is repetitive compared to recent dialogue history.

    Uses Jaccard similarity threshold of 0.65 to catch semantic repetition
    even when wording differs slightly.
    """
    current = _normalize_text(text)
    if not current:
        return False
    if turn_type == "action":
        previous_texts = _recent_turn_texts(history, turn_type="action", limit=8, speaker=speaker) or _recent_action_texts(history)
    else:
        previous_texts = _recent_turn_texts(history, turn_type="dialogue", limit=8, speaker=speaker) or _recent_dialogue_texts(history)
    for prev in previous_texts:
        prev_norm = _normalize_text(prev)
        if not prev_norm:
            continue
        if current == prev_norm:
            return True
        if len(current) > 24 and (current in prev_norm or prev_norm in current):
            return True
        if _jaccard_similarity(current, prev_norm) >= 0.65:
            return True
    return False


def _has_repeated_starter(
    text: str,
    history: Sequence[Mapping[str, Any]],
    *,
    limit: int = 8,
    speaker: str = "",
) -> bool:
    starter = _sentence_starter(text)
    if not starter:
        return False
    previous_texts = _recent_turn_texts(history, turn_type="dialogue", limit=limit, speaker=speaker) or _recent_dialogue_texts(
        history,
        limit=limit,
    )
    return any(_sentence_starter(prev) == starter for prev in previous_texts)


def _trajectory_repetition_count(texts: Sequence[str], *, threshold: float = 0.65) -> int:
    normalized = [_normalize_text(t) for t in texts if _normalize_text(t)]
    matches = 0
    for i, text in enumerate(normalized):
        for prev in normalized[:i]:
            if text == prev or _jaccard_similarity(text, prev) >= threshold:
                matches += 1
                break
    return matches


def _trajectory_repeated_starter_count(texts: Sequence[str]) -> int:
    return _repeated_starter_count(texts)


def _apply_dialogue_hard_penalties(
    reward: float,
    *,
    character_name: str,
    dialogue: str,
    history: Sequence[Mapping[str, Any]],
) -> float:
    """Apply hard caps/penalties for severe dialogue-format failures.

    Penalties are cumulative: each violation adds a fixed penalty.
    """
    reward = _clip01(reward)
    text = dialogue or ""

    # Base penalties for format issues
    if not text.strip():
        return 0.20
    if len(text.strip()) < 4:
        reward = min(reward, 0.35)

    # Accumulate penalties for each violation
    penalty = 0.0

    # Severe: dialogue is actually narrator-style synopsis.
    if _is_third_person_narration_dialogue(character_name, text):
        if REWARD_PROFILE == "lsh_rl":
            return min(reward, 0.10)
        penalty += 0.25

    # Medium: script-style prefix. This can be acceptable in some datasets, but
    # if the target is pure utterance, cap it.
    if _has_character_name_prefix(character_name, text):
        if REWARD_PROFILE == "lsh_rl":
            reward = min(reward, 0.20)
        penalty += 0.10

    if _contains_unexpected_cjk(text):
        penalty += 0.15

    recent_same_speaker = _recent_turn_texts(history, turn_type="dialogue", limit=6, speaker=character_name)
    recent_dialogue_context = _recent_dialogue_texts(history, limit=10)
    if _is_repetitive(text, history, speaker=character_name, turn_type="dialogue"):
        penalty += PENALTY_CONFIG["dialogue_repetitive_penalty"]

    if _has_repeated_starter(text, history, speaker=character_name):
        penalty += PENALTY_CONFIG["dialogue_repeated_starter_penalty"]

    if _is_question_heavy_template(text):
        penalty += PENALTY_CONFIG["dialogue_question_template_penalty"]

    if _is_low_value_template_turn(text, recent_dialogue_context):
        penalty += PENALTY_CONFIG.get(
            "dialogue_generic_template_penalty",
            PENALTY_CONFIG.get("dialogue_generic_question_penalty", 0.0),
        )

    if _is_recent_semantic_template_reuse(text, history, rounds=2):
        penalty += PENALTY_CONFIG.get("dialogue_recent_template_reuse_penalty", 0.0)

    if _has_template_expression(text):
        penalty += PENALTY_CONFIG["dialogue_template_penalty"]

    if _repeated_shell_count([*recent_same_speaker, text], _dialogue_shell) >= 1:
        penalty += PENALTY_CONFIG.get("dialogue_shell_repeat_penalty", 0.0)

    if _low_increment_against_history(text, recent_same_speaker, shell_fn=_dialogue_shell):
        penalty += PENALTY_CONFIG.get("dialogue_low_increment_penalty", 0.0)

    if _is_overdramatic(text):
        penalty += PENALTY_CONFIG["dialogue_overdramatic_penalty"]

    # Apply cumulative penalty
    reward = _clip01(reward - penalty)

    return reward


def _apply_action_hard_penalties(
    reward: float,
    *,
    character_name: str,
    action: str,
    history: Sequence[Mapping[str, Any]],
) -> float:
    """Apply hard caps/penalties for severe action-format failures.

    Penalties are cumulative: each violation adds a fixed penalty.
    """
    reward = _clip01(reward)
    text = action or ""

    # Base penalties for format issues
    if not text.strip():
        return 0.20
    if len(text.strip()) < 8:
        reward = min(reward, 0.45)

    # Accumulate penalties for each violation
    penalty = 0.0

    if _contains_unexpected_cjk(text):
        penalty += 0.15

    curr_norm = _normalize_text(text)
    recent_same_speaker = _recent_turn_texts(history, turn_type="action", limit=6, speaker=character_name)
    recent_action_texts = recent_same_speaker or _recent_action_texts(history)
    for prev in recent_action_texts:
        prev_norm = _normalize_text(prev)
        if curr_norm and prev_norm and (
            curr_norm == prev_norm
            or _jaccard_similarity(curr_norm, prev_norm) >= PENALTY_CONFIG["trajectory_action_repeat_threshold"]
        ):
            penalty += PENALTY_CONFIG["action_similarity_penalty"]
            break

    if _is_generic_action(text):
        penalty += PENALTY_CONFIG["action_generic_penalty"]

    if _has_template_expression(text):
        penalty += PENALTY_CONFIG["action_template_penalty"]

    if _repeated_shell_count([*recent_action_texts, text], _action_shell) >= 1:
        penalty += PENALTY_CONFIG.get("action_shell_repeat_penalty", 0.0)

    if _low_increment_against_history(text, recent_action_texts, shell_fn=_action_shell):
        penalty += PENALTY_CONFIG.get("action_low_increment_penalty", 0.0)

    if _is_overdramatic(text):
        penalty += PENALTY_CONFIG["action_overdramatic_penalty"]

    # Apply cumulative penalty
    reward = _clip01(reward - penalty)

    return _clip01(reward)


def _fallback_dialogue_reward(character_name: str, dialogue: str, history: Sequence[Mapping[str, Any]]) -> float:
    """Rule-based fallback when the LLM evaluator fails or times out."""
    reward = 0.40
    return _apply_dialogue_hard_penalties(reward, character_name=character_name, dialogue=dialogue, history=history)


def _fallback_action_reward(character_name: str, action: str, history: Sequence[Mapping[str, Any]]) -> float:
    reward = 0.40
    return _apply_action_hard_penalties(reward, character_name=character_name, action=action, history=history)


def _fallback_trajectory_reward(trajectory: Sequence[Mapping[str, Any]]) -> float:
    texts = [str(t.get("text", t.get("utterance", ""))) for t in trajectory]
    combined = "\n".join(texts)
    reward = 0.40
    if _contains_unexpected_cjk(combined):
        reward = min(reward, 0.35)
    if any(_has_template_expression(t) for t in texts):
        reward = min(reward, 0.38)
    # Detect repeated turns inside trajectory.
    normalized = [_normalize_text(t) for t in texts if _normalize_text(t)]
    for i, t in enumerate(normalized):
        for prev in normalized[:i]:
            if t == prev or _jaccard_similarity(t, prev) >= 0.78:
                reward = min(reward, 0.30)
                break
    return _clip01(reward)


def _apply_trajectory_hard_penalties(reward: float, *, trajectory: Sequence[Mapping[str, Any]]) -> float:
    """Apply long-term penalties for repetitive, template-like scene behavior."""
    reward = _clip01(reward)
    texts = [str(t.get("text", t.get("utterance", ""))) for t in trajectory]
    dialogue_texts = [str(t.get("text", t.get("utterance", ""))) for t in trajectory if str(t.get("type", "")) != "action"]
    action_texts = [str(t.get("text", t.get("utterance", ""))) for t in trajectory if str(t.get("type", "")) == "action"]

    if REWARD_PROFILE == "lsh_rl":
        generic_template_count = sum(
            1
            for idx, text in enumerate(dialogue_texts)
            if _is_low_value_template_turn(text, dialogue_texts[:idx])
        )
        narrator_dialogue_count = sum(
            1
            for turn in trajectory
            if str(turn.get("type", "")) != "action"
            and _is_third_person_narration_dialogue("", str(turn.get("text", turn.get("utterance", ""))))
        )
        shell_repeat_count = _repeated_shell_count(dialogue_texts, _dialogue_shell)
        action_shell_repeat_count = _repeated_shell_count(action_texts, _action_shell)

        light_penalty = 0.0
        if shell_repeat_count >= 3 or action_shell_repeat_count >= 3:
            reward = min(reward, 0.55)
            light_penalty += 0.10
        if generic_template_count >= 3:
            reward = min(reward, 0.58)
            light_penalty += 0.08
        if narrator_dialogue_count >= 2:
            reward = min(reward, 0.50)
            light_penalty += 0.12
        if _trajectory_repetition_count(dialogue_texts, threshold=0.72) >= 2:
            reward = min(reward, 0.52)
            light_penalty += 0.08
        return _clip01(reward - light_penalty)

    penalty = 0.0

    if _trajectory_repetition_count(dialogue_texts, threshold=PENALTY_CONFIG["trajectory_repeat_threshold"]) >= 2:
        penalty += PENALTY_CONFIG["trajectory_repeat_pair_penalty"]
        reward = min(reward, 0.68)

    if _trajectory_repeated_starter_count(dialogue_texts) >= 2:
        penalty += PENALTY_CONFIG["trajectory_repeat_starter_penalty"]
        reward = min(reward, 0.70)

    if _repeated_shell_count(dialogue_texts, _dialogue_shell) >= 2 or _repeated_shell_count(action_texts, _action_shell) >= 2:
        penalty += PENALTY_CONFIG.get("trajectory_shell_repeat_penalty", 0.0)
        reward = min(reward, 0.62)

    if any(
        _low_increment_against_history(text, dialogue_texts[:idx], shell_fn=_dialogue_shell)
        for idx, text in enumerate(dialogue_texts)
        if idx > 0
    ) or any(
        _low_increment_against_history(text, action_texts[:idx], shell_fn=_action_shell)
        for idx, text in enumerate(action_texts)
        if idx > 0
    ):
        penalty += PENALTY_CONFIG.get("trajectory_low_increment_penalty", 0.0)
        reward = min(reward, 0.66)

    question_heavy_count = sum(1 for text in dialogue_texts if _is_question_heavy_template(text))
    if dialogue_texts and question_heavy_count / len(dialogue_texts) >= 0.6:
        penalty += PENALTY_CONFIG["trajectory_question_heavy_penalty"]

    generic_template_count = sum(
        1
        for idx, text in enumerate(dialogue_texts)
        if _is_low_value_template_turn(text, dialogue_texts[:idx])
    )
    if generic_template_count >= 2:
        penalty += PENALTY_CONFIG.get(
            "trajectory_generic_template_penalty",
            PENALTY_CONFIG.get("trajectory_generic_question_penalty", 0.0),
        )
        reward = min(reward, 0.58)

    template_count = sum(1 for text in texts if _has_template_expression(text))
    if template_count >= 2:
        penalty += PENALTY_CONFIG["trajectory_template_penalty"]

    overdramatic_count = sum(1 for text in texts if _is_overdramatic(text))
    if overdramatic_count >= 2:
        penalty += PENALTY_CONFIG["trajectory_overdramatic_penalty"]
        reward = min(reward, 0.82)

    generic_action_count = sum(1 for text in action_texts if _is_generic_action(text))
    if action_texts and generic_action_count / len(action_texts) >= 0.5:
        penalty += PENALTY_CONFIG["trajectory_generic_action_penalty"]

    narrator_dialogue_count = 0
    for turn in trajectory:
        if str(turn.get("type", "")) == "action":
            continue
        text = str(turn.get("text", turn.get("utterance", "")))
        if _is_third_person_narration_dialogue("", text):
            narrator_dialogue_count += 1
    if narrator_dialogue_count >= 2:
        penalty += PENALTY_CONFIG["trajectory_narrator_dialogue_penalty"]
        reward = min(reward, 0.70)

    return _clip01(reward - penalty)


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------

def _checklists_to_prompt(checklists: Mapping[str, Sequence[str]]) -> str:
    blocks: List[str] = []
    for dim, items in checklists.items():
        item_lines = "\n".join(f"  {idx}. {item}" for idx, item in enumerate(items, start=1))
        blocks.append(f"{dim}:\n{item_lines}")
    return "\n\n".join(blocks)


def _json_template_for_checklists(checklists: Mapping[str, Sequence[str]]) -> str:
    lines = ["{"]
    dims = list(checklists.keys())
    for i, dim in enumerate(dims):
        comma = "," if i < len(dims) - 1 else ""
        values = ", ".join(["0"] * len(checklists[dim]))
        lines.append(f'  "{dim}": [{values}]{comma}')
    lines.append("}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Evaluators
# -----------------------------------------------------------------------------

def _parse_verifier_response(raw: str) -> Dict[str, Any]:
    """Parse verifier JSON with safe defaults."""
    fallback = {
        "pass": True,
        "severity": "none",
        "gate_factor": 1.0,
        "violations": [],
        "evidence": [],
        "reason": "verifier_parse_fallback",
    }
    try:
        obj = json.loads(_extract_json_object(raw))
        if not isinstance(obj, dict):
            return fallback
        severity = str(obj.get("severity", "none")).strip().lower()
        if severity not in {"none", "minor", "major", "fatal"}:
            severity = "none"
        gate_factor = _clip01(float(obj.get("gate_factor", 1.0)))
        violations = obj.get("violations", [])
        evidence = obj.get("evidence", [])
        return {
            "pass": bool(obj.get("pass", True)),
            "severity": severity,
            "gate_factor": gate_factor,
            "violations": violations if isinstance(violations, list) else [],
            "evidence": evidence if isinstance(evidence, list) else [],
            "reason": str(obj.get("reason", ""))[:500],
        }
    except Exception:
        return fallback


class TurnVerifier:
    """Quality-gating verifier to mitigate reward hacking at turn level."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        enabled: bool = True,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key or "xxx"
        self.model = model
        self.enabled = enabled
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def verify_turn(
        self,
        *,
        scene: Mapping[str, Any],
        character: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        turn_type: str,
        text: str,
        round_idx: int,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "pass": True,
                "severity": "none",
                "gate_factor": 1.0,
                "violations": [],
                "evidence": [],
                "reason": "disabled",
            }

        history_text = _history_to_text(list(history), max_chars=5000)
        system_prompt = (
            "You are a strict verifier for roleplay RL turns. "
            "Your job is NOT scoring quality, but detecting reward-hacking-friendly errors. "
            "Return JSON only."
        )
        user_prompt = f"""[TraceType] verifier

Evaluate ONLY the current turn and recent context.

[Scene]
Event: {scene.get('event', '')}
Time: {scene.get('time', '')}
Location: {scene.get('location', '')}
Description: {scene.get('description', '')}
Social Purpose: {scene.get('social_purpose', '')}

[Character]
Name: {character.get('name', '')}
Description: {character.get('description', '')}
Position: {character.get('position', '')}
State: {character.get('states', '')}

[Recent Context]
{history_text or '(empty)'}

[Current Turn]
Round: {round_idx}
Type: {turn_type}
Speaker: {character.get('name', '')}
Text: {text}

Detection scope:
- format_error
- third_person_or_role_leak
- obvious_template_pattern
- obvious_overdramatic_style
- repetitive_expression
- severe_persona_mismatch
- severe_scene_register_mismatch

Detailed guidance for key risks:

1) obvious_template_pattern
- Judge by structural repetition, not one forbidden phrase.
- Flag when expression strategy is reused with minimal variation, such as repeated
  question-led probing with nearly identical syntax across nearby turns.
- Flag formulaic conversational shells that can be swapped into many scenes
  without meaning change.
- If the line is locally fluent but clearly template-driven, still mark violation.

2) repetitive_expression
- Compare with recent context from the same speaker first, then global context.
- Flag repeated sentence starters, repeated rhetorical frames, or near-duplicate
  semantic content with light lexical substitution.
- Treat "same intent, same shape, new nouns" as repetition when interaction
  function does not change.
- Do not require exact string match.

3) obvious_overdramatic_style
- Evaluate whether emotional and stylistic intensity is proportional to scene stakes.
- Flag cinematic escalation language, catastrophe framing, melodramatic metaphors,
  or constant climax-style delivery when scene context is ordinary/social.
- Flag theatrical body-language narration that dominates pragmatic interaction value.
- Allow occasional strong emotion if well-supported by context; penalize sustained
  over-intensity without grounding.

4) severe_persona_mismatch / severe_scene_register_mismatch
- severe_persona_mismatch: utterance behavior clearly conflicts with character role,
  priorities, or speaking style in this scene.
- severe_scene_register_mismatch: tone/register is incompatible with scene purpose
  (e.g., analyst conversation suddenly becomes action-movie monologue).

General principles:
- Prefer evidence-backed decisions over generosity.
- If multiple weak signals align (template + repetition + overdrama), escalate severity.
- Use recent-context sensitivity: a pattern may be acceptable once but problematic when repeated.

Judgment rules:
- fatal: severe violation that should zero this turn reward (gate_factor=0.0).
- major: clear harmful issue; strong decay (use around gate_factor=0.40).
- minor: weak issue; light decay (use around gate_factor=0.70).
- none: no meaningful issue (gate_factor=1.0).
- Use evidence snippets from current turn and/or immediate context.
- If violations is non-empty, gate_factor must be < 1.0.

Output strict JSON only:
{{
  "pass": true,
  "severity": "none",
  "gate_factor": 1.0,
  "violations": [],
  "evidence": [],
  "reason": "short reason"
}}
"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                timeout=60,
            )
            content = response.choices[0].message.content or ""
            parsed = _parse_verifier_response(content)
            return self._postprocess_verifier_result(
                parsed=parsed,
                text=text,
                history=history,
                turn_type=turn_type,
                character_name=_safe_character_name(character),
            )
        except Exception as exc:
            logger.warning("[Turn Verifier] Failed: %s", exc)
            return {
                "pass": True,
                "severity": "none",
                "gate_factor": 1.0,
                "violations": [],
                "evidence": [],
                "reason": f"verifier_error:{type(exc).__name__}",
            }

    def _postprocess_verifier_result(
        self,
        *,
        parsed: Mapping[str, Any],
        text: str,
        history: Sequence[Mapping[str, Any]],
        turn_type: str,
        character_name: str,
    ) -> Dict[str, Any]:
        """Calibrate verifier output to avoid overly sparse gating."""
        out: Dict[str, Any] = dict(parsed)
        severity = str(out.get("severity", "none")).lower()
        gate = _clip01(float(out.get("gate_factor", 1.0)))
        violations = out.get("violations", [])
        if not isinstance(violations, list):
            violations = []
        violation_count = len(violations)

        # Enforce tiered gate by severity.
        severity_gate_floor = {
            "none": 1.0,
            "minor": 0.70,
            "major": 0.40,
            "fatal": 0.00,
        }
        floor = severity_gate_floor.get(severity, 1.0)
        if severity == "fatal":
            gate = 0.0
        elif severity in {"minor", "major"}:
            gate = min(gate, floor)
        elif severity == "none":
            gate = 1.0

        # If there are violations, force non-1 gate.
        if violation_count > 0 and gate >= 0.999:
            if severity == "major":
                gate = 0.40
            elif severity == "fatal":
                gate = 0.0
            else:
                gate = 0.70

        # Lightweight heuristic backstop for sparse misses.
        if turn_type == "dialogue":
            recent_same_speaker = _recent_turn_texts(history, turn_type="dialogue", limit=6, speaker=character_name)
            recent_dialogue_context = _recent_dialogue_texts(history, limit=10)
            repeat_hit = _has_repeated_starter(text, history, limit=10, speaker=character_name) or _is_repetitive(
                text,
                history,
                speaker=character_name,
                turn_type="dialogue",
            )
            template_hit = _is_question_heavy_template(text) or _has_template_expression(text)
            shell_hit = _repeated_shell_count([*recent_same_speaker, text], _dialogue_shell) >= 1
            low_increment_hit = _low_increment_against_history(text, recent_same_speaker, shell_fn=_dialogue_shell)
            generic_template_hit = _is_low_value_template_turn(text, recent_dialogue_context)
            recent_template_reuse_hit = _is_recent_semantic_template_reuse(text, history, rounds=2)
            drama_hit = _is_overdramatic(text)
            # Hard gate rule: dialogue should not include unnecessary speaker prefix
            # like "Lucas Moreau: ...". Keep gate < 1 even if verifier misses it.
            speaker_prefix_hit = _has_character_name_prefix(character_name, text)
            if gate >= 0.999:
                repeated_starter_hit = _has_repeated_starter(text, history, limit=6, speaker=character_name)
                if recent_template_reuse_hit and generic_template_hit:
                    gate = 0.25
                elif recent_template_reuse_hit and shell_hit:
                    gate = 0.28
                elif generic_template_hit and shell_hit:
                    gate = 0.22
                elif generic_template_hit:
                    gate = 0.30
                elif low_increment_hit and shell_hit:
                    gate = 0.30
                elif repeat_hit and shell_hit:
                    gate = 0.35
                elif low_increment_hit:
                    gate = 0.40
                elif repeated_starter_hit and template_hit:
                    gate = 0.45
                elif repeat_hit and template_hit:
                    gate = 0.55
                elif drama_hit and template_hit:
                    gate = 0.70
                elif speaker_prefix_hit:
                    gate = 0.30
        elif turn_type == "action":
            recent_same_speaker = _recent_turn_texts(history, turn_type="action", limit=6, speaker=character_name)
            repeat_hit = _is_repetitive(text, history, speaker=character_name, turn_type="action")
            shell_hit = _repeated_shell_count([*recent_same_speaker, text], _action_shell) >= 1
            low_increment_hit = _low_increment_against_history(text, recent_same_speaker, shell_fn=_action_shell)
            if gate >= 0.999:
                if low_increment_hit and shell_hit:
                    gate = 0.35
                elif (repeat_hit and shell_hit) or low_increment_hit:
                    gate = 0.42

        out["gate_factor"] = _clip01(gate)
        out["violations"] = violations
        out["severity"] = severity
        out["pass"] = bool(out.get("pass", True))
        return out

class TurnEvaluator:
    """Evaluates short-term reward immediately after each action/dialogue."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        enable_action_evaluation: bool = True,
        enable_dialogue_evaluation: bool = True,
    ):
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key or "xxx"
        self.model = model
        self.enable_action_evaluation = enable_action_evaluation
        self.enable_dialogue_evaluation = enable_dialogue_evaluation
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def evaluate_action(
        self,
        *,
        character: Any,
        action: str,
        round_idx: int,
        history: List[Dict[str, str]],
        scene: Any,
    ) -> float:
        if not self.enable_action_evaluation:
            return 0.2

        char_dict = dict(character) if not isinstance(character, dict) else character
        scene_dict = dict(scene) if not isinstance(scene, dict) else scene
        character_name = _safe_character_name(char_dict)
        active_checklists = _active_checklists(ACTION_CHECKLISTS, ACTION_WEIGHTS)

        try:
            response = await self._call_action_evaluator(
                character=char_dict,
                action=action,
                round_idx=round_idx,
                history=history,
                scene=scene_dict,
            )
            parsed = _parse_checklist_response(response, active_checklists)
            reward = _score_checklists(parsed, ACTION_WEIGHTS)
            reward = _apply_action_hard_penalties(
                reward,
                character_name=character_name,
                action=action,
                history=history,
            )
            logger.debug("[Action Eval] Round %s %s: %.3f", round_idx, character_name, reward)
            return reward
        except Exception as e:
            logger.warning("[Action Eval] Failed for round %s: %s", round_idx, e)
            return _fallback_action_reward(character_name, action, history)

    async def evaluate_dialogue(
        self,
        *,
        character: Any,
        dialogue: str,
        round_idx: int,
        history: List[Dict[str, str]],
        scene: Any,
    ) -> float:
        if not self.enable_dialogue_evaluation:
            return 0.2

        char_dict = dict(character) if not isinstance(character, dict) else character
        scene_dict = dict(scene) if not isinstance(scene, dict) else scene
        character_name = _safe_character_name(char_dict)
        active_checklists = _active_checklists(DIALOGUE_CHECKLISTS, DIALOGUE_WEIGHTS)

        try:
            response = await self._call_dialogue_evaluator(
                character=char_dict,
                dialogue=dialogue,
                round_idx=round_idx,
                history=history,
                scene=scene_dict,
            )
            parsed = _parse_checklist_response(response, active_checklists)
            reward = _score_checklists(parsed, DIALOGUE_WEIGHTS)
            reward = _apply_dialogue_hard_penalties(
                reward,
                character_name=character_name,
                dialogue=dialogue,
                history=history,
            )
            logger.debug("[Dialogue Eval] Round %s %s: %.3f", round_idx, character_name, reward)
            return reward
        except Exception as e:
            logger.warning("[Dialogue Eval] Failed for round %s: %s", round_idx, e)
            return _fallback_dialogue_reward(character_name, dialogue, history)

    async def _call_action_evaluator(
        self,
        *,
        character: Dict[str, Any],
        action: str,
        round_idx: int,
        history: List[Dict[str, str]],
        scene: Dict[str, Any],
    ) -> str:
        history_text = _history_to_text(history, max_chars=4000)
        active_checklists = _active_checklists(ACTION_CHECKLISTS, ACTION_WEIGHTS)

        system_prompt = (
            "You are a strict role-play reward evaluator. "
            "Return valid JSON only. Do not use markdown. Do not explain."
        )
        user_prompt = f"""[TraceType] action_evaluation

Evaluate the character action using checklist scoring.

{_score_item_guidance("action")}

Strict scoring policy:
- Default to 0 unless the evidence is strong.
- Do not infer hidden richness, coherence, or persona detail that is not actually present.
- Generic roleplay gestures such as looking, nodding, sighing, or gazing should not receive 1 unless they create specific scene value.
- Cinematic flourish without interaction value should be scored 0 or -1 on richness and naturalness.
- Penalize repeated action shells from the same speaker, such as repeating the same look/turn/reach gesture pattern with only light lexical changes.
- Penalize actions that do not create an incremental state change, new pressure, or new affordance for the next turn.
- A new prop name alone does not make the action non-redundant if the interaction function stays the same.
- For LSH-RL, prefer concrete interaction effect over decorative specificity or evidence-style prop listing.

[Scene]
Event: {scene.get('event', '')}
Time: {scene.get('time', '')}
Location: {scene.get('location', '')}
Description: {scene.get('description', '')}
Social Purpose: {scene.get('social_purpose', '')}

[Character]
Name: {character.get('name', '')}
Description: {character.get('description', '')}
Position: {character.get('position', '')}
State: {character.get('states', '')}

[Context - Recent Actions/Dialogue]
{history_text if history_text else '(This is the first action)'}

[Current Action to Evaluate - Round {round_idx}]
{action}

[Checklist]
{_checklists_to_prompt(active_checklists)}

Output ONLY valid JSON in this exact schema. Each array must contain integer values from [-1, 0, 1] matching the checklist length above:
{_json_template_for_checklists(active_checklists)}
"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            timeout=60,
        )
        return response.choices[0].message.content or ""

    async def _call_dialogue_evaluator(
        self,
        *,
        character: Dict[str, Any],
        dialogue: str,
        round_idx: int,
        history: List[Dict[str, str]],
        scene: Dict[str, Any],
    ) -> str:
        history_text = _history_to_text(history, max_chars=4000)
        active_checklists = _active_checklists(DIALOGUE_CHECKLISTS, DIALOGUE_WEIGHTS)

        system_prompt = (
            "You are a strict role-play reward evaluator. "
            "Return valid JSON only. Do not use markdown. Do not explain."
        )
        user_prompt = f"""[TraceType] dialogue_evaluation

Evaluate the character dialogue using checklist scoring.

{_score_item_guidance("dialogue")}

Strict dialogue-format rules:
- Dialogue should be what the character says, not a third-person plot summary.
- Penalize lines like "Han notices...", "She prepares...", or "Next, he..." as narrator-style dialogue.
- Penalize unnecessary speaker prefixes such as "Sun Li:" if the expected output is the utterance only.
- Penalize accidental non-target language tokens or mixed-language glitches.
- Penalize repeated, template-like, or melodramatic wording.
- Repeated reusable openers or shells should be scored low when they mainly recycle the same interaction function, whether phrased as a question or a statement.
- A line can be coherent but still score low if it is flat, repetitive, or too theatrical.
- Penalize structural reuse from the same speaker, including "same intent, same shell, new nouns".
- Penalize turns that do not add incremental information, commitment, boundary, reveal, or concrete response value.
- Generic probing questions do not count as rich interaction unless they are tightly anchored to the immediate context and add concrete value such as a new fact, explanation pressure, choice pressure, or actionable next move.
- Generic stock statements do not count as rich interaction either if they only sound reflective, formal, evidential, or specific without creating a real new development, commitment, boundary, reveal, topic shift, or next-step value.
- Judge template-ness by interaction function and incremental contribution, not by surface phrase alone.
- For LSH-RL, do not reward lines merely for sounding evidential, technical, formal, or detail-rich if they reduce social naturalness or trap the character in one interaction shell.

[Scene]
Event: {scene.get('event', '')}
Time: {scene.get('time', '')}
Location: {scene.get('location', '')}
Description: {scene.get('description', '')}
Social Purpose: {scene.get('social_purpose', '')}

[Character]
Name: {character.get('name', '')}
Description: {character.get('description', '')}
Position: {character.get('position', '')}
State: {character.get('states', '')}

[Context - Recent Actions/Dialogue]
{history_text if history_text else '(This is the first dialogue)'}

[Current Dialogue to Evaluate - Round {round_idx}]
Speaker: {character.get('name', '')}
Dialogue: {dialogue}

[Checklist]
{_checklists_to_prompt(active_checklists)}

Output ONLY valid JSON in this exact schema. Each array must contain integer values from [-1, 0, 1] matching the checklist length above:
{_json_template_for_checklists(active_checklists)}
"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            timeout=60,
        )
        return response.choices[0].message.content or ""


class TrajectoryEvaluator:
    """Evaluates long-term reward after complete scene trajectory."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        enable_evaluation: bool = True,
    ):
        self.base_url = _normalize_base_url(base_url)
        self.api_key = api_key or "xxx"
        self.model = model
        self.enable_evaluation = enable_evaluation
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def evaluate_trajectory(
        self,
        *,
        character: Any,
        trajectory: List[Dict[str, Any]],
        scene: Any,
    ) -> float:
        if not self.enable_evaluation:
            return 0.2

        char_dict = dict(character) if not isinstance(character, dict) else character
        scene_dict = dict(scene) if not isinstance(scene, dict) else scene

        try:
            response = await self._call_trajectory_evaluator(
                character=char_dict,
                trajectory=trajectory,
                scene=scene_dict,
            )
            if REWARD_PROFILE == "lsh_rl":
                parsed_scores = _parse_holistic_metric_scores(response, TRAJECTORY_HOLISTIC_METRICS)
                reward = _normalize_holistic_metric_scores(parsed_scores)
                logger.info("[Trajectory Eval Parsed] %s: %s", char_dict.get("name"), parsed_scores)
            else:
                active_checklists = _active_checklists(TRAJECTORY_CHECKLISTS, TRAJECTORY_WEIGHTS)
                parsed = _parse_checklist_response(response, active_checklists)
                reward = _score_checklists(parsed, TRAJECTORY_WEIGHTS)
            reward = _apply_trajectory_hard_penalties(reward, trajectory=trajectory)
            logger.info("[Trajectory Eval] %s: %.3f", char_dict.get("name"), reward)
            return reward
        except Exception as e:
            logger.warning("[Trajectory Eval] Failed for %s: %s", char_dict.get("name"), e)
            return _fallback_trajectory_reward(trajectory)

    async def _call_trajectory_evaluator(
        self,
        *,
        character: Dict[str, Any],
        trajectory: List[Dict[str, Any]],
        scene: Dict[str, Any],
    ) -> str:
        trajectory_lines: List[str] = []
        for turn in trajectory:
            round_idx = turn.get("round", 0)
            turn_type = turn.get("type", "")
            text = turn.get("text", turn.get("utterance", ""))
            if turn_type == "action":
                trajectory_lines.append(f"Round {round_idx} [ACTION]: {text}")
            else:
                trajectory_lines.append(f"Round {round_idx} [Dialogue]: {text}")
        trajectory_text = "\n".join(trajectory_lines)

        system_prompt = (
            "You are a strict role-play trajectory evaluator. "
            "Return valid JSON only. Do not use markdown. Do not explain."
        )
        if REWARD_PROFILE == "lsh_rl":
            user_prompt = f"""[TraceType] trajectory_evaluation

Evaluate the overall quality of this character's performance across the entire scene using holistic 1-5 scoring.

Scoring dimensions:
1. Personality Traits
   - 1: Frequently conflicts with the character's established personality, priorities, or voice.
   - 3: Generally matches the character, but includes some flattening or inconsistent choices.
   - 5: Consistently reflects a distinctive personality, priorities, and voice across the full trajectory.
2. Behavioral Coherence
   - 1: Often illogical, contradictory, or disconnected from prior turns and scene developments.
   - 3: Mostly coherent, but includes some weak transitions or partial resets in behavior.
   - 5: Actions and dialogue build naturally across turns, with clear causal continuity and believable progression.
3. Interaction Richness
   - 1: Repeats nearly identical interaction shells or statement patterns; little meaningful progress.
   - 3: Sometimes varies strategy and introduces some new information or movement, but still falls into repetition.
   - 5: Consistently fresh, varied, and socially meaningful; advances the conversation through multiple interaction strategies.

Important long-term judging rules:
- Evaluate the whole trajectory, not isolated turns.
- Strategy diversity matters more than named-detail accumulation.
- Repeated evidence gathering, procedural fact logging, or formal-request shells should score Interaction Richness low.
- Repeated reflective or explanatory turns should also score Interaction Richness low when they keep serving the same interaction function, even if the surface phrasing changes.
- Repeating the same interaction shell across turns with new nouns is still repetition.
- Judge template-ness by interaction function and incremental contribution, not by surface phrase alone.
- Do not reward a trajectory merely for sounding specific, technical, or detail-rich.
- Question-led turns only count as rich when they create real choice pressure, explanation pressure, or concrete next-step value.

[Scene]
Event: {scene.get('event', '')}
Time: {scene.get('time', '')}
Location: {scene.get('location', '')}
Description: {scene.get('description', '')}
Social Purpose: {scene.get('social_purpose', '')}

[Character]
Name: {character.get('name', '')}
Description: {character.get('description', '')}

[Complete Character Trajectory]
{trajectory_text}

Output ONLY valid JSON in this exact schema with integer values 1-5:
{{
  "Personality Traits": <int>,
  "Behavioral Coherence": <int>,
  "Interaction Richness": <int>
}}
"""
        else:
            active_checklists = _active_checklists(TRAJECTORY_CHECKLISTS, TRAJECTORY_WEIGHTS)
            user_prompt = f"""[TraceType] trajectory_evaluation

Evaluate the overall quality of this character's performance across the entire scene using checklist scoring.

{_score_item_guidance("trajectory")}

Strict long-term penalty rules:
- If the same or highly similar dialogue appears repeatedly, score Interaction Richness low.
- If dialogue repeatedly becomes narrator-style summary, score Immersion and Behavioral Coherence low.
- If the character ignores new information, score Adaptability low.
- If the character drifts from persona, score Persona Consistency low.
- If the character keeps using the same question pattern or sentence starter, score Interaction Richness and Adaptability low.
- If actions are mostly decorative stock gestures, score Interaction Richness low.
- Repeating the same interaction shell across turns with new nouns is still repetition and should score Behavioral Coherence / Interaction Richness low.
- Question-led turns count as rich interaction only when they add concrete value such as new facts, explanation pressure, specific constraints, or clear decision pressure.
- Reward trajectories that vary strategy while staying causally connected and persona-consistent.
- Do not give near-perfect scores unless the trajectory is both varied and grounded.
- For LSH-RL, strategy diversity matters more than named-detail accumulation. Repeated evidence gathering, procedural fact logging, or formal-request shells should score Interaction Richness low unless the interaction mode itself changes across turns.

[Scene]
Event: {scene.get('event', '')}
Time: {scene.get('time', '')}
Location: {scene.get('location', '')}
Description: {scene.get('description', '')}
Social Purpose: {scene.get('social_purpose', '')}

[Character]
Name: {character.get('name', '')}
Description: {character.get('description', '')}

[Complete Character Trajectory]
{trajectory_text}

[Checklist]
{_checklists_to_prompt(active_checklists)}

Output ONLY valid JSON in this exact schema. Each array must contain integer values from [-1, 0, 1] matching the checklist length above:
{_json_template_for_checklists(active_checklists)}
"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            timeout=120,
        )
        return response.choices[0].message.content or ""
