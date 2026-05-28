# Copyright (c) Microsoft. All rights reserved.

"""Train PersonaArena-like scene tasks with Agent-Lightning + VERL.

This script mirrors the Calc-X training pattern:
- use Agent-Lightning VERL algorithm
- one scene as one task
- run full trajectory and emit one terminal reward in rollout
- support both local managed store and external client/server split
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Sequence, cast

import agentlightning as agl
import verl
from agentlightning.adapter import TraceToTripletBase
from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var
from agentlightning.verl.trainer import (
    AgentLightningTrainer,
    _print_compact_step_metrics,
    _slim_logs_enabled,
    _timer,
    _tracking_backends,
    _write_resume_meta,
)
from omegaconf import OmegaConf
from tqdm import tqdm
from verl.utils.tracking import Tracking

from roleplay_agent import CharacterSpec, SceneTask, roleplay_persona_agent

logger = logging.getLogger(__name__)

DEFAULT_SCENES_DIR = str(Path(__file__).resolve().parent / "train_data")
DEFAULT_ACTOR_MODEL_PATH = os.environ.get("AGL_BASE_MODEL", "xxx")
DEFAULT_ADAPTER_PATH = os.environ.get("AGL_ADAPTER_PATH", "xxx")
DEFAULT_ENV_BASE_URL = os.environ.get("ROLEPLAY_ENV_BASE_URL", "xxx")
DEFAULT_ENV_API_KEY = os.environ.get("ROLEPLAY_ENV_API_KEY", "xxx")
DEFAULT_ENV_MODEL = os.environ.get("ROLEPLAY_ENV_MODEL", "xxx")
FIXED_EPISODE_ROUNDS = 5
DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION = 0.5
DEFAULT_TRAIN_BATCH_SIZE = 2
DEFAULT_MAX_PROMPT_LENGTH = 3000
DEFAULT_MAX_RESPONSE_LENGTH = 1024


def _default_checkpoint_root() -> Path:
    """Return the default checkpoint root used by roleplay training."""
    experiment_name = os.environ.get("AGL_EXPERIMENT_NAME", "roleplay_persona")
    return Path(__file__).resolve().parent / "checkpoints" / experiment_name


def _checkpoint_policy_from_env() -> tuple[str, int]:
    """Resolve checkpoint retention policy for roleplay training.

    Policies:
    - `inference`: keep only lightweight inference artifacts. Actor still exports
      `lora_adapter/`, while actor/critic skip model and optimizer shards.
    - `model_only`: save model shards and metadata, but skip optimizer state.
    - `full`: save model + optimizer + extra state for full resume fidelity.
    """
    policy = os.environ.get("AGL_CHECKPOINT_POLICY", "inference").strip().lower() or "inference"
    keep = max(1, int(os.environ.get("AGL_MAX_CKPT_TO_KEEP", "1")))
    if policy not in {"inference", "model_only", "full"}:
        raise ValueError(f"Unsupported AGL_CHECKPOINT_POLICY={policy!r}")
    return policy, keep


def _trainer_freqs_from_env() -> tuple[int, int]:
    """Resolve validation/save frequencies for the VERL trainer."""
    test_freq = max(0, int(os.environ.get("AGL_TEST_FREQ", "32")))
    save_freq = max(0, int(os.environ.get("AGL_SAVE_FREQ", "32")))
    return test_freq, save_freq


def _best_checkpoint_from_env() -> tuple[bool, str]:
    """Resolve whether to save only validation-best checkpoints."""
    enabled = os.environ.get("AGL_SAVE_BEST_ONLY", "").lower() in {"1", "true", "yes", "on"}
    metric_name = os.environ.get("AGL_BEST_CKPT_METRIC", "val/reward").strip() or "val/reward"
    return enabled, metric_name


def _resume_strategy_from_env() -> str:
    """Resolve how roleplay training should continue after interruptions.

    Strategies:
    - `exact`: use VERL checkpoint restore. This requires loading full actor/critic
      shards and is memory-heavy.
    - `lightweight`: skip VERL checkpoint loading and instead warm-start actor LoRA
      weights from the latest saved `actor/lora_adapter` directory. Critic,
      optimizer, dataloader state, and global step are reinitialized.
    """
    strategy = os.environ.get("AGL_RESUME_STRATEGY", "exact").strip().lower() or "exact"
    if strategy not in {"exact", "lightweight"}:
        raise ValueError(f"Unsupported AGL_RESUME_STRATEGY={strategy!r}")
    return strategy


def _logical_step_offset_from_env() -> int:
    """Return the logical step offset used to stitch lightweight restarts."""
    return max(0, int(os.environ.get("AGL_LOGICAL_STEP_OFFSET", "0")))


def _find_latest_lora_adapter(checkpoint_root: Path) -> tuple[str, int] | None:
    """Find the newest checkpoint that contains an actor LoRA adapter export."""
    if not checkpoint_root.exists():
        return None

    candidates: list[tuple[int, Path, float]] = []
    for ckpt_dir in checkpoint_root.glob("global_step_*"):
        if not ckpt_dir.is_dir():
            continue
        step_match = re.match(r"^global_step_(\d+)", ckpt_dir.name)
        if step_match is None:
            continue
        step = int(step_match.group(1))
        adapter_dir = ckpt_dir / "actor" / "lora_adapter"
        if (adapter_dir / "adapter_config.json").exists() and (adapter_dir / "adapter_model.safetensors").exists():
            # Use modification time as tiebreaker for checkpoints with the same step number
            mtime = (adapter_dir / "adapter_model.safetensors").stat().st_mtime
            candidates.append((step, adapter_dir, mtime))

    if not candidates:
        return None

    # First by step number (descending), then by modification time (descending)
    step, adapter_dir, _ = max(candidates, key=lambda item: (item[0], item[2]))
    return str(adapter_dir), step


def _prepare_lightweight_resume(
    *,
    adapter_path: str,
    no_resume: bool,
) -> tuple[str, int | None]:
    """Resolve actor adapter path for lightweight continuation.

    Returns the adapter path to use and the source checkpoint step if a newer
    actor adapter checkpoint was found. Lightweight continuation always disables
    VERL checkpoint loading because it intentionally avoids full actor/critic
    shard restore.
    """
    strategy = _resume_strategy_from_env()
    os.environ["AGL_LOGICAL_STEP_OFFSET"] = "0"
    os.environ["AGL_RESUME_SOURCE_ADAPTER_PATH"] = ""
    os.environ["AGL_RESUME_SOURCE_STEP"] = ""
    if strategy != "lightweight" or no_resume:
        return adapter_path, None

    latest = _find_latest_lora_adapter(_default_checkpoint_root())
    os.environ["AGL_NO_RESUME"] = "1"
    if latest is None:
        logger.info(
            "Lightweight resume enabled, but no saved actor LoRA adapter was found under %s. "
            "Falling back to the configured adapter path.",
            _default_checkpoint_root(),
        )
        return adapter_path, None

    latest_adapter_path, latest_step = latest
    logger.info(
        "Lightweight resume enabled. Warm-starting actor from latest LoRA adapter: step=%s path=%s",
        latest_step,
        latest_adapter_path,
    )
    os.environ["AGL_LOGICAL_STEP_OFFSET"] = str(latest_step)
    os.environ["AGL_RESUME_SOURCE_ADAPTER_PATH"] = latest_adapter_path
    os.environ["AGL_RESUME_SOURCE_STEP"] = str(latest_step)
    return latest_adapter_path, latest_step


class BestCheckpointAgentLightningTrainer(AgentLightningTrainer):
    """Save checkpoints only when the selected validation metric improves."""

    def _checkpoint_root(self) -> Path:
        checkpoint_root = Path(str(self.config.trainer.default_local_dir))
        if not checkpoint_root.is_absolute():
            checkpoint_root = Path.cwd() / checkpoint_root
        return checkpoint_root

    def _extract_best_metric_value(self, metrics: Dict[str, Any]) -> float | None:
        _, metric_name = _best_checkpoint_from_env()
        raw_value = metrics.get(metric_name)
        if raw_value is None:
            return None
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            logger.warning("Best-checkpoint metric %s is not numeric: %r", metric_name, raw_value)
            return None

    def _refresh_best_checkpoint_alias(self, *, best_step: int, best_score: float) -> None:
        checkpoint_root = self._checkpoint_root()
        checkpoint_root.mkdir(parents=True, exist_ok=True)

        best_meta_path = checkpoint_root / "best_checkpoint.txt"
        best_meta_path.write_text(f"step={best_step}\nmetric={best_score}\n", encoding="utf-8")

        symlink_path = checkpoint_root / "best_checkpoint"
        target = checkpoint_root / f"global_step_{best_step}"
        try:
            if symlink_path.is_symlink() or symlink_path.exists():
                symlink_path.unlink()
            symlink_path.symlink_to(target.name)
        except OSError:
            logger.warning("Unable to refresh best_checkpoint symlink at %s", symlink_path)

    def _cleanup_non_best_checkpoints(self, *, best_step: int) -> None:
        checkpoint_root = self._checkpoint_root()
        best_dir_name = f"global_step_{best_step}"
        for ckpt_dir in checkpoint_root.glob("global_step_*"):
            if ckpt_dir.name == best_dir_name:
                continue
            if ckpt_dir.is_dir():
                shutil.rmtree(ckpt_dir, ignore_errors=True)

    def fit(self):
        tracking_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=_tracking_backends(self.config.trainer.logger),
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        best_score = float("-inf")
        best_step = 0
        _, best_metric_name = _best_checkpoint_from_env()

        no_resume = os.environ.get("AGL_NO_RESUME", "").lower() in {"1", "true", "yes", "on"}
        if no_resume:
            pprint("AGL_NO_RESUME is set; skipping checkpoint load.")
        else:
            self._load_checkpoint()

        logical_step_offset = _logical_step_offset_from_env()
        start_epoch = 0
        skip_batches_in_first_epoch = 0
        completed_steps = self.global_steps
        if no_resume and logical_step_offset > 0:
            completed_steps = min(logical_step_offset, self.total_training_steps)
            self.global_steps = completed_steps
            steps_per_epoch = max(1, len(self.train_dataloader))
            start_epoch = min(self.config.trainer.total_epochs, completed_steps // steps_per_epoch)
            if completed_steps < self.total_training_steps:
                skip_batches_in_first_epoch = completed_steps % steps_per_epoch
            pprint(
                "Applying lightweight logical progress: "
                f"completed_steps={completed_steps} start_epoch={start_epoch} "
                f"skip_batches_in_first_epoch={skip_batches_in_first_epoch}"
            )

        steps_per_epoch = max(1, len(self.train_dataloader))
        resume_meta = {
            "resume_strategy": os.environ.get("AGL_RESUME_STRATEGY", "exact"),
            "no_resume_effective": no_resume,
            "logical_step_offset": logical_step_offset,
            "completed_steps": completed_steps,
            "start_epoch": start_epoch,
            "skip_batches_in_first_epoch": skip_batches_in_first_epoch,
            "steps_per_epoch": steps_per_epoch,
            "configured_total_epochs": int(self.config.trainer.total_epochs),
            "total_training_steps": int(self.total_training_steps),
            "source_adapter_path": os.environ.get("AGL_RESUME_SOURCE_ADAPTER_PATH", ""),
            "source_step": os.environ.get("AGL_RESUME_SOURCE_STEP", ""),
        }
        _write_resume_meta(self.config.trainer.default_local_dir, resume_meta)

        if self.global_steps >= self.total_training_steps:
            pprint(f"Training already complete at logical step {self.global_steps}. Nothing to do.")
            return

        assert self.async_rollout_mode, "If agent mode is enabled, async server must be enabled"
        if self.adapter is not None and not isinstance(self.adapter, TraceToTripletBase):
            raise ValueError("Adapter must be a TraceToTripletBase for currently VERL implementation.")
        model = (
            "/".join(self.config.actor_rollout_ref.model.path.split("/")[-2:])
            if verl.__version__ == "0.5.0"
            else self.config.actor_rollout_ref.model.path
        )
        self.agent_mode_daemon = self.daemon_cls(
            self.config.agentlightning.port,
            self.config.actor_rollout_ref.rollout.n,
            train_information={
                "model": model,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
            tokenizer=self.tokenizer,
            mini_batch_size=self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
            pad_token_id=self.tokenizer.pad_token_id,
            mode="v1" if self.store is not None else "v0",
            store=self.store,
            llm_proxy=self.llm_proxy,
            adapter=self.adapter,
            processor=self.processor,
            image_base_dir=getattr(self.config.data, "image_base_dir", None),
            trace_aggregator=self.config.agentlightning.trace_aggregator,
        )
        self.agent_mode_daemon.start()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            tracking_logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                if epoch == start_epoch and batch_idx < skip_batches_in_first_epoch:
                    continue
                metrics: Dict[str, Any] = {}
                timing_raw: Dict[str, float] = {}
                is_last_step = self.global_steps >= self.total_training_steps

                metrics = self._train_step(batch_dict)

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with _timer("validate", timing_raw):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                    current_score = self._extract_best_metric_value(val_metrics)
                    if current_score is not None and current_score > best_score:
                        best_score = current_score
                        best_step = self.global_steps
                        logger.info(
                            "Validation improved on %s: step=%s score=%.6f. Saving best checkpoint only.",
                            best_metric_name,
                            best_step,
                            best_score,
                        )
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                        self._cleanup_non_best_checkpoints(best_step=best_step)
                        self._refresh_best_checkpoint_alias(best_step=best_step, best_score=best_score)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                if best_step > 0:
                    metrics["checkpoint/best_step"] = best_step
                    metrics[f"checkpoint/best_{best_metric_name}"] = best_score

                if _slim_logs_enabled():
                    _print_compact_step_metrics(metrics, self.global_steps)
                tracking_logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    if best_step > 0:
                        pprint(f"Best checkpoint: step={best_step} {best_metric_name}={best_score}")
                    progress_bar.close()
                    pprint("Flush the logger...")
                    del tracking_logger
                    pprint(f"Training finished at step {self.global_steps}.")
                    return

                progress_bar.update(1)
                self.global_steps += 1


def _resolve_model_and_tokenizer_paths(model: str, base_model: str, adapter_path: str) -> tuple[str, str, str, int]:
    """Resolve actor/critic model paths and LoRA adapter metadata."""
    if not adapter_path:
        if base_model:
            logger.warning("--base-model is ignored because --adapter-path is not set.")
        return model, model, "", 0

    adapter_dir = Path(adapter_path)
    adapter_config_file = adapter_dir / "adapter_config.json"
    if not adapter_config_file.exists():
        raise FileNotFoundError(f"adapter_config.json not found under --adapter-path: {adapter_dir}")

    try:
        adapter_config = cast(Dict[str, Any], json.loads(adapter_config_file.read_text(encoding="utf-8")))
        adapter_base = str(adapter_config.get("base_model_name_or_path", "")).strip()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid adapter_config.json: {adapter_config_file}") from exc

    if base_model:
        if adapter_base and adapter_base != base_model:
            raise ValueError(
                "Base model mismatch: --base-model="
                f"{base_model} but adapter_config.json base_model_name_or_path={adapter_base}"
            )
        resolved_base_model = base_model
    elif adapter_base:
        resolved_base_model = adapter_base
        logger.info("Auto-detected base model from adapter_config.json: %s", resolved_base_model)
    else:
        raise ValueError(
            "When --adapter-path is set, you must provide --base-model if adapter_config.json "
            "does not contain base_model_name_or_path."
        )

    adapter_rank = int(adapter_config.get("r", 0) or 0)
    logger.info("Using base+adapter loading path via adapter checkpoint: %s", adapter_dir)
    logger.info("Tokenizer will be loaded from base model path: %s", resolved_base_model)
    return resolved_base_model, resolved_base_model, str(adapter_dir), adapter_rank


def verl_default_config(
    *,
    model_path: str,
    tokenizer_path: str,
    lora_adapter_path: str,
    lora_adapter_rank: int,
    adv_estimator: str,
    n_gpus_per_node: int,
    split_actor_critic_gpus: bool,
    actor_rollout_gpus_per_node: int,
    critic_gpus_per_node: int,
    rollout_gpu_memory_utilization: float,
    total_epochs: int,
    train_batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    rollout_log_prob_micro_batch_size_per_gpu: int,
    ref_log_prob_micro_batch_size_per_gpu: int,
    actor_ppo_micro_batch_size_per_gpu: int,
    lora: bool,
    lora_rank: int,
    trainer_loggers: Sequence[str],
    actor_use_kl_loss: bool,
    actor_kl_loss_coef: float,
    actor_entropy_coeff: float,
) -> Dict[str, Any]:
    """VERL config adapted from calc_x for roleplay scene training."""
    checkpoint_policy, max_ckpt_to_keep = _checkpoint_policy_from_env()
    test_freq, save_freq = _trainer_freqs_from_env()
    save_best_only, best_metric_name = _best_checkpoint_from_env()
    experiment_name = os.environ.get("AGL_EXPERIMENT_NAME", "roleplay_persona")
    checkpoint_root = _default_checkpoint_root()
    config: Dict[str, Any] = {
        "algorithm": {
            "adv_estimator": adv_estimator,
            "use_kl_in_reward": False,
        },
        "data": {
            "train_batch_size": max(1, train_batch_size),
            "max_prompt_length": max(1, max_prompt_length),
            "max_response_length": max(1, max_response_length),
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 1,
                "log_prob_micro_batch_size_per_gpu": max(1, rollout_log_prob_micro_batch_size_per_gpu),
                "multi_turn": {"format": "hermes"},
                "name": "vllm",
                "gpu_memory_utilization": rollout_gpu_memory_utilization,
            },
            "actor": {
                "ppo_mini_batch_size": max(1, train_batch_size),
                "ppo_micro_batch_size_per_gpu": max(1, actor_ppo_micro_batch_size_per_gpu),
                "optim": {"lr": 1e-6},
                "use_kl_loss": actor_use_kl_loss,
                "kl_loss_coef": actor_kl_loss_coef,
                "entropy_coeff": actor_entropy_coeff,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "fsdp_config": {
                    "param_offload": True,
                    "optimizer_offload": True,
                },
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": max(1, ref_log_prob_micro_batch_size_per_gpu),
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": model_path,
                "tokenizer_path": tokenizer_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": n_gpus_per_node,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": list(trainer_loggers),
            "project_name": "AgentLightning",
            "experiment_name": experiment_name,
            "default_local_dir": str(checkpoint_root),
            "nnodes": 1,
            "save_freq": 0 if save_best_only else save_freq,
            "test_freq": test_freq,
            "total_epochs": total_epochs,
            "max_actor_ckpt_to_keep": max_ckpt_to_keep,
            "max_critic_ckpt_to_keep": max_ckpt_to_keep,
        },
    }

    actor_checkpoint_cfg: Dict[str, Any]
    critic_checkpoint_cfg: Dict[str, Any]
    if checkpoint_policy == "full":
        actor_checkpoint_cfg = {"load_contents": ["model", "optimizer", "extra"], "save_contents": ["model", "optimizer", "extra"]}
        critic_checkpoint_cfg = {"load_contents": ["model", "optimizer", "extra"], "save_contents": ["model", "optimizer", "extra"]}
    elif checkpoint_policy == "model_only":
        actor_checkpoint_cfg = {"load_contents": ["model", "extra"], "save_contents": ["model", "extra"]}
        critic_checkpoint_cfg = {"load_contents": ["model", "extra"], "save_contents": ["model", "extra"]}
    else:
        # Inference-only mode: actor worker still exports LoRA adapter weights via
        # `lora_adapter/adapter_model.safetensors`, so we can skip massive FSDP
        # model and optimizer shards entirely.
        actor_checkpoint_cfg = {"load_contents": ["model", "optimizer", "extra"], "save_contents": []}
        critic_checkpoint_cfg = {"load_contents": ["model", "optimizer", "extra"], "save_contents": []}

    cast(Dict[str, Any], config["actor_rollout_ref"]["actor"])["checkpoint"] = actor_checkpoint_cfg

    if split_actor_critic_gpus:
        config["trainer"]["separate_actor_critic_pool"] = True
        config["trainer"]["actor_rollout_gpus_per_node"] = actor_rollout_gpus_per_node
        config["trainer"]["critic_gpus_per_node"] = critic_gpus_per_node

    if adv_estimator == "gae":
        actor_cfg = cast(Dict[str, Any], config["actor_rollout_ref"]["actor"])
        critic_cfg = cast(Dict[str, Any], config.setdefault("critic", {}))

        critic_cfg.setdefault("ppo_mini_batch_size", actor_cfg["ppo_mini_batch_size"])
        critic_cfg.setdefault("ppo_micro_batch_size_per_gpu", actor_cfg["ppo_micro_batch_size_per_gpu"])

        critic_model_cfg = cast(Dict[str, Any], critic_cfg.setdefault("model", {}))
        # Critic must load a full base model (value-head path), not an adapter-only checkpoint.
        critic_model_cfg["path"] = tokenizer_path
        critic_model_cfg["tokenizer_path"] = tokenizer_path
        critic_model_cfg.setdefault("enable_gradient_checkpointing", True)
        critic_model_cfg.setdefault("use_remove_padding", True)

        # Critical for 2-GPU PPO/GAE memory: offload critic params+optimizer states.
        critic_fsdp_cfg = cast(Dict[str, Any], critic_model_cfg.setdefault("fsdp_config", {}))
        critic_fsdp_cfg.setdefault("param_offload", True)
        critic_fsdp_cfg.setdefault("optimizer_offload", True)
        critic_fsdp_cfg.setdefault("dtype", "bfloat16")
        critic_fsdp_cfg.setdefault("model_dtype", "bfloat16")
        critic_cfg["checkpoint"] = critic_checkpoint_cfg

        # Do not force LoRA on critic:
        # critic uses token-classification path and PEFT LoRA may require generation APIs.

    if lora_adapter_path:
        actor_model_cfg = cast(Dict[str, Any], config["actor_rollout_ref"]["model"])
        actor_model_cfg["lora_adapter_path"] = lora_adapter_path
        actor_model_cfg["lora_rank"] = max(1, lora_adapter_rank)
    elif lora:
        config["actor_rollout_ref"]["model"]["lora_rank"] = lora_rank

    logger.info(
        "Trainer cadence: test_freq=%s save_freq=%s checkpoint_policy=%s max_ckpt_to_keep=%s save_best_only=%s best_metric=%s",
        test_freq,
        0 if save_best_only else save_freq,
        checkpoint_policy,
        max_ckpt_to_keep,
        save_best_only,
        best_metric_name,
    )

    return config


def _scene_files_from_dir(scenes_dir: Path, limit_files: int) -> List[Path]:
    files = sorted(scenes_dir.glob("*.json"))
    if limit_files > 0:
        files = files[:limit_files]
    if not files:
        raise ValueError(f"No scene json files found in: {scenes_dir}")
    return files


def _load_tasks(
    *,
    scene_files: Sequence[Path],
    evaluator_base_url: str,
    evaluator_api_key: str,
    evaluator_model: str,
) -> List[SceneTask]:
    tasks: List[SceneTask] = []

    for scene_file in scene_files:
        with scene_file.open("r", encoding="utf-8") as f:
            payload = cast(dict[str, Any], json.load(f))

        title = str(payload.get("title", scene_file.stem))
        scenes = cast(List[dict[str, Any]], payload.get("scenes", []))
        if not scenes:
            logger.warning("Skip file without 'scenes': %s", scene_file)
            continue

        for scene in scenes:
            scene_id = int(scene.get("id", 0))
            task: SceneTask = {
                "task_id": f"{scene_file.stem}::scene-{scene_id}",
                "scene_file": str(scene_file),
                "scene_id": scene_id,
                "title": title,
                "event": str(scene.get("event", "")),
                "time": str(scene.get("time", "")),
                "location": str(scene.get("location", "")),
                "description": str(scene.get("description", "")),
                "plot": str(scene.get("plot", "")),
                "social_purpose": str(scene.get("social_purpose", "")),
                "max_rounds": FIXED_EPISODE_ROUNDS,
                "characters": cast(List[CharacterSpec], scene.get("characters", [])),
                "actions": cast(List[dict[str, Any]], scene.get("actions", [])),
                "evaluator_base_url": evaluator_base_url,
                "evaluator_api_key": evaluator_api_key,
                "evaluator_model": evaluator_model,
            }
            tasks.append(task)

    if not tasks:
        raise ValueError("No valid scene tasks were loaded.")

    return tasks


def _split_train_val(tasks: Sequence[SceneTask], val_ratio: float, seed: int) -> tuple[List[SceneTask], List[SceneTask]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")

    indexed = list(tasks)
    random.Random(seed).shuffle(indexed)

    if len(indexed) == 1:
        return indexed, indexed

    n_val = int(round(len(indexed) * val_ratio))
    n_val = max(1, min(len(indexed) - 1, n_val))
    val_dataset = indexed[:n_val]
    train_dataset = indexed[n_val:]
    return train_dataset, val_dataset


def _parse_scene_index_list(raw: str) -> List[int]:
    values: List[int] = []
    for chunk in (raw or "").split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            index = int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid scene index {text!r}; expected comma-separated integers.") from exc
        if index <= 0:
            raise ValueError(f"Scene indices must be 1-based positive integers, got {index}.")
        values.append(index)
    deduped: List[int] = []
    seen: set[int] = set()
    for index in values:
        if index not in seen:
            deduped.append(index)
            seen.add(index)
    return deduped


def _split_scene_files_by_fixed_val_indices(scene_files: Sequence[Path], val_scene_indices: Sequence[int]) -> tuple[List[Path], List[Path]]:
    if not val_scene_indices:
        return list(scene_files), []

    n_files = len(scene_files)
    invalid = [index for index in val_scene_indices if index > n_files]
    if invalid:
        raise ValueError(f"Fixed val scene indices out of range for {n_files} files: {invalid}")

    val_index_set = set(val_scene_indices)
    val_files = [path for idx, path in enumerate(scene_files, 1) if idx in val_index_set]
    train_files = [path for idx, path in enumerate(scene_files, 1) if idx not in val_index_set]
    if not train_files:
        raise ValueError("Fixed val scene selection left no training files.")
    return train_files, val_files


def train(
    *,
    scenes_dir: str,
    scene_limit: int,
    val_ratio: float,
    val_scene_indices: Sequence[int],
    seed: int,
    model_path: str,
    tokenizer_path: str,
    lora_adapter_path: str,
    lora_adapter_rank: int,
    adv_estimator: str,
    n_runners: int,
    n_gpus_per_node: int,
    split_actor_critic_gpus: bool,
    actor_rollout_gpus_per_node: int,
    critic_gpus_per_node: int,
    rollout_gpu_memory_utilization: float,
    total_epochs: int,
    train_batch_size: int,
    max_prompt_length: int,
    max_response_length: int,
    rollout_log_prob_micro_batch_size_per_gpu: int,
    ref_log_prob_micro_batch_size_per_gpu: int,
    actor_ppo_micro_batch_size_per_gpu: int,
    lora: bool,
    lora_rank: int,
    trainer_loggers: Sequence[str],
    actor_use_kl_loss: bool,
    actor_kl_loss_coef: float,
    actor_entropy_coeff: float,
    external_store_address: str,
    evaluator_base_url: str,
    evaluator_api_key: str,
    evaluator_model: str,
) -> None:
    scene_files = _scene_files_from_dir(Path(scenes_dir), scene_limit)
    if val_scene_indices:
        train_scene_files, val_scene_files = _split_scene_files_by_fixed_val_indices(scene_files, val_scene_indices)
        train_dataset = _load_tasks(
            scene_files=train_scene_files,
            evaluator_base_url=evaluator_base_url,
            evaluator_api_key=evaluator_api_key,
            evaluator_model=evaluator_model,
        )
        val_dataset = _load_tasks(
            scene_files=val_scene_files,
            evaluator_base_url=evaluator_base_url,
            evaluator_api_key=evaluator_api_key,
            evaluator_model=evaluator_model,
        )
        logger.info(
            "Loaded %d scene files with fixed val indices=%s. Train files=%d Val files=%d Train tasks=%d Val tasks=%d",
            len(scene_files),
            list(val_scene_indices),
            len(train_scene_files),
            len(val_scene_files),
            len(train_dataset),
            len(val_dataset),
        )
    else:
        all_tasks = _load_tasks(
            scene_files=scene_files,
            evaluator_base_url=evaluator_base_url,
            evaluator_api_key=evaluator_api_key,
            evaluator_model=evaluator_model,
        )
        train_dataset, val_dataset = _split_train_val(all_tasks, val_ratio=val_ratio, seed=seed)

        logger.info(
            "Loaded %d tasks from %d files. Train=%d Val=%d",
            len(all_tasks),
            len(scene_files),
            len(train_dataset),
            len(val_dataset),
        )

    config = verl_default_config(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        lora_adapter_path=lora_adapter_path,
        lora_adapter_rank=lora_adapter_rank,
        adv_estimator=adv_estimator,
        n_gpus_per_node=n_gpus_per_node,
        split_actor_critic_gpus=split_actor_critic_gpus,
        actor_rollout_gpus_per_node=actor_rollout_gpus_per_node,
        critic_gpus_per_node=critic_gpus_per_node,
        rollout_gpu_memory_utilization=rollout_gpu_memory_utilization,
        total_epochs=total_epochs,
        train_batch_size=train_batch_size,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        rollout_log_prob_micro_batch_size_per_gpu=rollout_log_prob_micro_batch_size_per_gpu,
        ref_log_prob_micro_batch_size_per_gpu=ref_log_prob_micro_batch_size_per_gpu,
        actor_ppo_micro_batch_size_per_gpu=actor_ppo_micro_batch_size_per_gpu,
        lora=lora,
        lora_rank=lora_rank,
        trainer_loggers=trainer_loggers,
        actor_use_kl_loss=actor_use_kl_loss,
        actor_kl_loss_coef=actor_kl_loss_coef,
        actor_entropy_coeff=actor_entropy_coeff,
    )
    # VERL may drop incomplete batches. Ensure train batch size is never larger than
    # the actual train set size, otherwise train_dataloader can become empty.
    effective_batch_size = max(1, min(int(config["data"]["train_batch_size"]), len(train_dataset)))
    config["data"]["train_batch_size"] = effective_batch_size
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = effective_batch_size
    actor_micro_batch = min(
        int(config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"]),
        effective_batch_size,
    )
    config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = max(1, actor_micro_batch)
    if "critic" in config:
        config["critic"]["ppo_mini_batch_size"] = effective_batch_size
        critic_micro_batch = min(int(config["critic"]["ppo_micro_batch_size_per_gpu"]), effective_batch_size)
        config["critic"]["ppo_micro_batch_size_per_gpu"] = max(1, critic_micro_batch)
    logger.info(
        "Adaptive batch sizes: train_batch_size=%d ppo_mini_batch_size=%d",
        effective_batch_size,
        effective_batch_size,
    )
    logger.info(
        "VERL runtime knobs: model=%s adv_estimator=%s lora=%s lora_rank=%s "
        "n_gpus_per_node=%d split_actor_critic_gpus=%s actor_rollout_gpus_per_node=%d "
        "critic_gpus_per_node=%d rollout_gpu_memory_utilization=%.3f max_prompt_length=%d max_response_length=%d",
        model_path,
        adv_estimator,
        lora,
        lora_rank,
        n_gpus_per_node,
        split_actor_critic_gpus,
        actor_rollout_gpus_per_node,
        critic_gpus_per_node,
        float(config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"]),
        int(config["data"]["max_prompt_length"]),
        int(config["data"]["max_response_length"]),
    )
    logger.info(
        "Model routing: actor_model_path=%s actor_tokenizer_path=%s critic_model_path=%s critic_tokenizer_path=%s",
        cast(Dict[str, Any], config["actor_rollout_ref"]["model"]).get("path"),
        cast(Dict[str, Any], config["actor_rollout_ref"]["model"]).get("tokenizer_path"),
        cast(Dict[str, Any], cast(Dict[str, Any], config.get("critic", {})).get("model", {})).get("path"),
        cast(Dict[str, Any], cast(Dict[str, Any], config.get("critic", {})).get("model", {})).get("tokenizer_path"),
    )
    logger.info(
        "LoRA routing: actor_lora_adapter_path=%s actor_lora_rank=%s",
        cast(Dict[str, Any], config["actor_rollout_ref"]["model"]).get("lora_adapter_path"),
        cast(Dict[str, Any], config["actor_rollout_ref"]["model"]).get("lora_rank", 0),
    )
    logger.info(
        "Checkpoint policy: policy=%s keep=%s actor_save=%s critic_save=%s",
        _checkpoint_policy_from_env()[0],
        config["trainer"].get("max_actor_ckpt_to_keep"),
        cast(Dict[str, Any], cast(Dict[str, Any], config["actor_rollout_ref"]).get("actor", {}))
        .get("checkpoint", {})
        .get("save_contents"),
        cast(Dict[str, Any], cast(Dict[str, Any], config.get("critic", {})).get("checkpoint", {})).get("save_contents"),
    )
    if "critic" in config:
        critic_model_cfg = cast(Dict[str, Any], cast(Dict[str, Any], config["critic"]).get("model", {}))
        critic_fsdp_cfg = cast(Dict[str, Any], critic_model_cfg.get("fsdp_config", {}))
        logger.info(
            "Critic memory knobs: ppo_micro_batch_size_per_gpu=%s gc=%s remove_padding=%s "
            "param_offload=%s optimizer_offload=%s dtype=%s model_dtype=%s lora_rank=%s",
            cast(Dict[str, Any], config["critic"]).get("ppo_micro_batch_size_per_gpu"),
            critic_model_cfg.get("enable_gradient_checkpointing"),
            critic_model_cfg.get("use_remove_padding"),
            critic_fsdp_cfg.get("param_offload"),
            critic_fsdp_cfg.get("optimizer_offload"),
            critic_fsdp_cfg.get("dtype"),
            critic_fsdp_cfg.get("model_dtype"),
            critic_model_cfg.get("lora_rank", 0),
        )
    if ("8B" in model_path or "8b" in model_path) and adv_estimator == "gae" and actor_rollout_gpus_per_node <= 1:
        logger.warning(
            "8B + GAE puts actor+rollout on the same pool and can OOM on vLLM wake_up. "
            "If OOM persists, prefer --adv-estimator grpo or lower rollout_gpu_memory_utilization."
        )
    save_best_only, _ = _best_checkpoint_from_env()
    logger.info("Resume strategy: %s", _resume_strategy_from_env())
    trainer_cls = BestCheckpointAgentLightningTrainer if save_best_only else None
    algorithm = agl.VERL(config, trainer_cls=trainer_cls)

    if external_store_address:
        store = agl.LightningStoreClient(external_store_address)
    else:
        store = None

    trainer = agl.Trainer(algorithm=algorithm, n_runners=n_runners, store=store)
    trainer.fit(roleplay_persona_agent, train_dataset, val_dataset=val_dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RolePlay Persona scene tasks with Agent-Lightning VERL.")
    parser.add_argument("--scenes-dir", type=str, default=DEFAULT_SCENES_DIR)
    parser.add_argument("--scene-limit", type=int, default=40, help="Limit scene files loaded. <=0 means all files.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--val-scene-indices",
        type=str,
        default="",
        help="Comma-separated 1-based scene file indices to use as a fixed validation/probe set.",
    )
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--model", type=str, default=DEFAULT_ACTOR_MODEL_PATH, help="Actor model path for VERL training.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="",
        help=(
            "Base model path for base+adapter training. "
            "Used for adapter/base consistency check when --adapter-path is set."
        ),
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help=(
            "LoRA/PEFT adapter checkpoint path (contains adapter_config.json). "
            "When set, training starts from base+adapter and skips merged model requirement."
        ),
    )
    parser.add_argument(
        "--adv-estimator",
        type=str,
        default="gae",
        choices=["grpo", "gae"],
        help="Advantage estimator for VERL. Use 'gae' for PPO-style updates.",
    )

    parser.add_argument("--n-runners", type=int, default=1)
    parser.add_argument("--n-gpus-per-node", type=int, default=2)
    parser.add_argument(
        "--split-actor-critic-gpus",
        action="store_true",
        dest="split_actor_critic_gpus",
        help="Place ActorRollout and Critic in separate Ray resource pools.",
    )
    parser.add_argument(
        "--no-split-actor-critic-gpus",
        action="store_false",
        dest="split_actor_critic_gpus",
        help="Disable separate ActorRollout/Critic pools.",
    )
    parser.set_defaults(split_actor_critic_gpus=True)
    parser.add_argument("--actor-rollout-gpus-per-node", type=int, default=1)
    parser.add_argument("--critic-gpus-per-node", type=int, default=1)
    parser.add_argument(
        "--rollout-gpu-memory-utilization",
        type=float,
        default=DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Train batch size for VERL (smaller is more stable for backend 502 pressure).",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=DEFAULT_MAX_PROMPT_LENGTH,
        help="Max prompt token length for VERL.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=DEFAULT_MAX_RESPONSE_LENGTH,
        help="Max response token length for VERL.",
    )
    parser.add_argument(
        "--rollout-log-prob-micro-batch-size-per-gpu",
        type=int,
        default=2,
        help="vLLM rollout log-prob micro batch size per GPU.",
    )
    parser.add_argument(
        "--ref-log-prob-micro-batch-size-per-gpu",
        type=int,
        default=4,
        help="Reference model log-prob micro batch size per GPU.",
    )
    parser.add_argument(
        "--actor-ppo-micro-batch-size-per-gpu",
        type=int,
        default=2,
        help="Actor PPO micro batch size per GPU.",
    )
    parser.add_argument("--lora", action="store_true", help="Enable LoRA training to reduce memory usage.")
    parser.add_argument("--no-lora", action="store_false", dest="lora", help="Disable LoRA training.")
    parser.set_defaults(lora=True)
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank when --lora is enabled.")
    parser.add_argument(
        "--actor-use-kl-loss",
        action="store_true",
        help="Enable KL loss term in actor optimization for more stable PPO updates.",
    )
    parser.add_argument(
        "--actor-no-kl-loss",
        action="store_false",
        dest="actor_use_kl_loss",
        help="Disable KL loss term in actor optimization.",
    )
    parser.set_defaults(actor_use_kl_loss=True)
    parser.add_argument(
        "--actor-kl-loss-coef",
        type=float,
        default=0.01,
        help="KL loss coefficient for actor optimization.",
    )
    parser.add_argument(
        "--actor-entropy-coeff",
        type=float,
        default=0.001,
        help="Entropy regularization coefficient for actor optimization.",
    )
    parser.add_argument(
        "--logger",
        action="append",
        choices=["console", "tensorboard", "wandb"],
        default=None,
        help=(
            "Trainer logger backend. Repeat to enable multiple backends, "
            "e.g. --logger console --logger tensorboard"
        ),
    )

    parser.add_argument(
        "--external-store-address",
        type=str,
        default="",
        help="Use an external store, e.g. xxx",
    )

    parser.add_argument("--evaluator-base-url", type=str, default=DEFAULT_ENV_BASE_URL)
    parser.add_argument("--evaluator-api-key", type=str, default=DEFAULT_ENV_API_KEY)
    parser.add_argument("--evaluator-model", type=str, default=DEFAULT_ENV_MODEL)
    parser.add_argument("--no-resume", action="store_true", help="Do not load checkpoints; start training from scratch.")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    if args.no_resume:
        # Must be set before Ray workers are created so it propagates to subprocesses.
        os.environ["AGL_NO_RESUME"] = "1"

    if args.scene_limit <= 0:
        args.scene_limit = 0
    if not args.logger:
        args.logger = ["console"]
    args.adapter_path, lightweight_resume_step = _prepare_lightweight_resume(
        adapter_path=args.adapter_path,
        no_resume=args.no_resume,
    )
    args.model, resolved_tokenizer_path, resolved_lora_adapter_path, resolved_lora_adapter_rank = _resolve_model_and_tokenizer_paths(
        args.model, args.base_model, args.adapter_path
    )
    if lightweight_resume_step is not None:
        logger.info(
            "Actor warm-start source: global_step_%s via adapter path %s. "
            "Critic and dataloader will be reinitialized.",
            lightweight_resume_step,
            resolved_lora_adapter_path,
        )
    if args.adapter_path and not args.lora:
        raise ValueError("--adapter-path requires LoRA training. Do not pass --no-lora.")

    if args.external_store_address and resolve_bool_env_var(LightningEnvVar.AGL_MANAGED_STORE, fallback=True):
        raise ValueError(
            "When using an external store, set AGL_MANAGED_STORE=0, "
            "or omit --external-store-address to let Trainer manage store lifecycle."
        )

    train(
        scenes_dir=args.scenes_dir,
        scene_limit=args.scene_limit,
        val_ratio=args.val_ratio,
        val_scene_indices=_parse_scene_index_list(args.val_scene_indices),
        seed=args.seed,
        model_path=args.model,
        tokenizer_path=resolved_tokenizer_path,
        lora_adapter_path=resolved_lora_adapter_path,
        lora_adapter_rank=resolved_lora_adapter_rank,
        adv_estimator=args.adv_estimator,
        n_runners=args.n_runners,
        n_gpus_per_node=args.n_gpus_per_node,
        split_actor_critic_gpus=args.split_actor_critic_gpus,
        actor_rollout_gpus_per_node=args.actor_rollout_gpus_per_node,
        critic_gpus_per_node=args.critic_gpus_per_node,
        rollout_gpu_memory_utilization=args.rollout_gpu_memory_utilization,
        total_epochs=args.total_epochs,
        train_batch_size=args.train_batch_size,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        rollout_log_prob_micro_batch_size_per_gpu=args.rollout_log_prob_micro_batch_size_per_gpu,
        ref_log_prob_micro_batch_size_per_gpu=args.ref_log_prob_micro_batch_size_per_gpu,
        actor_ppo_micro_batch_size_per_gpu=args.actor_ppo_micro_batch_size_per_gpu,
        lora=args.lora,
        lora_rank=args.lora_rank,
        trainer_loggers=args.logger,
        actor_use_kl_loss=args.actor_use_kl_loss,
        actor_kl_loss_coef=args.actor_kl_loss_coef,
        actor_entropy_coeff=args.actor_entropy_coeff,
        external_store_address=args.external_store_address,
        evaluator_base_url=args.evaluator_base_url,
        evaluator_api_key=args.evaluator_api_key,
        evaluator_model=args.evaluator_model,
    )


if __name__ == "__main__":
    main()
