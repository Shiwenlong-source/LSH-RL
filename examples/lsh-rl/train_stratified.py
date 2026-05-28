#!/usr/bin/env python3
"""Train RolePlay Persona scene tasks with Agent-Lightning VERL + Stratified Sampling.

This is a MINIMAL modification of train_roleplay_agent.py.
ONLY CHANGE: Uses stratified sampling instead of loading from scenes_dir.
EVERYTHING ELSE is identical to V3.
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
    _print_compact_step_metrics,
    _slim_logs_enabled,
    _timer,
    _tracking_backends,
    _write_resume_meta,
)
from omegaconf import OmegaConf
from tqdm import tqdm

from roleplay_agent import CharacterSpec, SceneTask, roleplay_persona_agent

# Import stratified sampler
from stratified_sampler import load_stratified_datasets

# Import everything from the original training script
from train_roleplay_agent import (
    FIXED_EPISODE_ROUNDS,
    DEFAULT_ACTOR_MODEL_PATH,
    DEFAULT_ADAPTER_PATH,
    DEFAULT_ENV_BASE_URL,
    DEFAULT_ENV_API_KEY,
    DEFAULT_ENV_MODEL,
    DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_MAX_RESPONSE_LENGTH,
    _checkpoint_policy_from_env,
    _trainer_freqs_from_env,
    _best_checkpoint_from_env,
    _resume_strategy_from_env,
    _prepare_lightweight_resume,
    _resolve_model_and_tokenizer_paths,
    verl_default_config,
    BestCheckpointAgentLightningTrainer,
)

logger = logging.getLogger(__name__)


def train_stratified(
    *,
    # NEW: Stratified sampling parameters
    train_data_dir: Path,
    batch_size: int,
    num_batches: int,
    low_ratio: int,
    medium_ratio: int,
    high_ratio: int,
    # Same as V3
    seed: int,
    model_path: str,
    tokenizer_path: str,
    lora_adapter_path: str,
    lora_adapter_rank: int,
    adv_estimator: str,
    n_runners: int,
    n_gpus_per_node: int,
    split_actor_critic_gpus: bool = False,
    actor_rollout_gpus_per_node: int = 4,
    critic_gpus_per_node: int = 0,  # 0 means share with actor pool
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
    """Train with stratified sampling - IDENTICAL to V3's train() except data loading."""

    # ============================================================
    # ONLY CHANGE: Use stratified sampling instead of scenes_dir
    # ============================================================
    print(f"\n{'='*60}")
    print("Loading datasets with stratified sampling...")
    print(f"{'='*60}")
    print(f"Train data dir: {train_data_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Num batches: {num_batches}")
    print(f"Sampling ratios: low={low_ratio}, medium={medium_ratio}, high={high_ratio}")
    print(f"{'='*60}\n")

    train_dataset, val_dataset = load_stratified_datasets(
        train_data_dir=train_data_dir,
        batch_size=batch_size,
        num_batches=num_batches,
        evaluator_base_url=evaluator_base_url,
        evaluator_api_key=evaluator_api_key,
        evaluator_model=evaluator_model,
        low_ratio=low_ratio,
        medium_ratio=medium_ratio,
        high_ratio=high_ratio,
        seed=seed,
    )

    logger.info(
        "Loaded %d train tasks and %d val tasks using stratified sampling",
        len(train_dataset),
        len(val_dataset),
    )
    # ============================================================
    # END OF CHANGE - Everything below is identical to V3
    # ============================================================

    config = verl_default_config(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        lora_adapter_path=lora_adapter_path,
        lora_adapter_rank=lora_adapter_rank,
        adv_estimator=adv_estimator,
        n_gpus_per_node=n_gpus_per_node,
        split_actor_critic_gpus=split_actor_critic_gpus,
        actor_rollout_gpus_per_node=1,  # Each runner uses 1 GPU in shared mode
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
    parser = argparse.ArgumentParser(
        description="Train RolePlay Persona with stratified difficulty sampling."
    )

    # ============================================================
    # NEW: Stratified sampling parameters (replacing scenes-dir)
    # ============================================================
    parser.add_argument("--train-data-dir", type=Path, default=Path("train_data"))
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--low-ratio", type=int, default=2)
    parser.add_argument("--medium-ratio", type=int, default=2)
    parser.add_argument("--high-ratio", type=int, default=1)
    # ============================================================

    # SAME AS V3
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", type=str, default=DEFAULT_ACTOR_MODEL_PATH)
    parser.add_argument(
        "--base-model",
        type=str,
        default="",
    )
    parser.add_argument("--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--adv-estimator", type=str, default="gae", choices=["grpo", "gae"])
    parser.add_argument("--n-runners", type=int, default=4)
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
    parser.add_argument("--rollout-gpu-memory-utilization", type=float, default=DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--total-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    parser.add_argument("--max-response-length", type=int, default=DEFAULT_MAX_RESPONSE_LENGTH)
    parser.add_argument("--rollout-log-prob-micro-batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--ref-log-prob-micro-batch-size-per-gpu", type=int, default=4)
    parser.add_argument("--actor-ppo-micro-batch-size-per-gpu", type=int, default=2)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--actor-use-kl-loss", action="store_true")
    parser.add_argument("--actor-no-kl-loss", action="store_false", dest="actor_use_kl_loss")
    parser.set_defaults(actor_use_kl_loss=False)
    parser.add_argument("--actor-kl-loss-coef", type=float, default=0.05)
    parser.add_argument("--actor-entropy-coeff", type=float, default=0.001)
    parser.add_argument("--logger", action="append", dest="logger")
    parser.add_argument("--external-store-address", type=str, default="")
    parser.add_argument("--evaluator-base-url", type=str, default=DEFAULT_ENV_BASE_URL)
    parser.add_argument("--evaluator-api-key", type=str, default=DEFAULT_ENV_API_KEY)
    parser.add_argument("--evaluator-model", type=str, default=DEFAULT_ENV_MODEL)
    parser.add_argument("--no-resume", action="store_true", help="Do not load checkpoints; start training from scratch.")

    args = parser.parse_args()

    if not args.logger:
        args.logger = ["console"]
    args.adapter_path, _ = _prepare_lightweight_resume(
        adapter_path=args.adapter_path,
        no_resume=args.no_resume,
    )
    args.model, args.tokenizer_path, args.lora_adapter_path, args.lora_adapter_rank = _resolve_model_and_tokenizer_paths(
        args.model, args.base_model, args.adapter_path
    )
    if args.adapter_path and not args.lora:
        raise ValueError("--adapter-path requires LoRA training. Do not pass --no-lora.")

    train_stratified(
        train_data_dir=args.train_data_dir,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        low_ratio=args.low_ratio,
        medium_ratio=args.medium_ratio,
        high_ratio=args.high_ratio,
        seed=args.seed,
        model_path=args.model,
        tokenizer_path=args.tokenizer_path,
        lora_adapter_path=args.lora_adapter_path,
        lora_adapter_rank=args.lora_adapter_rank,
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
