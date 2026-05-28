# Copyright (c) Microsoft. All rights reserved.

# type: ignore

from __future__ import annotations

import random
import os
import json
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Any, DefaultDict, Dict, List, Tuple, Type

import numpy as np
import torch
import verl
from codetiming import Timer
from omegaconf import OmegaConf
from tqdm import tqdm
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    _compute_response_info,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking

from agentlightning.adapter import TraceAdapter, TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore

from .daemon import AgentModeDaemon

__all__ = [
    "AgentLightningTrainer",
]


def _slim_logs_enabled() -> bool:
    """Return whether roleplay training should emit compact logs only."""
    return os.environ.get("AGL_SLIM_LOGS", "").lower() in {"1", "true", "yes", "on"}


def _tracking_backends(default_backend: Any) -> Any:
    """Drop the noisy console backend when slim logging is enabled."""
    if not _slim_logs_enabled():
        return default_backend
    if isinstance(default_backend, str):
        return default_backend
    return [backend for backend in default_backend if backend != "console"]


def _compact_metric_keys() -> List[str]:
    """Return the whitelist of per-step metrics kept in slim logs."""
    return [
        "training/reward",
        "training/reward_all_triplets",
        "training/n_rollouts",
        "training/n_triplets",
        "val/reward",
        "actor/pg_loss",
        "actor/ppo_kl",
        "actor/kl_loss",
        "actor/grad_norm",
        "critic/vf_loss",
        "critic/grad_norm",
        "response_length/mean_after_processing",
        "perf/throughput",
        "timing_s/step",
        "training/global_step",
        "training/epoch",
    ]


def _print_compact_step_metrics(metrics: Dict[str, Any], step: int) -> None:
    """Emit a compact step summary using the existing plot parser format."""
    fields: List[str] = []
    for key in _compact_metric_keys():
        if key not in metrics:
            continue
        value = metrics[key]
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        fields.append(f"{key}:{value}")
    if fields:
        print(f"step:{step} - " + " - ".join(fields), flush=True)


def _patch_verl_fsdp_checkpoint_manager_cpu_load() -> None:
    """Avoid CUDA OOM when resuming from sharded FSDP checkpoints.

    verl's FSDPCheckpointManager uses ShardedStateDictConfig(offload_to_cpu=True) but still calls
    torch.load() without map_location, which restores tensors to their original device (often CUDA).
    Loading those shards directly on GPU can OOM before FSDP consumes them.

    We patch the module-local torch.load to default to map_location="cpu".
    """
    if os.environ.get("AGL_DISABLE_CPU_CKPT_LOAD_PATCH", "").lower() in {"1", "true", "yes", "on"}:
        return

    try:
        import verl.utils.checkpoint.fsdp_checkpoint_manager as fsdp_ckpt  # type: ignore
    except Exception:
        return

    if getattr(fsdp_ckpt, "_AGL_CPU_LOAD_PATCHED", False):
        return

    orig_load = fsdp_ckpt.torch.load

    def _cpu_load(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs.setdefault("map_location", "cpu")
        return orig_load(path, *args, **kwargs)

    fsdp_ckpt.torch.load = _cpu_load  # type: ignore[assignment]
    fsdp_ckpt._AGL_CPU_LOAD_PATCHED = True


_patch_verl_fsdp_checkpoint_manager_cpu_load()


def _logical_step_offset_from_env() -> int:
    """Return the logical step offset used to stitch lightweight restarts."""
    return max(0, int(os.environ.get("AGL_LOGICAL_STEP_OFFSET", "0")))


def _write_resume_meta(checkpoint_root: str | Path, payload: Dict[str, Any]) -> None:
    """Write lightweight resume metadata for operational visibility."""
    root = Path(checkpoint_root)
    if not root.is_absolute():
        root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    meta_path = root / "resume_meta.json"
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


# This function is adapted from verl.
# We introduce a new parameter `suffix` to distinguish between metrics computed
# before and after AgentLightning’s post-processing.
# - "Before" refers to raw reward and advantage values.
# - "After" refers to values computed following post-processing, which involves:
#     (1) Dropping prompts that exceed the maximum allowed length.
#     (2) Adjusting the batch size to be a multiple of the mini PPO size.
# Different suffixes are used to label these two stages accordingly.
def compute_data_metrics(batch: DataProto, use_critic: bool = True, suffix: str = "") -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean" + suffix: torch.mean(sequence_score).detach().item(),
        "critic/score/max" + suffix: torch.max(sequence_score).detach().item(),
        "critic/score/min" + suffix: torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean" + suffix: torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max" + suffix: torch.max(sequence_reward).detach().item(),
        "critic/rewards/min" + suffix: torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean" + suffix: torch.mean(valid_adv).detach().item(),
        "critic/advantages/max" + suffix: torch.max(valid_adv).detach().item(),
        "critic/advantages/min" + suffix: torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean" + suffix: torch.mean(valid_returns).detach().item(),
        "critic/returns/max" + suffix: torch.max(valid_returns).detach().item(),
        "critic/returns/min" + suffix: torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean" + suffix: torch.mean(valid_values).detach().item(),
                "critic/values/max" + suffix: torch.max(valid_values).detach().item(),
                "critic/values/min" + suffix: torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var" + suffix: (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean" + suffix: torch.mean(response_length).detach().item(),
        "response_length/max" + suffix: torch.max(response_length).detach().item(),
        "response_length/min" + suffix: torch.min(response_length).detach().item(),
        "response_length/clip_ratio"
        + suffix: torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean" + suffix: torch.mean(prompt_length).detach().item(),
        "prompt_length/max" + suffix: torch.max(prompt_length).detach().item(),
        "prompt_length/min" + suffix: torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio"
        + suffix: torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


class AgentLightningTrainer(RayPPOTrainer):
    """
    Specialized PPO trainer for agent-based reinforcement learning.

    This trainer is designed specifically for scenarios where the model interacts with
    external environments, tools, or APIs through an AgentLightningServer. It simplifies
    the training loop by removing the complex conditional logic present in the original
    RayPPOTrainer and focusing on the agent mode workflow.

    Key differences from RayPPOTrainer:

    1. Uses AgentModeDaemon for server communication
    2. Simplified data flow without pop/union operations
    3. Direct batch processing through agent daemon
    4. Streamlined validation using agent_mode validation
    """

    def __init__(
        self,
        store: LightningStore | None,
        llm_proxy: LLMProxy | None,
        adapter: TraceAdapter | None,
        daemon_cls: Type[AgentModeDaemon],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.store = store
        self.llm_proxy = llm_proxy
        self.adapter = adapter
        self.daemon_cls = daemon_cls

    def _validate(self):
        assert len(self.val_dataloader) == 1, "Please set val_batch_size to None for better throughput."

        test_data = next(iter(self.val_dataloader))
        test_batch = DataProto.from_single_dict(test_data)

        self.async_rollout_manager.wake_up()
        self.agent_mode_daemon.set_up_data_and_server(
            test_batch.non_tensor_batch,
            self.async_rollout_manager.server_addresses,
            is_train=False,
        )
        self.agent_mode_daemon.run_until_all_finished()
        test_metrics = self.agent_mode_daemon.get_test_metrics()
        self.agent_mode_daemon.clear_data_and_server()
        self.async_rollout_manager.sleep()
        return test_metrics

    def _compute_reference_log_prob(self, batch: DataProto) -> DataProto:
        """Compute reference log probability using the correct worker based on LoRA configuration.

        In verl 0.6.0+, when LoRA is detected (indicated by ref_in_actor=True),
        the reference policy is computed by the actor rollout worker instead of a separate
        ref policy worker. This method handles both scenarios by checking the ref_in_actor flag.
        Note: verl sets ref_in_actor=True when it detects LoRA configuration (e.g., lora_rank > 0 or lora_adapter_path is set).

        Args:
            batch: The data batch to compute reference log probabilities for.

        Returns:
            DataProto with reference log probabilities added.

        Raises:
            RuntimeError: If the required worker is not available.
        """
        if getattr(self, "ref_in_actor", False):
            actor_worker = getattr(self, "actor_rollout_wg", None)
            if actor_worker is None:
                raise RuntimeError("actor_rollout_wg is required when ref_in_actor is True.")
            return actor_worker.compute_ref_log_prob(batch)

        ref_worker = getattr(self, "ref_policy_wg", None)
        if ref_worker is None:
            raise RuntimeError(
                "Reference policy worker was not initialized. "
                "Ensure `use_reference_policy` is enabled and the VERL config exposes the ref worker."
            )
        return ref_worker.compute_ref_log_prob(batch)

    def _compute_triplet_level_omar_scores(self, batch: DataProto) -> tuple[torch.Tensor, Dict[str, float]]:
        """Allocate long-term reward over triplets using OMAR-style sequence credit.

        This approximates OMAR's turn-level stage at triplet granularity:
        1. treat each triplet as one step in the character trajectory
        2. use the last token value of each triplet as the step value
        3. place the character's long-term reward on the last triplet only
        4. run GAE over the triplet sequence to derive relative credit
        5. redistribute the long-term reward budget across triplets accordingly
        6. use the merged per-triplet reward as the token-level pseudo reward
        """
        required_keys = [
            "rollout_id_list",
            "turn_index_list",
            "character_name_list",
            "short_term_reward_list",
            "long_term_reward_list",
            "combine_short_weight_list",
            "combine_long_weight_list",
        ]
        if any(key not in batch.non_tensor_batch for key in required_keys):
            return batch.batch["token_level_scores"], {}
        if "values" not in batch.batch:
            return batch.batch["token_level_scores"], {}

        response_mask = batch.batch["response_mask"].bool()
        values = batch.batch["values"]
        token_level_scores = torch.zeros_like(batch.batch["token_level_scores"])

        rollout_ids = batch.non_tensor_batch["rollout_id_list"].tolist()
        turn_indices = batch.non_tensor_batch["turn_index_list"].tolist()
        character_names = batch.non_tensor_batch["character_name_list"].tolist()
        short_rewards = batch.non_tensor_batch["short_term_reward_list"].tolist()
        long_rewards = batch.non_tensor_batch["long_term_reward_list"].tolist()
        short_weights = batch.non_tensor_batch["combine_short_weight_list"].tolist()
        long_weights = batch.non_tensor_batch["combine_long_weight_list"].tolist()

        last_token_values: List[float] = []
        last_token_positions: List[int] = []
        for idx in range(values.shape[0]):
            valid_positions = torch.nonzero(response_mask[idx], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                last_token_positions.append(-1)
                last_token_values.append(0.0)
                continue
            last_pos = int(valid_positions[-1].item())
            last_token_positions.append(last_pos)
            last_token_values.append(float(values[idx, last_pos].detach().item()))

        grouped_indices: DefaultDict[Tuple[str, str], List[int]] = defaultdict(list)
        for idx, (rollout_id, character_name) in enumerate(zip(rollout_ids, character_names)):
            grouped_indices[(str(rollout_id), str(character_name))].append(idx)

        merged_scores = [0.0 for _ in range(values.shape[0])]
        long_credit_contribs: List[float] = []
        group_sizes: List[int] = []

        gamma = float(self.config.algorithm.gamma)
        lam = float(self.config.algorithm.lam)
        temp = 1.0

        for _, seq_indices in grouped_indices.items():
            ordered = sorted(seq_indices, key=lambda i: int(turn_indices[i]))
            if not ordered:
                continue

            step_values = [last_token_values[i] for i in ordered]
            terminal_long_reward = float(long_rewards[ordered[-1]])
            short_weight = float(short_weights[ordered[-1]])
            long_weight = float(long_weights[ordered[-1]])

            advantages = [0.0 for _ in ordered]
            gae = 0.0
            for local_idx in range(len(ordered) - 1, -1, -1):
                reward_t = terminal_long_reward if local_idx == len(ordered) - 1 else 0.0
                value_t = step_values[local_idx]
                if local_idx == len(ordered) - 1:
                    next_value = 0.0
                    nonterminal = 0.0
                else:
                    next_value = step_values[local_idx + 1]
                    nonterminal = 1.0
                delta = reward_t + gamma * next_value * nonterminal - value_t
                gae = delta + gamma * lam * nonterminal * gae
                advantages[local_idx] = gae

            adv_tensor = torch.tensor(advantages, dtype=torch.float32)
            norm_weights = torch.softmax(adv_tensor / temp, dim=0)

            for local_idx, global_idx in enumerate(ordered):
                allocated_long = long_weight * terminal_long_reward * float(norm_weights[local_idx].item())
                merged_score = short_weight * float(short_rewards[global_idx]) + allocated_long
                merged_scores[global_idx] = merged_score
                long_credit_contribs.append(allocated_long)
            group_sizes.append(len(ordered))

        # Token-level allocation inside each triplet:
        # use token values + GAE-style backward recursion within the turn, then
        # allocate each triplet's merged score to all valid response tokens.
        for idx, score in enumerate(merged_scores):
            valid_positions = torch.nonzero(response_mask[idx], as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            turn_advantages: List[float] = [0.0 for _ in range(int(valid_positions.numel()))]
            turn_gae = 0.0
            for rev_local_idx in range(int(valid_positions.numel()) - 1, -1, -1):
                pos = int(valid_positions[rev_local_idx].item())
                reward_t = float(score) if rev_local_idx == int(valid_positions.numel()) - 1 else 0.0
                value_t = float(values[idx, pos].detach().item())
                if rev_local_idx == int(valid_positions.numel()) - 1:
                    next_value = 0.0
                    nonterminal = 0.0
                else:
                    next_pos = int(valid_positions[rev_local_idx + 1].item())
                    next_value = float(values[idx, next_pos].detach().item())
                    nonterminal = 1.0
                delta = reward_t + gamma * next_value * nonterminal - value_t
                turn_gae = delta + gamma * lam * nonterminal * turn_gae
                turn_advantages[rev_local_idx] = turn_gae

            token_weights = torch.softmax(torch.tensor(turn_advantages, dtype=torch.float32) / temp, dim=0)
            for local_idx, pos in enumerate(valid_positions.tolist()):
                token_level_scores[idx, int(pos)] = float(score) * float(token_weights[local_idx].item())

        metrics = {
            "training/triplet_omar_enabled": 1.0,
            "training/triplet_omar_mean_score": float(np.mean(merged_scores)) if merged_scores else 0.0,
            "training/triplet_omar_mean_long_credit": float(np.mean(long_credit_contribs)) if long_credit_contribs else 0.0,
            "training/triplet_omar_mean_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        }
        return token_level_scores, metrics

    def _train_step(self, batch_dict: dict) -> dict:
        # Isolate in a separate method to automatically recycle the variables before validation.
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        metrics = {}
        timing_raw = {}

        with _timer("step", timing_raw):

            # When agent mode is enabled, we read the batch as it is.
            gen_batch = batch

            # generate a batch
            with _timer("gen", timing_raw):
                self.async_rollout_manager.wake_up()
                self.agent_mode_daemon.set_up_data_and_server(
                    gen_batch.non_tensor_batch, self.async_rollout_manager.server_addresses
                )
                self.agent_mode_daemon.run_until_all_finished()
                batch, agent_metrics = self.agent_mode_daemon.get_train_data_batch(
                    max_prompt_length=(
                        self.config.agentlightning.trace_aggregator.trajectory_max_prompt_length
                        if self.config.agentlightning.trace_aggregator.level.startswith("trajectory")
                        else self.config.data.max_prompt_length
                    ),
                    max_response_length=(
                        self.config.agentlightning.trace_aggregator.trajectory_max_response_length
                        if self.config.agentlightning.trace_aggregator.level.startswith("trajectory")
                        else self.config.data.max_response_length
                    ),
                    device=gen_batch.batch["fake_ids"].device,
                    global_steps=self.global_steps,
                )
                metrics.update(agent_metrics)
                if batch is None:
                    print("Warning: No valid training triplets in this step; skipping optimization update.")
                    metrics["training/skipped_no_triplets"] = 1
                    self.agent_mode_daemon.clear_data_and_server()
                    self.async_rollout_manager.sleep()
                    return metrics
                self.agent_mode_daemon.clear_data_and_server()
                self.async_rollout_manager.sleep()

            if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                with _timer("gen_max", timing_raw):
                    gen_baseline_batch = deepcopy(gen_batch)
                    gen_baseline_batch.meta_info["do_sample"] = False
                    gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                    batch = batch.union(gen_baseline_output)
                    reward_baseline_tensor = self.reward_fn(batch)
                    reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                    batch.batch["reward_baselines"] = reward_baseline_tensor

                    del gen_baseline_batch, gen_baseline_output

            # uid is used for algorithm like GRPO, should be aligned to data id
            batch.non_tensor_batch["uid"] = batch.non_tensor_batch["data_id_list"]

            if "response_mask" not in batch.batch:
                batch.batch["response_mask"] = compute_response_mask(batch)

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

            with _timer("reward", timing_raw):
                # compute reward model score
                if self.use_rm:
                    reward_tensor = self.rm_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                reward_extra_infos_dict = {}

            # for agent mode, pad the lengths to calculate old log prob, ref, and values
            batch, pad_size = pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

            # recompute old_log_probs
            with _timer("old_log_prob", timing_raw):
                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                entropys = old_log_prob.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                metrics.update(old_log_prob_metrics)
                old_log_prob.batch.pop("entropys")
                batch = batch.union(old_log_prob)

            if self.use_reference_policy:
                # compute reference log_prob
                with _timer("ref", timing_raw):
                    ref_log_prob = self._compute_reference_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            # compute values
            if self.use_critic:
                with _timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            # for agent mode, unpad to calculate adv
            # it is important, as adv should be based on the raw traces
            batch = unpad_dataproto(batch, pad_size=pad_size)

            with _timer("adv", timing_raw):
                # if agent_mode is enabled, there is already token_level_scores
                # token_level_scores is not needed to compute here

                if self.config.agentlightning.trace_aggregator.get("level", "transition") == "transition":
                    token_level_scores, omar_metrics = self._compute_triplet_level_omar_scores(batch)
                    batch.batch["token_level_scores"] = token_level_scores
                    metrics.update(omar_metrics)

                # compute rewards. apply_kl_penalty if available
                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(kl_metrics)
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                # compute advantages, executed on the driver process

                norm_adv_by_std_in_grpo = self.config.algorithm.get(
                    "norm_adv_by_std_in_grpo", True
                )  # GRPO adv normalization factor

                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            # Calculate the metrics before processing. Refer to the comments of function `compute_data_metrics` for details.
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, suffix="_before_processing"))

            # after advantages are assigned, we begin to drop (1) long prompt (2) floor to ppo minisize
            keep_indices = (~batch.batch["is_drop_mask"]).nonzero(as_tuple=True)[0]
            metrics["training/n_triplets_prompt_too_long"] = (
                batch.batch["is_drop_mask"].shape[0] - keep_indices.shape[0]
            )
            batch = batch[keep_indices]
            # next, round to minibatch size
            mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            n_transition = len(batch)
            random_indices = list(range(n_transition))
            random.shuffle(random_indices)
            batch.reorder(torch.tensor(random_indices).type(torch.int32))
            n_remained_transition = n_transition // mini_batch_size * mini_batch_size
            batch = batch[list(range(n_remained_transition))]
            metrics["training/n_triplets_dropped_remainder"] = n_transition - n_remained_transition

            # Agent mode note: Change the order of balance batch;
            #     1. first calculate advantage
            #     2. then drop the samples (too long prompt & floor to ppo minisize)
            #     3. balance
            # balance the number of valid tokens on each dp rank.
            # Note that this breaks the order of data inside the batch.
            # Please take care when you implement group based adv computation such as GRPO and rloo
            if self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)

            # update critic
            if self.use_critic:
                with _timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if self.config.trainer.critic_warmup <= self.global_steps:
                # update actor
                with _timer("update_actor", timing_raw):
                    batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

            # Log rollout generations if enabled
            rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
            if rollout_data_dir:
                with _timer("dump_rollout_generations", timing_raw):
                    print(batch.batch.keys())
                    inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                    outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                    scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                    self._dump_generations(
                        inputs=inputs,
                        outputs=outputs,
                        scores=scores,
                        reward_extra_infos_dict=reward_extra_infos_dict,
                        dump_path=rollout_data_dir,
                    )

        # compute training metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic, suffix="_after_processing"))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        # TODO: implement actual tflpo and theoretical tflpo
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

        return metrics

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=_tracking_backends(self.config.trainer.logger),
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Load checkpoint before doing anything, unless explicitly disabled.
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
        verl_version = verl.__version__
        if verl_version == "0.5.0":
            # Note (Zhiyuan): To avoid further patch into vllm async server, using the same sentence to get the naming here.
            # However, it is possible that verl updates the naming and causes incompatibility.
            # Reference: https://github.com/volcengine/verl/blob/5b5e09d9cc20625e436d01f69d9cc739ff681c54/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L217
            model = "/".join(self.config.actor_rollout_ref.model.path.split("/")[-2:])
        else:
            # For other versions (e.g., 0.6.0), we use the full path to the model.
            model = self.config.actor_rollout_ref.model.path
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
            processor=self.processor,  # For Qwen2-VL mrope position_ids
            image_base_dir=getattr(self.config.data, "image_base_dir", None),
            trace_aggregator=self.config.agentlightning.trace_aggregator,
        )
        self.agent_mode_daemon.start()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            for batch_idx, batch_dict in enumerate(self.train_dataloader):
                if epoch == start_epoch and batch_idx < skip_batches_in_first_epoch:
                    continue
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                # train step
                metrics = self._train_step(batch_dict)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with _timer("validate", timing_raw):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                # step metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                # TODO: make a canonical logger that supports various backend
                if _slim_logs_enabled():
                    _print_compact_step_metrics(metrics, self.global_steps)
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()

                    # This exit logic is to ensure a robust CI.
                    pprint(f"Flush the logger...")
                    del logger  # Make sure the loggers are flushed and closed properly
                    pprint(f"Training finished at step {self.global_steps}.")
                    return

                progress_bar.update(1)
                self.global_steps += 1
