# Copyright (c) Microsoft. All rights reserved.

# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Type

import hydra
import ray
from ray.actor import ActorClass
from verl.trainer.main_ppo import create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager

from agentlightning.adapter import TraceAdapter
from agentlightning.llm_proxy import LLMProxy
from agentlightning.store.base import LightningStore
from agentlightning.types import Dataset

from .dataset import AgentDataset, LoadedDataset

if TYPE_CHECKING:
    from .daemon import AgentModeDaemon
    from .trainer import AgentLightningTrainer

__all__ = [
    "main",
    "run_ppo",
    "TaskRunner",
]


@hydra.main(config_path="pkg://agentlightning/verl", config_name="config", version_base=None)
def main(config: Any):
    from .daemon import AgentModeDaemon
    from .trainer import AgentLightningTrainer

    run_ppo(
        config,
        train_dataset=None,
        val_dataset=None,
        store=None,
        llm_proxy=None,
        adapter=None,
        trainer_cls=AgentLightningTrainer,
        daemon_cls=AgentModeDaemon,
    )


def run_ppo(
    config: Any,
    train_dataset: Dataset[Any] | None,
    val_dataset: Dataset[Any] | None,
    store: LightningStore | None,
    llm_proxy: LLMProxy | None,
    adapter: TraceAdapter[Any] | None,
    trainer_cls: Type[AgentLightningTrainer],
    daemon_cls: Type[AgentModeDaemon],
) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        try:
            # verl >= 0.6.0
            num_cpus = config.ray_kwargs.ray_init.num_cpus
        except AttributeError:
            # verl < 0.6.0
            num_cpus = config.ray_init.num_cpus
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "NO_PROXY": "localhost,127.0.0.1,::1,172.21.238.21,172.31.226.176,0.0.0.0",
                    "no_proxy": "localhost,127.0.0.1,::1,172.21.238.21,172.31.226.176,0.0.0.0",
                }
            },
            num_cpus=num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(
        runner.run.remote(  # type: ignore
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            store=store,
            llm_proxy=llm_proxy,
            adapter=adapter,
            trainer_cls=trainer_cls,
            daemon_cls=daemon_cls,
        )
    )


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(
        self,
        config: Any,
        train_dataset: Dataset[Any] | None,
        val_dataset: Dataset[Any] | None,
        store: LightningStore | None,
        llm_proxy: LLMProxy | None,
        adapter: TraceAdapter[Any] | None,
        trainer_cls: Type[AgentLightningTrainer],
        daemon_cls: Type[AgentModeDaemon],
    ):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        if os.environ.get("AGL_SLIM_LOGS", "").lower() not in {"1", "true", "yes", "on"}:
            pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download model/tokenizer sources from fs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        tokenizer_source = config.actor_rollout_ref.model.get("tokenizer_path", config.actor_rollout_ref.model.path)
        tokenizer_local_path = copy_to_local(tokenizer_source)

        # instantiate tokenizer
        from verl.utils.tokenizer import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(tokenizer_local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(tokenizer_local_path, use_fast=True)  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            # FIXME: This import is outdated
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup  # type: ignore
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        try:
            # verl >= 0.6.0
            from verl.trainer.ppo.utils import Role, need_critic
        except ImportError:
            # Fallback for verl <= 0.5.0
            from verl.trainer.ppo.ray_trainer import Role, need_critic  # type: ignore

        use_critic = need_critic(config)

        role_worker_mapping: dict[Role, ActorClass[Any]] = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
        }
        if use_critic:
            role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

        global_pool_id = "global_pool"
        split_actor_critic = bool(config.trainer.get("separate_actor_critic_pool", False))

        if split_actor_critic:
            actor_pool_id = "actor_rollout_pool"
            critic_pool_id = "critic_pool"
            actor_gpus_per_node = int(config.trainer.get("actor_rollout_gpus_per_node", 1))
            critic_gpus_per_node = int(config.trainer.get("critic_gpus_per_node", 1))
            n_gpus_per_node = int(config.trainer.n_gpus_per_node)
            need_critic_pool = use_critic or bool(config.reward_model.enable)

            if actor_gpus_per_node <= 0:
                raise ValueError("actor_rollout_gpus_per_node must be > 0.")
            if need_critic_pool and critic_gpus_per_node <= 0:
                raise ValueError("critic_gpus_per_node must be > 0 when critic or reward model is enabled.")

            requested_gpus_per_node = actor_gpus_per_node + (critic_gpus_per_node if need_critic_pool else 0)
            if requested_gpus_per_node > n_gpus_per_node:
                raise ValueError(
                    "Requested GPUs per node for configured pools cannot exceed trainer.n_gpus_per_node."
                )

            resource_pool_spec = {
                actor_pool_id: [actor_gpus_per_node] * config.trainer.nnodes,
            }
            if need_critic_pool:
                resource_pool_spec[critic_pool_id] = [critic_gpus_per_node] * config.trainer.nnodes
            mapping = {
                Role.ActorRollout: actor_pool_id,
            }
            if use_critic:
                mapping[Role.Critic] = critic_pool_id
        else:
            resource_pool_spec = {
                global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            mapping = {
                Role.ActorRollout: global_pool_id,
            }
            if use_critic:
                mapping[Role.Critic] = global_pool_id

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = critic_pool_id if split_actor_critic else global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = mapping[Role.ActorRollout] if split_actor_critic else global_pool_id

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Use our special dataset
        if train_dataset is None:
            train_dataset = AgentDataset(
                data_files=config.data.train_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        else:
            train_dataset = LoadedDataset(train_dataset)

        if val_dataset is None:
            val_dataset = AgentDataset(
                data_files=config.data.val_files,
                tokenizer=tokenizer,
                processor=processor,
                config=config.data,
            )
        else:
            val_dataset = LoadedDataset(val_dataset)

        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = trainer_cls(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            store=store,
            llm_proxy=llm_proxy,
            adapter=adapter,
            daemon_cls=daemon_cls,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
