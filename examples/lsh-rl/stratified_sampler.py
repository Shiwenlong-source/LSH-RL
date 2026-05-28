#!/usr/bin/env python3
"""Stratified sampling from difficulty-tiered scene directories.

This module implements sampling without replacement from low/medium/high difficulty
scene pools, ensuring balanced batches with configurable ratios.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from train_roleplay_agent import SceneTask, _load_tasks

logger = logging.getLogger(__name__)


class StratifiedSceneSampler:
    """Sample scenes from difficulty tiers without replacement.

    Args:
        train_data_dir: Path to directory containing low/, medium/, high/ subdirectories
        low_ratio: Proportion of low-difficulty scenes in each batch
        medium_ratio: Proportion of medium-difficulty scenes in each batch
        high_ratio: Proportion of high-difficulty scenes in each batch
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        train_data_dir: Path,
        low_ratio: float = 0.4,
        medium_ratio: float = 0.4,
        high_ratio: float = 0.2,
        seed: int = 42,
    ):
        # Normalize ratios to sum to 1.0
        total = low_ratio + medium_ratio + high_ratio
        self.low_ratio = low_ratio / total
        self.medium_ratio = medium_ratio / total
        self.high_ratio = high_ratio / total

        self.rng = random.Random(seed)

        # Load scene file lists from each tier
        self.low_files = sorted((train_data_dir / "low").glob("*.json"))
        self.medium_files = sorted((train_data_dir / "medium").glob("*.json"))
        self.high_files = sorted((train_data_dir / "high").glob("*.json"))

        # Shuffle for randomness
        self.rng.shuffle(self.low_files)
        self.rng.shuffle(self.medium_files)
        self.rng.shuffle(self.high_files)

        # Tracking indices for sampling without replacement
        self.low_idx = 0
        self.medium_idx = 0
        self.high_idx = 0

        logger.info(
            f"Initialized StratifiedSceneSampler: "
            f"low={len(self.low_files)}, medium={len(self.medium_files)}, high={len(self.high_files)}"
        )
        logger.info(
            f"Sampling ratios: low={self.low_ratio:.2f}, "
            f"medium={self.medium_ratio:.2f}, high={self.high_ratio:.2f}"
        )

    def get_pool_sizes(self) -> Dict[str, int]:
        """Return remaining available scenes in each tier."""
        return {
            "low": max(0, len(self.low_files) - self.low_idx),
            "medium": max(0, len(self.medium_files) - self.medium_idx),
            "high": max(0, len(self.high_files) - self.high_idx),
        }

    def sample_batch(
        self,
        batch_size: int,
        evaluator_base_url: str,
        evaluator_api_key: str,
        evaluator_model: str,
    ) -> List[SceneTask]:
        """Sample a stratified batch of scene tasks.

        Args:
            batch_size: Total number of scenes to sample
            evaluator_base_url: Evaluator LLM endpoint
            evaluator_api_key: Evaluator API key
            evaluator_model: Evaluator model name

        Returns:
            List of SceneTask objects sampled from difficulty tiers

        Raises:
            ValueError: If insufficient scenes remain in any tier
        """
        if batch_size < 1:
            return []

        # Calculate desired counts from each tier.
        # Keep zero-ratio tiers at zero, and ensure final sum equals batch_size.
        ratios = [self.low_ratio, self.medium_ratio, self.high_ratio]
        raw_counts = [batch_size * ratio for ratio in ratios]
        counts = [int(raw) for raw in raw_counts]

        # Distribute leftover by largest fractional part among tiers with positive ratio.
        remainder = batch_size - sum(counts)
        if remainder > 0:
            order = sorted(
                range(3),
                key=lambda idx: ((raw_counts[idx] - counts[idx]), ratios[idx]),
                reverse=True,
            )
            for idx in order:
                if remainder == 0:
                    break
                if ratios[idx] > 0:
                    counts[idx] += 1
                    remainder -= 1

        # Fallback for edge cases where all ratios are zero after normalization.
        if sum(counts) == 0 and batch_size > 0:
            counts[1] = batch_size  # default to medium

        n_low, n_medium, n_high = counts

        # Check availability
        pool_sizes = self.get_pool_sizes()
        if pool_sizes["low"] < n_low:
            raise ValueError(
                f"Insufficient low-difficulty scenes: need {n_low}, "
                f"have {pool_sizes['low']} remaining"
            )
        if pool_sizes["medium"] < n_medium:
            raise ValueError(
                f"Insufficient medium-difficulty scenes: need {n_medium}, "
                f"have {pool_sizes['medium']} remaining"
            )
        if pool_sizes["high"] < n_high:
            raise ValueError(
                f"Insufficient high-difficulty scenes: need {n_high}, "
                f"have {pool_sizes['high']} remaining"
            )

        # Sample files from each tier
        low_batch = self.low_files[self.low_idx : self.low_idx + n_low]
        medium_batch = self.medium_files[self.medium_idx : self.medium_idx + n_medium]
        high_batch = self.high_files[self.high_idx : self.high_idx + n_high]

        # Update indices
        self.low_idx += n_low
        self.medium_idx += n_medium
        self.high_idx += n_high

        # Combine and shuffle to avoid ordering bias
        all_files = list(low_batch) + list(medium_batch) + list(high_batch)
        self.rng.shuffle(all_files)

        # Load tasks
        tasks = _load_tasks(
            scene_files=all_files,
            evaluator_base_url=evaluator_base_url,
            evaluator_api_key=evaluator_api_key,
            evaluator_model=evaluator_model,
        )

        # Also shuffle tasks (though file shuffle should suffice)
        self.rng.shuffle(tasks)

        logger.info(
            f"Sampled stratified batch: {len(tasks)} tasks "
            f"(low={len(low_batch)}, medium={len(medium_batch)}, high={len(high_batch)})"
        )

        return tasks

    def is_exhausted(self) -> bool:
        """Check if any tier is exhausted."""
        pool_sizes = self.get_pool_sizes()
        return any(size == 0 for size in pool_sizes.values())

    def get_total_remaining(self) -> int:
        """Get total remaining scenes across all tiers."""
        return sum(self.get_pool_sizes().values())

    def reset(self, seed: int | None = None) -> None:
        """Reset the sampler to start over (with optional new seed)."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.rng.shuffle(self.low_files)
        self.rng.shuffle(self.medium_files)
        self.rng.shuffle(self.high_files)

        self.low_idx = 0
        self.medium_idx = 0
        self.high_idx = 0

        logger.info("StratifiedSceneSampler reset to initial state")


def load_stratified_datasets(
    train_data_dir: Path,
    batch_size: int,
    num_batches: int,
    evaluator_base_url: str,
    evaluator_api_key: str,
    evaluator_model: str,
    low_ratio: float = 0.4,
    medium_ratio: float = 0.4,
    high_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[SceneTask], List[SceneTask]]:
    """Load training and validation datasets using stratified sampling.

    This function creates multiple batches by repeatedly calling the sampler,
    then separates the last batch for validation.

    Args:
        train_data_dir: Path to train_data/ with low/, medium/, high/ subdirs
        batch_size: Number of scenes per batch (e.g., 5)
        num_batches: Total number of batches to create
        evaluator_base_url: Evaluator LLM endpoint
        evaluator_api_key: Evaluator API key
        evaluator_model: Evaluator model name
        low_ratio: Sampling ratio for low difficulty
        medium_ratio: Sampling ratio for medium difficulty
        high_ratio: Sampling ratio for high difficulty
        seed: Random seed

    Returns:
        (train_tasks, val_tasks) tuple

    Example:
        With batch_size=5, num_batches=20, ratios=2:2:1:
        - Creates 20 batches total (100 scenes)
        - Each batch: ~2 low + ~2 medium + ~1 high
        - Last batch (batch 20) becomes validation set
        - First 19 batches become training set
    """
    sampler = StratifiedSceneSampler(
        train_data_dir=train_data_dir,
        low_ratio=low_ratio,
        medium_ratio=medium_ratio,
        high_ratio=high_ratio,
        seed=seed,
    )

    train_tasks: List[SceneTask] = []
    val_tasks: List[SceneTask] = []

    for batch_idx in range(num_batches):
        try:
            batch = sampler.sample_batch(
                batch_size=batch_size,
                evaluator_base_url=evaluator_base_url,
                evaluator_api_key=evaluator_api_key,
                evaluator_model=evaluator_model,
            )

            # Last batch goes to validation
            if batch_idx == num_batches - 1:
                val_tasks.extend(batch)
                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches} -> VALIDATION ({len(batch)} tasks)"
                )
            else:
                train_tasks.extend(batch)
                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches} -> TRAINING ({len(batch)} tasks)"
                )

        except ValueError as exc:
            logger.error(f"Failed to sample batch {batch_idx + 1}: {exc}")
            raise

    logger.info(
        f"Stratified sampling complete: "
        f"train={len(train_tasks)} tasks, val={len(val_tasks)} tasks, "
        f"total={len(train_tasks) + len(val_tasks)} tasks"
    )

    return train_tasks, val_tasks


if __name__ == "__main__":
    # Test the sampler
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    train_data_dir = Path("train_data")
    if not train_data_dir.exists():
        sys.exit(f"train_data directory not found: {train_data_dir}")

    # Test parameters
    BATCH_SIZE = 5
    NUM_BATCHES = 10  # Total 50 scenes for testing

    print(f"\n{'='*60}")
    print(f"Testing Stratified Sampling")
    print(f"{'='*60}")
    print(f"Train data dir: {train_data_dir}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num batches: {NUM_BATCHES}")
    print(f"Sampling ratios: 2:2:1 (low:medium:high)")
    print(f"{'='*60}\n")

    train_tasks, val_tasks = load_stratified_datasets(
        train_data_dir=train_data_dir,
        batch_size=BATCH_SIZE,
        num_batches=NUM_BATCHES,
        evaluator_base_url="xxx",
        evaluator_api_key="xxx",
        evaluator_model="xxx",
        low_ratio=0.4,
        medium_ratio=0.4,
        high_ratio=0.2,
        seed=42,
    )

    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"Training tasks: {len(train_tasks)}")
    print(f"Validation tasks: {len(val_tasks)}")
    print(f"Total: {len(train_tasks) + len(val_tasks)}")
    print(f"{'='*60}\n")
