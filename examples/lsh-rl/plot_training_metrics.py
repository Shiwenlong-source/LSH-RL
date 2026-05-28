#!/usr/bin/env python3
"""Parse roleplay training logs and visualize step-level metrics."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


STEP_LINE_RE = re.compile(r"step:(\d+)\s+-\s+(.+)$")
PAIR_RE = re.compile(r"([A-Za-z0-9_./-]+):(.+)")
FLOAT_RE = re.compile(r"np\.float64\(([^)]+)\)")


def _to_float(raw: str) -> float | None:
    text = raw.strip()
    match = FLOAT_RE.fullmatch(text)
    if match:
        text = match.group(1).strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_log(log_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = STEP_LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            kv_blob = m.group(2)
            row: Dict[str, float] = {"step": float(step)}
            for segment in kv_blob.split(" - "):
                pair = PAIR_RE.fullmatch(segment.strip())
                if not pair:
                    continue
                key = pair.group(1).strip()
                value = _to_float(pair.group(2))
                if value is not None:
                    row[key] = value
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, float]], out_csv: Path) -> List[str]:
    all_keys = sorted({k for row in rows for k in row.keys()})
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    return all_keys


def _series(rows: List[Dict[str, float]], key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        if key in row:
            xs.append(row["step"])
            ys.append(row[key])
    return xs, ys


def _moving_average(values: List[float], window: int) -> List[float]:
    """Return a trailing moving average with a shorter window near the beginning."""
    if window <= 1:
        return values
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def _plot_series(ax: Any, x: List[float], y: List[float], *, smooth_window: int) -> None:
    if smooth_window > 1 and len(y) > 1:
        ax.plot(x, y, linewidth=1.0, alpha=0.25, label="raw")
        ax.plot(x, _moving_average(y, smooth_window), linewidth=2.0, label=f"MA-{smooth_window}")
        ax.legend(fontsize=8)
    else:
        ax.plot(x, y, linewidth=1.8)


def plot_png(rows: List[Dict[str, float]], out_png: Path, *, smooth_window: int = 1) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    candidates = [
        "training/reward",
        "val/reward",
        "actor/pg_loss",
        "critic/vf_loss",
        "actor/ppo_kl",
        "actor/grad_norm",
        "critic/grad_norm",
        "response_length/mean_after_processing",
        "perf/throughput",
    ]
    selected = [k for k in candidates if any(k in row for row in rows)]
    if not selected:
        return []

    n = len(selected)
    cols = 3
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 3.2 * rows_n))
    if not hasattr(axes, "ravel"):
        axes = [axes]
    flat_axes = list(axes.ravel()) if hasattr(axes, "ravel") else list(axes)

    for idx, key in enumerate(selected):
        ax = flat_axes[idx]
        x, y = _series(rows, key)
        _plot_series(ax, x, y, smooth_window=smooth_window)
        ax.set_title(key, fontsize=10)
        ax.set_xlabel("step")
        ax.grid(alpha=0.25)

    for idx in range(len(selected), len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.suptitle("Roleplay Training Metrics", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return selected


def _safe_metric_filename(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def plot_metric_images(rows: List[Dict[str, float]], out_dir: Path, *, smooth_window: int = 1) -> List[str]:
    """Write one PNG per metric, similar to Swift's images directory style."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    candidates = [
        "training/reward",
        "training/reward_all_triplets",
        "training/triplet_omar_mean_score",
        "training/triplet_omar_mean_long_credit",
        "val/reward",
        "actor/pg_loss",
        "critic/vf_loss",
        "actor/ppo_kl",
        "actor/kl_loss",
        "actor/entropy_loss",
        "actor/grad_norm",
        "critic/grad_norm",
        "critic/advantages/mean_before_processing",
        "perf/throughput",
        "response_length/mean_after_processing",
    ]
    selected = [k for k in candidates if any(k in row for row in rows)]
    if not selected:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for key in selected:
        x, y = _series(rows, key)
        if not x:
            continue
        plt.figure(figsize=(8, 4.5))
        ax = plt.gca()
        _plot_series(ax, x, y, smooth_window=smooth_window)
        plt.title(key)
        plt.xlabel("step")
        plt.grid(alpha=0.25)
        out_file = out_dir / f"{_safe_metric_filename(key)}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=160)
        plt.close()
        written.append(str(out_file))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse train log and visualize key metrics.")
    parser.add_argument("--log-file", type=Path, required=True, help="Path to train_*.log")
    parser.add_argument("--out-dir", type=Path, default=Path("logs/plots"), help="Output directory")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional directory for one-metric-per-image PNG outputs.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Trailing moving-average window. Use 1 to plot raw metrics only.",
    )
    args = parser.parse_args()
    smooth_window = max(1, args.smooth_window)

    rows = parse_log(args.log_file)
    if not rows:
        raise SystemExit(f"No step metric lines found in: {args.log_file}")

    out_csv = args.out_dir / f"{args.log_file.stem}_metrics.csv"
    write_csv(rows, out_csv)
    print(f"[OK] Wrote CSV: {out_csv}")

    out_png = args.out_dir / f"{args.log_file.stem}_metrics.png"
    plotted = plot_png(rows, out_png, smooth_window=smooth_window)
    if plotted:
        print(f"[OK] Wrote PNG: {out_png}")
        print("[PLOTS] " + ", ".join(plotted))
    else:
        print("[WARN] matplotlib unavailable or no plottable keys found; CSV generated only.")

    if args.images_dir is not None:
        images_written = plot_metric_images(rows, args.images_dir, smooth_window=smooth_window)
        if images_written:
            print(f"[OK] Wrote {len(images_written)} metric images to: {args.images_dir}")
        else:
            print("[WARN] No per-metric images generated.")


if __name__ == "__main__":
    main()
