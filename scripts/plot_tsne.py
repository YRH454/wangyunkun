#!/usr/bin/env python3
"""Plot prespecified descriptive t-SNE from saved pair representations."""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.manifold import TSNE

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pairmatch.data import DataError, digest, read_json  # noqa: E402
from plot_results import save  # noqa: E402


def plot(directory: Path, output: Path, perplexity: float = 30, iterations: int = 1000,
         seed: int = 42, tiff: bool = False) -> dict:
    manifest = read_json(directory / "manifest.json")
    path = directory / "embeddings.npz"
    if (manifest.get("status") != "COMPLETE" or not manifest.get("prespecified_analysis_user_attestation")
            or digest(path) != manifest.get("embeddings_sha256")):
        raise DataError("feature manifest is incomplete or embeddings were modified")
    if output.exists():
        raise DataError("output directory exists; refusing to overwrite")
    with np.load(path, allow_pickle=False) as data:
        embeddings, labels, ids = data["embeddings"], data["labels"], data["sample_ids"]
    if (embeddings.ndim != 2 or embeddings.shape[1] < 2 or len(embeddings) != manifest["sample_count"]
            or len(labels) != len(ids) or len(ids) != len(embeddings) or len(set(ids.tolist())) != len(ids)
            or not np.isfinite(embeddings).all() or not set(labels.tolist()).issubset({0, 1})
            or not 0 < perplexity < len(embeddings) or iterations < 250):
        raise DataError("invalid embeddings, labels, perplexity or iteration count")
    iteration_argument = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    options = {iteration_argument: iterations, "n_components": 2, "perplexity": perplexity,
               "learning_rate": 200.0, "init": "pca", "random_state": seed, "n_jobs": 1}
    positions = TSNE(**options).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    for label, color, marker in ((0, "#484878", "o"), (1, "#B66D83", "^")):
        selected = labels == label
        ax.scatter(positions[selected, 0], positions[selected, 1], label=f"class {label} (n={int(selected.sum())})",
                   color=color, marker=marker, s=12, alpha=0.65, linewidths=0)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Descriptive embedding projection; not a performance test", fontsize=8)
    ax.legend(fontsize=7)
    source = {"status": "COMPLETE", "figure_contract": "Describe the selected model's pair embeddings without claiming separation proves performance.",
              "archetype": "quantitative grid / descriptive embedding panel", "split": manifest["split"],
              "split_sha256": manifest["split_sha256"], "condition": manifest["condition"],
              "embedding_manifest_sha256": digest(directory / "manifest.json"), "embeddings_sha256": digest(path),
              "sklearn_version": sklearn.__version__, "parameters": {"perplexity": perplexity, "iterations": iterations,
                    "learning_rate": 200, "initialization": "pca", "seed": seed},
              "points": [{"sample_id": str(identifier), "label": int(label), "x": float(x), "y": float(y)}
                         for identifier, label, (x, y) in zip(ids, labels, positions)],
              "limitation": "t-SNE is descriptive; independently fitted projections have no comparable axes."}
    save(fig, output, source, tiff)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiff", action="store_true")
    args = parser.parse_args()
    try:
        report = plot(args.embeddings_dir, args.output_dir, args.perplexity, args.iterations, args.seed, args.tiff)
        print(f"Descriptive t-SNE written for {len(report['points'])} samples; inspect locally.")
        return 0
    except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"t-SNE failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check inputs'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
