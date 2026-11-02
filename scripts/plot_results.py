#!/usr/bin/env python3
"""Plot traceable learning curves, LDP comparisons or confusion matrices from local results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pairmatch.analysis import METRICS, load_result, metric_value  # noqa: E402
from pairmatch.data import DataError, digest  # noqa: E402
from pairmatch.runner import dump  # noqa: E402

COLORS = ("#484878", "#7884B4", "#B66D83", "#42949E", "#9A4D8E", "#606060")


def save(fig, output: Path, source: dict, tiff: bool) -> None:
    if output.exists():
        raise DataError("figure directory exists; refusing to overwrite")
    output.mkdir(parents=True, exist_ok=False)
    fig.savefig(output / "figure.svg", bbox_inches="tight")
    fig.savefig(output / "figure.pdf", bbox_inches="tight")
    fig.savefig(output / "preview.png", dpi=200, bbox_inches="tight")
    if tiff:
        fig.savefig(output / "figure.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    dump(output / "source_data.json", source)


def plot_ldp(paths: list[Path], metric: str, output: Path, tiff: bool = False) -> dict:
    if metric not in METRICS or not paths:
        raise DataError("a supported metric and results are required")
    items = [load_result(path) for path in paths]
    first = items[0]["result"]
    expected_parent = first["training_provenance"]["split_manifest_sha256"]
    expected_eval = first["split_sha256"]
    records: dict[tuple[str, int, int], dict] = {}
    l_hashes: dict[tuple[int, int], str] = {}
    method_configs: dict[str, dict] = {}
    method_implementations: dict[str, dict] = {}
    for item in items:
        result = item["result"]
        config, prov = result["training_config"], result["training_provenance"]
        if (result["split"] != first["split"] or result["split_sha256"] != expected_eval
                or prov["split_manifest_sha256"] != expected_parent):
            raise DataError("plot input mixes evaluation or parent splits")
        key = (config["condition"], prov["labeled_percent"], config["seed"])
        fixed = {k: v for k, v in config.items() if k != "seed"}
        if key[0] in method_configs and (fixed != method_configs[key[0]]
                or prov["implementation_sha256"] != method_implementations[key[0]]):
            raise DataError("a method mixes training configurations or code versions")
        method_configs[key[0]] = fixed
        method_implementations[key[0]] = prov["implementation_sha256"]
        if key in records:
            raise DataError("duplicate condition, fraction and seed")
        sample_key = (key[1], key[2])
        if sample_key in l_hashes and l_hashes[sample_key] != prov["labeled_sha256"]:
            raise DataError("methods use different labeled samples for the same fraction and seed")
        l_hashes[sample_key] = prov["labeled_sha256"]
        records[key] = {"condition": key[0], "labeled_percent": key[1], "seed": key[2],
                        "value": metric_value(item["metrics"], metric), "labeled_count": prov["labeled_count"],
                        "unlabeled_count": prov["u_count_used"], "metrics_sha256": item["metrics_sha256"],
                        "predictions_sha256": item["predictions_sha256"]}
    methods = sorted({key[0] for key in records})
    if len(methods) > len(COLORS):
        raise DataError("plot at most six methods in a single panel")
    fig, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
    summaries = []
    for index, method in enumerate(methods):
        fractions = sorted({key[1] for key in records if key[0] == method})
        xs, means, errors = [], [], []
        for fraction in fractions:
            rows = [value for key, value in records.items() if key[0] == method and key[1] == fraction]
            values = [r["value"] for r in rows]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else None
            xs.append(fraction)
            means.append(mean)
            errors.append(sd if sd is not None else 0.0)
            summaries.append({"condition": method, "labeled_percent": fraction, "n_seeds": len(values),
                              "mean": mean, "sample_sd_ddof_1": sd})
            for offset, row in enumerate(sorted(rows, key=lambda x: x["seed"])):
                jitter = (offset - (len(rows) - 1) / 2) * 0.28
                ax.scatter(fraction + jitter, row["value"], s=10, alpha=0.5, color=COLORS[index], zorder=2)
        ax.errorbar(xs, means, yerr=errors, marker="o", markersize=3.5, linewidth=1.0, capsize=2,
                    color=COLORS[index], label=method, zorder=3)
    ax.set_xlabel("Labeled training fraction (%)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)
    ax.set_title(f"{first['split']} set; points: seeds, bars: sample SD", fontsize=8)
    source = {"status": "COMPLETE", "figure_contract": "Compare measured performance across fractions and methods without assuming superiority.",
              "archetype": "quantitative grid / one comparison panel", "metric": metric, "split": first["split"],
              "split_sha256": expected_eval, "split_manifest_sha256": expected_parent,
              "uncertainty": "sample standard deviation across seeds; one-seed conditions have no SD",
              "summaries": summaries, "runs": sorted(records.values(), key=lambda row: (row["condition"], row["labeled_percent"], row["seed"])),
              "limitation": "Training-information differences, budgets, sample counts and pretraining must be reported separately."}
    save(fig, output, source, tiff)
    return source


def load_log(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def plot_training(run: Path, output: Path, tiff: bool = False) -> dict:
    training = load_log(run / "train_log.jsonl")
    validation = load_log(run / "validation_log.jsonl")
    with (run / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("status") != "COMPLETE" or not training or not validation:
        raise DataError("complete training and validation logs are required")
    fig, (loss_ax, score_ax) = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True, layout="constrained")
    for stage in sorted({row["stage"] for row in training}):
        tr = [row for row in training if row["stage"] == stage]
        val = [row for row in validation if row["stage"] == stage]
        color = COLORS[(stage - 1) % len(COLORS)]
        loss_ax.plot([r["global_executed_step"] for r in tr], [r["total"] for r in tr], color=color, alpha=0.65,
                     linewidth=0.8, label=f"stage {stage} train objective")
        loss_ax.plot([r["global_executed_step"] for r in val], [r["cross_entropy"] for r in val],
                     color=color, linestyle="--", marker="o", markersize=2, label=f"stage {stage} validation CE")
        score_ax.plot([r["global_executed_step"] for r in val], [r["macro_f1"] for r in val],
                      color=color, marker="o", markersize=2, linewidth=0.9)
        best = summary["stages"][stage - 1]["best_step"]
        selected = [r for r in val if r["stage_step"] == best]
        fixed_pretraining = summary["stages"][stage - 1].get("selection") == "fixed_final_pretraining_step_without_validation_selection"
        if not fixed_pretraining and len(selected) != 1:
            raise DataError("best stage checkpoint does not match the validation log")
        if selected:
            score_ax.scatter([selected[0]["global_executed_step"]], [selected[0]["macro_f1"]], marker="*", s=50, color=color, zorder=4)
        if stage > 1:
            boundary = tr[0]["global_executed_step"] - 0.5
            for ax in (loss_ax, score_ax):
                ax.axvline(boundary, color="#767676", linestyle=":", linewidth=0.8)
    loss_ax.set_ylabel("Loss (different objectives)")
    loss_ax.legend(fontsize=6, loc="best")
    score_ax.set_ylabel("Validation macro-F1")
    score_ax.set_xlabel("Executed optimizer updates")
    score_ax.set_ylim(-0.02, 1.02)
    score_ax.set_title("Star: selected validation checkpoint", fontsize=8)
    source = {"status": "COMPLETE", "figure_contract": "Show stage objectives and validation selection without using test data.",
              "archetype": "quantitative grid / aligned training and validation panels",
              "train_log_sha256": digest(run / "train_log.jsonl"), "validation_log_sha256": digest(run / "validation_log.jsonl"),
              "summary_sha256": digest(run / "summary.json"), "train_log": training, "validation_log": validation,
              "stages": summary["stages"], "note": "Training total and validation cross-entropy differ in objective; stage reset/rollback is indicated."}
    save(fig, output, source, tiff)
    return source


def plot_confusion(path: Path, output: Path, tiff: bool = False) -> dict:
    item = load_result(path)
    matrix = item["metrics"]["confusion_matrix_true_rows_predicted_columns"]
    fig, ax = plt.subplots(figsize=(3.0, 2.8), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues")
    maximum = max(max(row) for row in matrix)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="white" if matrix[i][j] > maximum / 2 else "black")
    ax.set_xticks((0, 1), ("0", "1"))
    ax.set_yticks((0, 1), ("0", "1"))
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{item['result']['split']} set; n={len(item['ids'])}")
    fig.colorbar(image, ax=ax, label="Pairs", shrink=0.8)
    source = {"status": "COMPLETE", "figure_contract": "Show all binary prediction outcomes for one selected run.",
              "archetype": "quantitative grid / confusion matrix", "metrics_sha256": item["metrics_sha256"],
              "predictions_sha256": item["predictions_sha256"], "matrix_true_rows_predicted_columns": matrix,
              "metrics": item["metrics"], "sample_count": len(item["ids"])}
    save(fig, output, source, tiff)
    return source


def plot_length(path: Path, output: Path, tiff: bool = False) -> dict:
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "COMPLETE" or not report.get("prespecified_user_attestation"):
        raise DataError("a complete prespecified length analysis is required")
    bins = report["bins"]
    fig, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
    labels = []
    for index, row in enumerate(bins):
        upper = "∞" if row["right_exclusive"] is None else str(row["right_exclusive"])
        labels.append(f"{row['left_inclusive']}–{upper}\nn={row['pair_count']}\npos={row['positive_count']}")
        value, interval = row["positive_recall"], row["positive_recall_wilson_95_iid_assumption"]
        if value is not None:
            ax.errorbar(index, value, yerr=[[value - interval[0]], [interval[1] - value]],
                        marker="o", markersize=4, capsize=2, color=COLORS[0])
        else:
            ax.text(index, 0.04, "NA", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlabel("Mean pair length (untruncated characters)")
    ax.set_ylabel("Positive recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("95% Wilson interval (independent-pair assumption)", fontsize=7)
    source = {"status": "COMPLETE", "figure_contract": "Show positive recall by prespecified sentence length with counts and uncertainty.",
              "archetype": "quantitative grid / subgroup interval panel", "analysis_sha256": digest(path),
              "bins": bins, "limitation": report["limitation"]}
    save(fig, output, source, tiff)
    return source


def plot_ablation(paths: list[Path], metric: str, output: Path, tiff: bool = False) -> dict:
    if metric not in METRICS or not paths:
        raise DataError("supported metric and results are required")
    items = [load_result(path) for path in paths]
    first = items[0]["result"]
    parent = first["training_provenance"]["split_manifest_sha256"]
    fraction = first["training_provenance"]["labeled_percent"]
    records = {}
    labeled_by_seed = {}
    for item in items:
        result, metrics = item["result"], item["metrics"]
        config, provenance = result["training_config"], result["training_provenance"]
        if (result["split"] != first["split"] or result["split_sha256"] != first["split_sha256"]
                or provenance["split_manifest_sha256"] != parent or provenance["labeled_percent"] != fraction):
            raise DataError("ablation plot must use the same split and labeled fraction")
        key = (config["condition"], config["seed"])
        if key in records:
            raise DataError("duplicate condition and seed")
        if key[1] in labeled_by_seed and labeled_by_seed[key[1]] != provenance["labeled_sha256"]:
            raise DataError("ablation conditions use different labeled examples for a seed")
        labeled_by_seed[key[1]] = provenance["labeled_sha256"]
        records[key] = {"condition": key[0], "seed": key[1], "value": metric_value(metrics, metric),
                        "labeled_count": provenance["labeled_count"], "unlabeled_count": provenance["u_count_used"],
                        "metrics_sha256": item["metrics_sha256"], "predictions_sha256": item["predictions_sha256"]}
    conditions = sorted({key[0] for key in records})
    seed_sets = [{seed for condition, seed in records if condition == method} for method in conditions]
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise DataError("ablation conditions do not contain the same seeds")
    fig, ax = plt.subplots(figsize=(3.5, max(2.8, 0.33 * len(conditions) + 1.2)), layout="constrained")
    summaries = []
    for index, condition in enumerate(conditions):
        rows = [records[(condition, seed)] for seed in sorted(seed_sets[0])]
        values = [row["value"] for row in rows]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else None
        for offset, row in enumerate(rows):
            ax.scatter(row["value"], index + (offset - (len(rows) - 1) / 2) * 0.06,
                       s=12, color="#7884B4", alpha=0.7, zorder=2)
        ax.errorbar(mean, index, xerr=sd if sd is not None else 0, marker="D", markersize=3.5,
                    capsize=2, linewidth=0.9, color="#484878", zorder=3)
        summaries.append({"condition": condition, "n_seeds": len(values), "mean": mean, "sample_sd_ddof_1": sd})
    ax.set_yticks(range(len(conditions)), conditions)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(f"{first['split']}, {fraction}% L; points: seeds, bars: sample SD", fontsize=8)
    source = {"status": "COMPLETE", "figure_contract": "Compare ablation outcomes without assuming any component improves performance.",
              "archetype": "quantitative grid / matched-condition comparison", "metric": metric,
              "split": first["split"], "split_sha256": first["split_sha256"], "labeled_percent": fraction,
              "summaries": summaries, "runs": sorted(records.values(), key=lambda row: (row["condition"], row["seed"])),
              "limitation": "Stage order, U information, realized updates, and mask strength may differ; interpret the design and logs."}
    save(fig, output, source, tiff)
    return source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    ldp = sub.add_parser("ldp")
    ldp.add_argument("--metrics", nargs="+", type=Path, required=True)
    ldp.add_argument("--metric", choices=METRICS, default="macro_f1")
    training = sub.add_parser("training")
    training.add_argument("--run-dir", type=Path, required=True)
    confusion = sub.add_parser("confusion")
    confusion.add_argument("--metrics", type=Path, required=True)
    length = sub.add_parser("length")
    length.add_argument("--length-summary", type=Path, required=True)
    ablation = sub.add_parser("ablation")
    ablation.add_argument("--metrics", nargs="+", type=Path, required=True)
    ablation.add_argument("--metric", choices=METRICS, default="macro_f1")
    for child in (ldp, training, confusion, length, ablation):
        child.add_argument("--output-dir", type=Path, required=True)
        child.add_argument("--tiff", action="store_true", help="Also export 600-dpi TIFF")
    args = parser.parse_args()
    try:
        if args.kind == "ldp":
            plot_ldp(args.metrics, args.metric, args.output_dir, args.tiff)
        elif args.kind == "training":
            plot_training(args.run_dir, args.output_dir, args.tiff)
        elif args.kind == "confusion":
            plot_confusion(args.metrics, args.output_dir, args.tiff)
        elif args.kind == "length":
            plot_length(args.length_summary, args.output_dir, args.tiff)
        else:
            plot_ablation(args.metrics, args.metric, args.output_dir, args.tiff)
        print("Figure and traceable source data saved locally; visually inspect before submission.")
        return 0
    except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Plot failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check inputs and output'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
