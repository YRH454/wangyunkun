#!/usr/bin/env python3
"""Compute verified seed-paired differences, exact signed-rank tests and paired bootstrap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pairmatch.analysis import METRICS, bootstrap_difference, load_groups, paired  # noqa: E402
from pairmatch.data import DataError  # noqa: E402
from pairmatch.runner import dump  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", nargs="+", type=Path, required=True, help="Left method metrics.json files")
    parser.add_argument("--right", nargs="+", type=Path, required=True, help="Right method metrics.json files")
    parser.add_argument("--metric", choices=METRICS, default="macro_f1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-dir", type=Path, help="Groups generated from this evaluation split")
    parser.add_argument("--bootstrap-unit", choices=("groups", "samples"), default="groups")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261030)
    parser.add_argument("--allow-different-u-information", action="store_true")
    parser.add_argument("--confirm-prespecified-analysis", action="store_true")
    args = parser.parse_args()
    try:
        if not args.confirm_prespecified_analysis:
            raise DataError("confirm comparison, metric and analysis were prespecified before viewing test results")
        if args.output.exists():
            raise DataError("output exists; refusing to overwrite")
        report, pairs = paired(args.left, args.right, args.metric, args.allow_different_u_information)
        ids = pairs[0][0]["ids"]
        for a, b in pairs:
            if a["ids"] != ids or b["ids"] != ids:
                raise DataError("evaluation sample IDs differ between seeds")
        if args.bootstrap_unit == "groups":
            if args.groups_dir is None:
                raise DataError("group bootstrap needs --groups-dir; sample bootstrap requires explicit choice")
            groups, group_manifest = load_groups(args.groups_dir, report["split_sha256"], ids)
            report["bootstrap_grouping"] = group_manifest
        else:
            if args.groups_dir is not None:
                raise DataError("do not provide groups for an explicitly sample-level bootstrap")
            groups = [[identifier] for identifier in ids]
            report["bootstrap_grouping"] = {"grouping": "individual samples; assumes independent samples",
                                            "warning": "May understate uncertainty when examples share sources or sentences."}
        report["bootstrap"] = bootstrap_difference(pairs, ids, groups, args.metric,
                                                   args.bootstrap_repetitions, args.bootstrap_seed)
        report["prespecified_analysis_user_attestation"] = True
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dump(args.output, report)
        print(f"Paired analysis complete for {len(pairs)} seed(s); output is local.")
        return 0
    except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Paired analysis failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check inputs'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
