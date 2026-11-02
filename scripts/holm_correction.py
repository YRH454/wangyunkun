#!/usr/bin/env python3
"""Apply Holm correction to an explicitly declared family of paired tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pairmatch.analysis import holm  # noqa: E402
from pairmatch.data import DataError, digest  # noqa: E402
from pairmatch.runner import dump  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--family-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-complete-prespecified-family", action="store_true")
    args = parser.parse_args()
    try:
        if not args.confirm_complete_prespecified_family or not args.family_name.strip():
            raise DataError("confirm the complete predeclared hypothesis family")
        if args.output.exists() or len({path.resolve() for path in args.reports}) != len(args.reports):
            raise DataError("output exists or a report is repeated")
        reports = []
        for path in args.reports:
            with path.open(encoding="utf-8") as handle:
                report = json.load(handle)
            if report.get("status") != "COMPLETE" or not report.get("prespecified_analysis_user_attestation"):
                raise DataError("invalid or non-prespecified paired report")
            reports.append((report, digest(path)))
        adjusted = holm([report["signed_rank_exact"]["p_two_sided"] for report, _ in reports])
        result = {"status": "COMPLETE", "family_name": args.family_name, "method": "Holm step-down",
                  "family_complete_user_attestation": True,
                  "tests": [{"report_sha256": sha, "metric": report["metric"], "left": report["left_method"],
                             "right": report["right_method"], "p_raw": report["signed_rank_exact"]["p_two_sided"],
                             "p_holm": value} for (report, sha), value in zip(reports, adjusted)]}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dump(args.output, result)
        print(f"Holm correction complete for {len(reports)} declared tests.")
        return 0
    except (DataError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Correction failed ({type(exc).__name__}): {exc if isinstance(exc, DataError) else 'check reports'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
