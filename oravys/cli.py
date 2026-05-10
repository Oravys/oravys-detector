# Copyright (c) 2026 Oravys Inc. All rights reserved.
"""ORAVYS CLI - voice deepfake detection from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oravys",
        description="ORAVYS - Voice deepfake detection. thousands of engines.",
    )
    sub = parser.add_subparsers(dest="command")

    # detect
    p_detect = sub.add_parser("detect", help="Analyze an audio file")
    p_detect.add_argument("audio", type=str, help="Path to audio file")
    p_detect.add_argument("--api-key", "-k", type=str, help="API key for cloud mode")
    p_detect.add_argument(
        "--mode",
        "-m",
        choices=["api", "local"],
        default="local",
        help="Analysis mode (default: local)",
    )
    p_detect.add_argument(
        "--json", action="store_true", dest="as_json", help="Output JSON"
    )
    p_detect.add_argument(
        "--verbose", "-v", action="store_true", help="Show per-engine results"
    )

    # version
    sub.add_parser("version", help="Show version")

    args = parser.parse_args(argv)

    if args.command == "version":
        from oravys import __version__

        print(f"oravys {__version__}")
        return 0

    if args.command == "detect":
        return _cmd_detect(args)

    parser.print_help()
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    from oravys import Detector

    p = Path(args.audio)
    if not p.exists():
        sys.stderr.write(f"File not found: {p}\n")
        return 1

    mode = args.mode
    api_key = args.api_key
    if api_key:
        mode = "api"

    try:
        det = Detector(api_key=api_key, mode=mode)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    result = det.analyze(p)

    if args.as_json:
        out = {
            "verdict": result.verdict,
            "is_synthetic": result.is_synthetic,
            "confidence": result.confidence,
            "engines_fired": result.engines_fired,
            "engines_total": result.engines_total,
            "duration_seconds": round(result.duration_seconds, 3),
        }
        if args.verbose and result.engine_results:
            out["engines"] = [
                {
                    "name": e.name,
                    "domain": e.domain,
                    "score": round(e.score, 4),
                    "label": e.label,
                    "detail": e.detail,
                }
                for e in result.engine_results
            ]
        print(json.dumps(out, indent=2))
    else:
        print(f"ORAVYS Analysis: {p.name}")
        print(f"  Verdict:    {result.verdict}")
        print(f"  Confidence: {result.confidence_pct}")
        print(f"  Synthetic:  {result.is_synthetic}")
        print(f"  Engines:    {result.engines_fired}/{result.engines_total}")
        print(f"  Duration:   {result.duration_seconds:.2f}s")

        if args.verbose and result.engine_results:
            print()
            print("  Per-engine results:")
            for e in result.engine_results:
                marker = "!" if e.label == "SUSPICIOUS" else " "
                print(
                    f"  {marker} [{e.domain:>14}] {e.name:<25} {e.label:<20} {e.detail}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
