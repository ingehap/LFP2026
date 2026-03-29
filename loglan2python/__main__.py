"""CLI entry-point: ``python -m loglan2python <input.log> [-o output.py]``"""

import argparse
import sys
from pathlib import Path

from .translator import translate_file


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="loglan2python",
        description="Translate a Loglan (Geolog) source file to Python.",
    )
    ap.add_argument("input", metavar="INPUT", help="Loglan source file (.log)")
    ap.add_argument(
        "-o",
        "--output",
        metavar="OUTPUT",
        help="Destination Python file (default: <INPUT>.py)",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else input_path.with_suffix(".py")
    try:
        translate_file(input_path, output_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Translated  {input_path}  →  {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
