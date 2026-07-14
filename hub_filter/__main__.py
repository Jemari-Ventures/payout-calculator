"""CLI: python -m hub_filter --hub pen353"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter JMS exports into Template JMR–shaped workbooks per hub."
    )
    parser.add_argument(
        "--hub",
        required=True,
        help="Hub config stem under hubs/ (e.g. pen353, pen364)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit path to hub JSON config",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    config_path = args.config or (root / "hubs" / f"{args.hub}.json")
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}", file=sys.stderr)
        return 1

    # Allow `python -m hub_filter` from payout-calculator/
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from hub_filter.pipeline import run_hub_filter

    run_hub_filter(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
