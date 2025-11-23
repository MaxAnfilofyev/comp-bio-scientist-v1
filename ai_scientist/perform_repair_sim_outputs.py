"""
CLI entry point to bulk-repair simulation outputs that are missing exported arrays.
"""

import argparse
import json
from pathlib import Path

from ai_scientist.tools.repair_sim_outputs import repair_sim_outputs


def main():
    parser = argparse.ArgumentParser(description="Bulk repair sim outputs using sim_postprocess + validation.")
    parser.add_argument(
        "--manifest",
        dest="manifest_path",
        default=None,
        help="Optional manifest index path or experiment_results directory (defaults to active run manifest).",
    )
    parser.add_argument(
        "--manifest-paths",
        nargs="*",
        dest="manifest_paths",
        default=None,
        help="Explicit sim.json paths to repair (default: auto-scan manifest for sim.json).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Maximum number of entries to process in one call (default 10).",
    )
    parser.add_argument("--force", action="store_true", help="Re-run postprocess even when arrays exist.")
    parser.add_argument(
        "--run-root",
        default=None,
        help="Optional run root to anchor relative paths (defaults to env-configured run).",
    )
    args = parser.parse_args()

    result = repair_sim_outputs(
        manifest_paths=args.manifest_paths,
        batch_size=args.batch_size,
        force=args.force,
        manifest_path=args.manifest_path,
        run_root=args.run_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
