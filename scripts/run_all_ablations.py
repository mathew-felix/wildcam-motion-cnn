from __future__ import annotations

"""Run all ablation YAML configs and build one merged benchmark_summary_all.csv.

You already have:
- scripts/eval_vibe_mhi_cnn.py  (the evaluator)
- configs/ablations/*.yaml      (ablations: vibe, vibe_AT, vibe_ALR, vibe_AT_ALR,
  vibe_MHI, vibe_AT_ALR_MHI, vibe_cnn, vibe_AT_ALR_cnn, vibe_MHI_cnn,
  vibe_AT_ALR_MHI_cnn)

This runner:
1) Executes each YAML by calling the evaluator (same behavior as manual runs).
2) Reads each YAML's `benchmark_summary_csv` output.
3) Concatenates all rows into a single merged CSV with an extra `variant` column.

Typical usage (from repo root):
    python scripts/run_all_ablations.py

It will:
- run all configs under configs/ablations/
- write results/benchmark_summary_all.csv
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None
    _YAML_IMPORT_ERROR = e
else:
    _YAML_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]

# Preferred order (so plots/README tables look consistent)
DEFAULT_CONFIG_ORDER = [
    "vibe.yaml",
    "vibe_AT.yaml",
    "vibe_ALR.yaml",
    "vibe_AT_ALR.yaml",
    "vibe_MHI.yaml",
    "vibe_AT_ALR_MHI.yaml",
    "vibe_cnn.yaml",
    "vibe_AT_ALR_cnn.yaml",
    "vibe_MHI_cnn.yaml",
    "vibe_AT_ALR_MHI_cnn.yaml",
]


def _require_yaml() -> None:
    if yaml is None:
        raise ImportError(
            "PyYAML is required for this script. Install with: pip install pyyaml"
        ) from _YAML_IMPORT_ERROR


def load_yaml(path: Path) -> Dict:
    _require_yaml()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict at top-level: {path}")
    return cfg


def resolve_out_path(cfg_dir: Path, out_path: str) -> Path:
    """Resolve an output path the same way the evaluator does."""
    p = Path(out_path)
    if p.is_absolute():
        return p
    return (cfg_dir / p).resolve()


def summary_csv_path_from_cfg(cfg: Dict, cfg_path: Path) -> Path:
    """Determine the per-config benchmark summary path (supports legacy keys)."""
    out = cfg.get("benchmark_summary_csv") or cfg.get("output_csv") or "benchmark_summary.csv"
    return resolve_out_path(cfg_path.parent, str(out))


def run_eval(config_path: Path, evaluator_path: Path, quiet: bool = False) -> None:
    """Run the evaluator for one config."""
    cmd = [sys.executable, str(evaluator_path), "--config", str(config_path)]
    print(f"\n=== Running: {config_path.name} ===")
    print("CMD:", " ".join(cmd))

    if quiet:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


def read_csv_dicts(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV as rows of dicts; returns (header, rows)."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        rows = list(reader)
    return header, rows


def write_csv_dicts(path: Path, header: List[str], rows: List[Dict[str, str]]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def discover_configs(configs_dir: Path, order: Optional[List[str]] = None) -> List[Path]:
    """Return config paths in a stable order."""
    if not configs_dir.exists():
        raise FileNotFoundError(f"configs_dir does not exist: {configs_dir}")

    if order:
        out: List[Path] = []
        for name in order:
            p = (configs_dir / name).resolve()
            if p.exists():
                out.append(p)

        # also include any other yaml files not listed (sorted)
        known = {p.name for p in out}
        extras = sorted([p for p in configs_dir.glob("*.yaml") if p.name not in known])
        out.extend(extras)
        return out

    return sorted(configs_dir.glob("*.yaml"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs_dir",
        default=str(ROOT / "configs" / "ablations"),
        help="Folder containing ablation YAML files (default: configs/ablations).",
    )
    ap.add_argument(
        "--evaluator",
        default=str(ROOT / "scripts" / "eval_vibe_mhi_cnn.py"),
        help="Path to evaluator script (default: scripts/eval_vibe_mhi_cnn.py).",
    )
    ap.add_argument(
        "--merged_csv",
        default=str(ROOT / "results" / "benchmark_summary_all.csv"),
        help="Output merged CSV path (default: results/benchmark_summary_all.csv).",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Hide per-config evaluator output (only shows high-level progress).",
    )
    ap.add_argument(
        "--skip_eval",
        action="store_true",
        help="Do not run evaluation; only merge existing per-variant summary CSVs.",
    )
    ap.add_argument(
        "--no_default_order",
        action="store_true",
        help="Do not force the default ablation order; just sort configs alphabetically.",
    )
    args = ap.parse_args()

    configs_dir = Path(args.configs_dir).resolve()
    evaluator_path = Path(args.evaluator).resolve()
    merged_csv_path = Path(args.merged_csv).resolve()

    if not evaluator_path.exists():
        raise FileNotFoundError(f"Evaluator not found: {evaluator_path}")

    order = None if args.no_default_order else DEFAULT_CONFIG_ORDER
    cfg_paths = discover_configs(configs_dir, order=order)

    if not cfg_paths:
        raise RuntimeError(f"No YAML configs found in: {configs_dir}")

    merged_rows: List[Dict[str, str]] = []
    merged_header: List[str] = []

    for cfg_path in cfg_paths:
        cfg = load_yaml(cfg_path)
        variant = cfg_path.stem

        if not args.skip_eval:
            run_eval(cfg_path, evaluator_path, quiet=bool(args.quiet))

        summary_path = summary_csv_path_from_cfg(cfg, cfg_path)
        if not summary_path.exists():
            raise FileNotFoundError(
                f"Expected summary CSV was not created for {cfg_path.name}: {summary_path}"
            )

        header, rows = read_csv_dicts(summary_path)

        # Add 'variant' and 'config_file' so the merged file is self-explanatory.
        for r in rows:
            r["variant"] = variant
            r["config_file"] = cfg_path.name

        # Build a union header: start with variant/config_file, then keep original fields,
        # then append any new fields from later files.
        if not merged_header:
            merged_header = ["variant", "config_file"] + [h for h in header if h not in {"variant", "config_file"}]
        else:
            for h in header:
                if h not in merged_header and h not in {"variant", "config_file"}:
                    merged_header.append(h)

        merged_rows.extend(rows)

    write_csv_dicts(merged_csv_path, merged_header, merged_rows)
    print(f"\n✅ Wrote merged summary CSV: {merged_csv_path}")
    print(f"Merged {len(cfg_paths)} configs and {len(merged_rows)} total rows.")


if __name__ == "__main__":
    main()
