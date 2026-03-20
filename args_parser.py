from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import re
import yaml

YamlDict = Dict[str, Dict[str, Any]]
_NUMBER_RE = re.compile(r"^-?\d[\d_]*(?:\.\d[\d_]*)?(?:[eE][-+]?\d+)?$")


def coerce_num(val: Any) -> Any:
    if isinstance(val, str) and _NUMBER_RE.match(val):
        s = val.replace("_", "")
        try:
            return int(s)
        except ValueError:
            return float(s)
    return val

def load_yaml(path: Path) -> YamlDict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def merge_into_namespace(ns: argparse.Namespace, cfg: YamlDict) -> None:
    """Flatten a two-level YAML mapping into attributes of *ns*."""
    for k1, v1 in cfg.items():
        if isinstance(v1, dict):
            for k2, v2 in v1.items():
                setattr(ns, k2, coerce_num(v2))
        else:
            setattr(ns, k1, coerce_num(v1))

def process_includes(base_cfg: YamlDict, root_dir: Path) -> YamlDict:
    """Recursively merge ``includes`` files (depth-first)."""
    merged: YamlDict = {}
    includes = base_cfg.pop("includes", []) or []
    for inc in includes:
        inc_path = (root_dir / inc).expanduser().resolve()
        inc_cfg = process_includes(load_yaml(inc_path), inc_path.parent)
        merged.update(inc_cfg)
    merged.update(base_cfg)  # root overrides included
    return merged

# -----------------------------------------------------------------------------
# PUBLIC FUNCTION
# -----------------------------------------------------------------------------

def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Return an argument namespace with YAML + CLI merged."""
    args = parser.parse_args()

    # Collect CLI-provided args (non-default values) so they take precedence over YAML
    cli_provided = {
        action.dest: getattr(args, action.dest)
        for action in parser._actions
        if action.dest != 'help'
        and getattr(args, action.dest, None) is not None
        and (not action.option_strings or getattr(args, action.dest) != action.default)
    }

    # Load YAML config, then restore CLI overrides
    cfg_path = Path(args.configs).expanduser().resolve()
    full_cfg = process_includes(load_yaml(cfg_path), cfg_path.parent)
    merge_into_namespace(args, full_cfg)
    for key, value in cli_provided.items():
        setattr(args, key, value)

    project_root = Path(__file__).parent
    global_data_directory = getattr(args, 'data_directory', None)
    if 'data_dir' not in cli_provided:
        if global_data_directory:
            args.data_dir = str(project_root / global_data_directory)
        else:
            print("Warning: 'data_directory' not set in config. Please specify it in your YAML config or via --data_dir.")

    from paths import set_data_paths
    set_data_paths(args.data_dir)

    return args

def print_all_hyperparameters(args: argparse.Namespace) -> None:
    print("\n" + "="*80)
    print("LOADED HYPERPARAMETERS FROM CONFIG")
    print("="*80)

    config_items = []
    for key, value in vars(args).items():
        if not key.startswith('_'):
            config_items.append((key, value))
    for key, value in sorted(config_items):
        print(f"{key:30} = {value}")

    print("="*80)
    print(f"Configuration loaded from: {args.configs}")
    print(f"Data directory: {getattr(args, 'data_directory', 'Not set')}")
    print("="*80 + "\n")