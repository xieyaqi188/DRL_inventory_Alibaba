import os
import sys
import argparse
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
torch.set_num_threads(4)

from ds.trainer import DLTrainer
from ds.configs import DLConfig as Config
from utils import fix_seed, setup_stdout_log
from args_parser import parse_args, print_all_hyperparameters

_cli_parser = argparse.ArgumentParser(description="")
# used config file
_cli_parser.add_argument("--configs",
                         default=str(project_root / "config/ds/ds_base.yaml"))
_cli_parser.add_argument("--seed", type=int, help="Random seed (overrides config file)")
_cli_parser.add_argument("--skip-full-eval", action="store_true", help="Skip the full evaluation (during hyperparameter search)")


def main():
    args = parse_args(_cli_parser)

    # Tee stdout to log file for single runs (not during hyperparameter search)
    tee = None
    if not args.skip_full_eval:
        config_stem = Path(args.configs).stem
        tee = setup_stdout_log("ds", config_stem)

    print_all_hyperparameters(args)

    fix_seed(args.seed)
    trainer = DLTrainer(Config, args)
    trainer.fit()

    # Evaluate all three datasets in RL environment
    if not args.skip_full_eval:
        train_metrics = trainer.eval_on_train_set()
        val_metrics = trainer.eval_on_val_set()
        test_metrics = trainer.eval_on_test_set()

    if tee:
        tee.close()


if __name__ == "__main__":
    main()
