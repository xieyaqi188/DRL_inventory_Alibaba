import os
import sys
import argparse
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl.trainer import Trainer
# from rl.trainer_ddpg_critic import Trainer
from utils import fix_seed, setup_stdout_log
from args_parser import parse_args, print_all_hyperparameters

_cli_parser = argparse.ArgumentParser(description="")
# used config file
_cli_parser.add_argument("--configs",
                         default=str(project_root / "config/rl/ddpg_base.yaml"))
_cli_parser.add_argument("--seed", type=int, help="Random seed (overrides config file)")
_cli_parser.add_argument("--skip-full-eval", action="store_true", help="Skip the full evaluation (during hyperparameter search)")


def main():
    args = parse_args(_cli_parser)

    # Tee stdout to log file for single runs (not during hyperparameter search)
    tee = None
    if not args.skip_full_eval:
        config_stem = Path(args.configs).stem
        tee = setup_stdout_log("rl", config_stem)

    print_all_hyperparameters(args)

    fix_seed(args.seed)

    rl_trainer = Trainer(args)
    rl_trainer.train_model()

    best_model = rl_trainer.load_model(args.model_name)

    # Evaluate all three datasets in RL environment
    if not args.skip_full_eval:
        train_metrics = rl_trainer.eval_on_train_set(best_model)
        val_metrics = rl_trainer.eval_on_val_set(best_model)
        test_metrics = rl_trainer.eval_on_test_set(best_model)

    if tee:
        tee.close()


if __name__ == "__main__":
    main()
