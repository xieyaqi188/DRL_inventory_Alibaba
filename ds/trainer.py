import copy
import time
import torch
import json
import sys
from pathlib import Path

import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ds.neural_network import NeuralNetworkCreator
from ds.dataset import build_dataloaders
from rl.inv_env import InventorySimEnv
from ds.simulation import Simulator
from paths import get_data_file_path
from utils import to_metrics_dict
import paths

# Global variable to switch between tr_eval_loss and val_loss for evaluation
USE_VAL_LOSS_FOR_EVAL = True

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter = float('inf'), 0
        self.best_state = None

    def step(self, metric, model):
        if metric < self.best - self.min_delta:
            self.best, self.counter = metric, 0
            self.best_state = copy.deepcopy(model.state_dict())
            return False  # continue training
        self.counter += 1
        return self.counter >= self.patience


class DLTrainer:
    def __init__(self, cfg, args):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.cfg, self.device, self.args = cfg, device, args
        self.loaders = build_dataloaders(cfg, seed=args.seed, batch_size=args.batch_size)  # train / val / test
        creator = NeuralNetworkCreator()
        self.model = creator.create_neural_network(args, device=device)
        self.criterion = torch.nn.Identity()
        self.opt = None
        self.scheduler = None
        self.early_stop = EarlyStopping(patience=args.patience)

        self.simulator = Simulator(device=device)
    
    def initialize_optimizer_if_needed(self):
        if self.opt is None and list(self.model.parameters()):
            self.opt = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            self.scheduler = ReduceLROnPlateau(self.opt, patience=3, factor=self.args.scheduler_factor)

    # ------------------------------------------------------------------
    def run_epoch(self, loader, train: bool):
        epoch_loss, samples = 0.0, 0
        epoch_turnover, epoch_stockout = 0.0, 0.0
        self.model.train() if train else self.model.eval()

        turnover_loss_ratio = self.args.loss_alpha

        for data in loader:
            data = {k: v.to(self.device) for k, v in data.items()}
            if train and self.opt is not None:
                self.opt.zero_grad()

            states = self.simulator.initialize(data, self.cfg.PROBLEM_PARAMS, self.cfg.STATE_PARAMS)

            for _ in range(self.cfg.PROBLEM_PARAMS['max_period_num']):
                action = self.model(states)
                self.initialize_optimizer_if_needed()
                states, _ = self.simulator.transition(action)

            turnover, stockout = self.simulator.ali_loss_syn(train)

            turnover = turnover.sum()
            stockout = stockout.sum()
            batch_loss = turnover * turnover_loss_ratio + stockout * (1-turnover_loss_ratio)

            epoch_loss += batch_loss.item()
            epoch_turnover += turnover.item()
            epoch_stockout += stockout.item()
            samples += len(data['demands'])

            if train and self.opt is not None:
                if not torch.isfinite(batch_loss):
                    print("Non-finite loss, skip this batch!")
                    continue

                (batch_loss / len(data['demands'])).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
        return epoch_loss / samples, epoch_turnover / samples, epoch_stockout / samples

    def fit(self):
        for epoch in range(self.args.epochs):
            train_loss, train_turnover, train_stockout = self.run_epoch(self.loaders['train'], train=True)
            train_eval_loss, _, _ = self.run_epoch(self.loaders['train'], train=False)
            val_loss, val_turnover, val_stockout = self.run_epoch(self.loaders['val'], train=False)

            eval_metric = val_loss if USE_VAL_LOSS_FOR_EVAL else train_eval_loss
            if self.scheduler is not None:
                self.scheduler.step(eval_metric)

            lr = f"{self.opt.param_groups[0]['lr']:.1e}" if self.opt is not None else "N/A"
            print(f"[{epoch:03d}] train={train_loss:.4f}, turnover={train_turnover:.4f}, stockout={train_stockout:.4f}| "
                  f"val={val_loss:.4f}, turnover={val_turnover:.4f}, stockout={val_stockout:.4f} | eval_train={train_eval_loss:.4f}| "
                  f"lr={lr}")

            if self.early_stop.step(eval_metric, self.model):
                print("Early stopping!")
                break

        if self.early_stop.best_state:
            self.model.load_state_dict(self.early_stop.best_state)

    def eval_on_sku_set(self, sku_list, set_name):
        """Generic evaluation method for any set of SKUs"""
        print(f"\n--------------- Testing in RL env ---------------")
        service_level_list, avg_inv_list, avg_sales_list, turnover_list = [], [], [], []
        overage_list, underage_list, loss_list = [], [], []

        print(f"=== Start evaluating on {len(sku_list)} SKUs {set_name}-set ===")

        for sku in sku_list:
            env = InventorySimEnv(sku_id=sku, args=self.args, action_mode=self.args.action_mode, train=False,
                                  data_path=self.cfg.DATASET_PARAMS.get('file_location'))
            ds_policy = InventorySimEnv.predict_wrapper(self.model, env, device=self.device)
            metrics = env.run_full_episode(ds_policy)
            env.close()

            # Focus on overage, underage in synthetic settings
            service_level, avg_inv, avg_sales, overage, underage, loss, inv_tr = metrics
            service_level_list.append(service_level)
            turnover = avg_inv / (avg_sales + 1e-6)
            turnover = min(turnover, self.args.max_turnover)
            turnover_list.append(turnover)

            # Use overage/underage from compute_episode_metrics
            underage_list.append(underage)
            overage_list.append(overage)
            tmp_loss = overage * self.args.loss_alpha + underage * (1 - self.args.loss_alpha)
            loss_list.append(tmp_loss)

            avg_inv_list.append(avg_inv)
            avg_sales_list.append(avg_sales)

        weighted_loss = np.mean(overage_list)*self.args.loss_alpha + np.mean(underage_list)*(1-self.args.loss_alpha)

        print(f" underage                = {np.mean(underage_list):.4f}")
        print(f" overage                 = {np.mean(overage_list):.4f}")
        print(f" ratio_overage           = {self.args.loss_alpha}")
        print(f" loss                    = {weighted_loss:.4f}\n")

        print(f" service level           = {np.mean(service_level_list):.3f}")
        print(f" turnover days           = {np.mean(turnover_list):.3f}")

        return to_metrics_dict(
            weighted_loss, underage_list, overage_list,
            service_level_list, turnover_list, avg_inv_list, avg_sales_list
        )

    def eval_on_test_set(self):
        """Evaluate on test set"""
        loss, turnover, stockout = self.run_epoch(self.loaders['test'], train=False)
        print(f"\n--------------- Testing in DS env ---------------\n "
              f"DS env +Test: loss={loss:.4f}, turnover={turnover:.4f}, stockout={stockout:.4f}")

        test_skus = paths.load_json(paths.TEST_SKUS_FILE, 'test.json')
        return self.eval_on_sku_set(test_skus, "test")
    
    def eval_on_train_set(self):
        """Evaluate on training set"""
        loss, turnover, stockout = self.run_epoch(self.loaders['train'], train=False)
        print(f"\n--------------- Testing in DS env ---------------\n "
              f"DS env +Train: loss={loss:.4f}, turnover={turnover:.4f}, stockout={stockout:.4f}")

        train_skus = paths.load_json(paths.TRAIN_SKUS_FILE, 'train.json')
        return self.eval_on_sku_set(train_skus, "train")
    
    def eval_on_val_set(self):
        """Evaluate on validation set"""
        loss, turnover, stockout = self.run_epoch(self.loaders['val'], train=False)
        print(f"\n--------------- Testing in DS env ---------------\n "
              f"DS env +Val: loss={loss:.4f}, turnover={turnover:.4f}, stockout={stockout:.4f}")

        val_skus = paths.load_json(paths.VAL_SKUS_FILE, 'val.json')
        return self.eval_on_sku_set(val_skus, "validation")
