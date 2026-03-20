import math
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback, StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.utils import get_linear_fn
import torch.nn as nn

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl.inv_env import InventorySimEnv
from utils import make_env, add_action_noise, to_metrics_dict
from paths import get_data_file_path
import paths
import torch

# Global variable to switch between train_eval_env and val_env for evaluation
USE_VAL_ENV_FOR_EVAL = True


class Trainer:
    def __init__(self, args):

        import tempfile
        log_dir = tempfile.mkdtemp(prefix=f"{args.model_name}_inv_{args.action_mode}_")

        self.train_skus, self.val_skus, self.test_skus = [], [], []
        self.log_dir = log_dir
        self.data_path = str(get_data_file_path('syn_full.csv'))
        self.action_mode = args.action_mode
        self.args = args

        # datasets
        self.train_skus = paths.load_json(paths.TRAIN_SKUS_FILE, 'train.json')
        self.val_skus = paths.load_json(paths.VAL_SKUS_FILE, 'val.json')
        self.test_skus = paths.load_json(paths.TEST_SKUS_FILE, 'test.json')


    def _extra_callbacks(self):
        """Override in subclasses to add extra training callbacks."""
        return []

    def train_model(self):
        print(f"Training model for action_mode = {self.action_mode}")

        # Create environment for training and validation
        train_env = DummyVecEnv(
            [make_env(self.train_skus, action_mode=self.action_mode, args=self.args)])

        train_eval_env = DummyVecEnv(
            [make_env(self.train_skus, action_mode=self.action_mode, args=self.args, trainenv=False, deterministic_sku_cycle=True)])
        val_env = DummyVecEnv(
                [make_env(self.val_skus, action_mode=self.action_mode, args=self.args, trainenv=False, deterministic_sku_cycle=True)])

        # Early stopping callback
        # note: at least stops after (min_evals + patience) evals
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=self.args.patience, min_evals=5, verbose=1)

        # Choose evaluation environment based on global variable
        eval_env = val_env if USE_VAL_ENV_FOR_EVAL else train_eval_env
        eval_episodes_num = len(self.val_skus) if USE_VAL_ENV_FOR_EVAL else len(self.train_skus)
        eval_cb = EvalCallback(
            eval_env, best_model_save_path=self.log_dir,
            eval_freq=self.args.eval_freq, n_eval_episodes=eval_episodes_num,
            callback_after_eval=stop_cb, verbose=1
        )


        lr_schedule = get_linear_fn(self.args.learning_rate, self.args.lr_min, self.args.lr_fraction)
        initial_bias = getattr(self.args, 'initial_action_bias', 0.5)

        if self.args.model_name == 'ppo':
            clip_schedule = get_linear_fn(self.args.clip_range, self.args.clip_range * 0.5, self.args.lr_fraction)
            
            model = PPO(
                "MlpPolicy", train_env,
                learning_rate=lr_schedule,
                clip_range=clip_schedule,
                n_steps=self.args.n_steps,
                gamma=self.args.gamma,
                gae_lambda=self.args.gae_lambda,
                batch_size=self.args.batch_size,
                policy_kwargs=dict(net_arch=getattr(self.args, 'net_arch', [256, 256])),
                tensorboard_log=None,
                verbose=0,
                target_kl=self.args.target_kl,
                ent_coef=self.args.ent_coef,
                vf_coef=self.args.vf_coef,
                seed=self.args.seed,
                device="cpu"
            )
            
            # Initialize action bias
            self.initialize_ppo_action_bias(model, initial_bias)
            model.learn(total_timesteps=self.args.total_steps, callback=eval_cb)

        else:  # DDPG
            action_noise = add_action_noise(self.args.action_noise, train_env)
            
            model = DDPG(
                "MlpPolicy", train_env,
                learning_rate=lr_schedule,
                batch_size=self.args.batch_size,
                tau=self.args.tau,
                gamma=self.args.gamma,
                policy_kwargs=dict(net_arch=getattr(self.args, 'net_arch', [64, 64])),
                tensorboard_log=None,
                buffer_size=self.args.buffer_size,
                train_freq=self.args.train_freq,
                gradient_steps=self.args.gradient_steps,
                seed=self.args.seed,
                verbose=0, # 0 no print; 1 print rollout
                action_noise=action_noise,
                learning_starts=self.args.learning_starts,
                device="cpu"
            )
            
            # Initialize action bias after model creation
            self.initialize_ddpg_action_bias(model, initial_bias)

            callbacks = [eval_cb] + self._extra_callbacks()
            model.learn(total_timesteps=self.args.total_steps, callback=callbacks)

        return model

    def load_model(self, model_name):
        best_path = os.path.join(self.log_dir, "best_model.zip")
        if not os.path.exists(best_path):
            raise FileNotFoundError(f"Model file not found: {best_path}. Training may have failed.")
        if model_name == 'ppo':
            best_model = PPO.load(best_path, env=None)
        else:
            best_model = DDPG.load(best_path, env=None)

        return best_model

    def eval_on_sku_set(self, agent, sku_list, set_name):
        service_level_list, avg_inv_list, avg_sales_list, turnover_list = [], [], [], []
        overage_list, underage_list, loss_list = [], [], []

        print(f"\n=== Start evaluating on {len(sku_list)} SKUs {set_name}-set ===")

        for sku in sku_list:
            env = InventorySimEnv(sku_id=sku, args=self.args, action_mode=self.action_mode, train=False,
                                  data_path=self.data_path, deterministic_sku_cycle=True)
            metrics = env.roll_with_agent(agent)
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

    def eval_on_test_set(self, agent):
        """Evaluate on test set"""
        return self.eval_on_sku_set(agent, self.test_skus, "test")
    
    def eval_on_train_set(self, agent):
        """Evaluate on training set"""
        return self.eval_on_sku_set(agent, self.train_skus, "train")
    
    def eval_on_val_set(self, agent):
        """Evaluate on validation set"""
        return self.eval_on_sku_set(agent, self.val_skus, "validation")

    @staticmethod
    def _find_last_linear(module):
        for layer in reversed(list(module.modules())):
            if isinstance(layer, torch.nn.Linear):
                return layer
        return None

    def initialize_ppo_action_bias(self, model, initial_bias):
        """Initialize the PPO policy network with action bias."""
        policy = model.policy
        action_net = getattr(policy, 'action_net', None)
        if action_net is None and hasattr(policy, 'mlp_extractor'):
            action_net = self._find_last_linear(getattr(policy.mlp_extractor, 'policy_net', nn.Module()))
        if action_net is None:
            return

        action_space = model.get_env().action_space
        action_range = action_space.high - action_space.low
        desired_action = action_space.low + initial_bias * action_range
        bias_value = torch.tensor(np.atleast_1d(desired_action), dtype=torch.float32)

        with torch.no_grad():
            if action_net.bias.shape[0] >= bias_value.shape[0]:
                action_net.bias[:bias_value.shape[0]] = bias_value
            else:
                action_net.bias.fill_(bias_value.mean().item())
        print(f"Initialized PPO action bias to {initial_bias * 100:.1f}% of action space "
              f"[{action_space.low}, {action_space.high}]; Applied action: {desired_action}")

    def initialize_ddpg_action_bias(self, model, initial_bias):
        """Initialize the DDPG actor network with action bias."""
        actor = getattr(model.policy, 'actor', None)
        if actor is None:
            return
        action_net = self._find_last_linear(actor)
        if action_net is None:
            return

        action_space = model.get_env().action_space
        action_low = float(action_space.low.flatten()[0])
        action_high = float(action_space.high.flatten()[0])
        action_range = action_high - action_low
        desired_tanh = initial_bias * 2 - 1
        bias_value = math.atanh(desired_tanh) if abs(desired_tanh) < 0.99 else desired_tanh * 3.0

        with torch.no_grad():
            if action_net.bias is not None:
                action_net.bias += torch.tensor([bias_value], dtype=torch.float32)
            else:
                action_net.bias = torch.nn.Parameter(torch.tensor([bias_value], dtype=torch.float32))

        rescaled = (math.tanh(bias_value) + 1) / 2 * action_range + action_low
        print(f"Initialized DDPG action bias to {initial_bias * 100:.1f}% of action space "
              f"[{action_low}, {action_high}]; Applied bias: {bias_value:.3f} (rescaled action = {rescaled:.3f})\n")
