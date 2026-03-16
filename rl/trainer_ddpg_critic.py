import sys
from pathlib import Path

from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl.trainer import Trainer as BaseTrainer


class CriticLossCallback(BaseCallback):
    """Collect DDPG critic loss after each gradient update, and print every 500 timesteps."""

    def __init__(self, print_freq=500, state_log_path=None, state_log_steps=10000, verbose=0, args=None):
        super().__init__(verbose)
        self.critic_losses = []
        self.timesteps = []
        self.periodic_losses = []      # losses sampled at each print_freq interval
        self.periodic_timesteps = []   # corresponding timesteps
        self._last_n_updates = -1
        self._print_freq = print_freq
        self._last_print_timestep = 0
        if state_log_path is None:
            log_dir = project_root / "rl" / "logs_ddpg_critic_loss"
            log_dir.mkdir(parents=True, exist_ok=True)
            method = getattr(args, 'model_name', 'ddpg') if args else 'ddpg'
            action_mode = getattr(args, 'action_mode', 'none') if args else 'none'
            action_mode = action_mode.replace('rl_', '')
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            state_log_path = str(log_dir / f"{method}_{action_mode}_buffer_first10k_{timestamp}.txt")
        self._state_log_path = state_log_path
        self._state_log_steps = state_log_steps
        self._state_file = None
        self._args = args
        self._logs_saved = False

    def format_hyperparams(self) -> str:
        """Format args as comment lines for the file header."""
        if self._args is None:
            return ""
        lines = ["# Hyperparameters:\n"]
        for key, val in sorted(vars(self._args).items()):
            lines.append(f"#   {key}: {val}\n")
        lines.append("#\n")
        return "".join(lines)

    def _on_training_start(self) -> None:
        self._state_file = open(self._state_log_path, 'w')
        self._state_file.write(self.format_hyperparams())
        self._state_file.write(f"# States collected into replay buffer (first {self._state_log_steps} timesteps)\n")
        self._state_file.write("# Format: timestep\tstate\n")

    def _on_training_end(self) -> None:
        if self._state_file is not None:
            self._state_file.flush()
            self._state_file.close()
            self._state_file = None
            if not self._logs_saved:
                self.save_critic_losses()
                self.print_summary()

    def save_critic_losses(self):
        loss_log_path = self._state_log_path.replace("_buffer_", "_critic_loss_")
        with open(loss_log_path, 'w') as f:
            f.write(self.format_hyperparams())
            f.write(f"# Critic losses recorded (first {self._state_log_steps} timesteps)\n")
            f.write("# Format: timestep\tcritic_loss\n")
            for t, l in zip(self.timesteps, self.critic_losses):
                f.write(f"{t}\t{l:.6f}\n")

    def print_summary(self):
        losses = self.periodic_losses
        if not losses:
            print("\nNo critic loss data collected in first 10k steps.")
            return
        print("\n" + "=" * 60)
        print(f"DDPG CRITIC LOSS SUMMARY (first {self._state_log_steps} steps)")
        print("=" * 60)
        print(f"Total gradient updates tracked: {len(self.critic_losses)}")
        print(f"Mean:  {np.mean(losses):.4f}")
        print(f"Std:   {np.std(losses):.4f}")
        print(f"Min:   {np.min(losses):.4f}")
        print(f"Max:   {np.max(losses):.4f}")
        print(f"First: {losses[0]:.4f}")
        print(f"Last:  {losses[-1]:.4f}")
        print("=" * 60)

    def _on_step(self) -> bool:
        if self._state_file is not None and self.num_timesteps <= self._state_log_steps:
            raw_obs = self.locals.get('new_obs', None)
            if raw_obs is not None:
                obs = raw_obs[0] if hasattr(raw_obs, '__len__') else raw_obs
                state_str = '\t'.join(f"{v:.6f}" for v in np.array(obs, dtype=float).flatten())
                self._state_file.write(f"{self.num_timesteps}\t{state_str}\n")
            if self.num_timesteps == self._state_log_steps:
                self._state_file.flush()
                self.save_critic_losses()
                self.print_summary()
                self._logs_saved = True

        if self.num_timesteps <= self._state_log_steps and hasattr(self.model, 'logger') and self.model.logger is not None:
            logs = self.model.logger.name_to_value
            n_updates = logs.get('train/n_updates', None)
            critic_loss = logs.get('train/critic_loss', None)
            if critic_loss is not None and n_updates is not None and n_updates != self._last_n_updates:
                self.critic_losses.append(float(critic_loss))
                self.timesteps.append(self.num_timesteps)
                self._last_n_updates = n_updates
            if self.num_timesteps - self._last_print_timestep >= self._print_freq:
                if len(self.critic_losses) > 0:
                    loss = self.critic_losses[-1]
                    print(f"[Timestep {self.num_timesteps}] critic_loss = {loss:.4f}")
                    self.periodic_losses.append(loss)
                    self.periodic_timesteps.append(self.num_timesteps)
                self._last_print_timestep = self.num_timesteps
        return True


class Trainer(BaseTrainer):

    def __init__(self, args):
        super().__init__(args)
        self.critic_loss_cb = None

    def _extra_callbacks(self):
        """Return additional callbacks for DDPG training (critic loss tracking)."""
        critic_loss_cb = CriticLossCallback(args=self.args)
        self.critic_loss_cb = critic_loss_cb
        return [critic_loss_cb]

    def train_model(self):
        model = super().train_model()
        if self.critic_loss_cb is not None:
            self.print_critic_loss_summary()
        return model

    def get_critic_losses(self):
        """Return (timesteps, critic_losses) collected during DDPG training."""
        if self.critic_loss_cb is None:
            return [], []
        return self.critic_loss_cb.timesteps, self.critic_loss_cb.critic_losses

    def print_critic_loss_summary(self):
        """Print a summary of critic losses collected during DDPG training."""
        cb = self.critic_loss_cb
        if cb is None or len(cb.critic_losses) == 0:
            print("\nNo critic loss data was collected during training.")
            return
        losses = cb.periodic_losses
        print("\n" + "=" * 60)
        print("DDPG CRITIC LOSS SUMMARY")
        print("=" * 60)
        print(f"Total gradient updates tracked: {len(cb.critic_losses)}")
        print(f"Critic Loss (every {cb._print_freq} timesteps): {losses}")
        print(f"Mean:  {np.mean(losses):.4f}")
        print(f"Std:   {np.std(losses):.4f}")
        print(f"Min:   {np.min(losses):.4f}")
        print(f"Max:   {np.max(losses):.4f}")
        print(f"First: {losses[0]:.4f}")
        print(f"Last:  {losses[-1]:.4f}")
        print("=" * 60)
