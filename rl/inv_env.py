"""
Inventory Simulation Environment:

Dynamics match ds/simulation.py (Simulator) exactly so that
DS-env loss == RL-env loss for the same trained policy.
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import gymnasium
import torch
from pathlib import Path
from gymnasium import spaces
from gymnasium.utils import seeding


def load_global_config():
    global_path = Path(__file__).parent.parent / "config" / "global.yaml"
    with open(global_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    g = cfg.get('global', {})
    return (g.get('bp', 2), g.get('max_lead_time', 1),
            g.get('lost_sales_demand', False), g.get('max_replenish', 100))


_GLOBAL_BP, _GLOBAL_LEAD_TIME, _GLOBAL_LOST_SALES, _GLOBAL_MAX_REPLENISH = load_global_config()

# FEATS = ["feat1", "feat2", "feat3", "feat4"] # for COEFF
FEATS = ["feat1"]
REG_COLS = ["reg_input1", "reg_input2", "reg_input3", "reg_input4"]


def parse_ds_csv(data_path: str):
    """
    Returns
    -------
    groups : dict[int, pd.DataFrame]
        sku_id -> DataFrame with columns
        [day, init_stock, demand, feat1..feat4, reg_input1..reg_input4]
    sku_meta : dict[int, dict]
        sku_id -> {'lt_seq': list[int], 'bp': int, 'period_num': int}
    """
    df = pd.read_csv(data_path)
    # accept both old ('date_num') and new ('period_num') column names
    if 'period_num' not in df.columns and 'date_num' in df.columns:
        df.rename(columns={'date_num': 'period_num'}, inplace=True)
    groups = {}
    sku_meta = {}

    for _, row in df.iterrows():
        sid = int(row['sku_id'])
        period_num = int(row['period_num'])
        bp = int(row['bp'])

        demands = list(map(float, str(row['demand']).split(',')))
        lt_seq = list(map(int, map(float, str(row['lt']).split(','))))
        stocks = list(map(float, str(row['stock']).split(',')))
        init_stock = stocks[0] if stocks else 0.0

        # features: "f1_t0,f1_t1,...;f2_t0,f2_t1,...;..."
        feat_seqs = [list(map(float, f.split(',')))
                     for f in str(row['feature']).split(';')]
        reg_seqs = [list(map(float, r.split(',')))
                    for r in str(row['reg_input']).split(';')]

        T = len(demands)
        rows = []
        for t in range(T):
            r = {
                'day': t,
                'init_stock': init_stock,
                'demand': demands[t],
            }
            for fi, fname in enumerate(FEATS):
                r[fname] = feat_seqs[fi][t] if fi < len(feat_seqs) and t < len(feat_seqs[fi]) else 0.0
            for ri, rname in enumerate(REG_COLS):
                r[rname] = reg_seqs[ri][t] if ri < len(reg_seqs) and t < len(reg_seqs[ri]) else 0.0
            rows.append(r)

        groups[sid] = pd.DataFrame(rows)
        sku_meta[sid] = {
            'lt_seq': lt_seq[:T],
            'bp': bp,
            'period_num': period_num,
        }

    return groups, sku_meta


class InventorySimEnv(gymnasium.Env):
    """
    Inventory dynamics replicate ds/simulation.py exactly (Supports bp >= or < lead_time):
      1. Place order into pipeline[lt_t]
      2. Fulfil demand:  post_inv = pipeline[0] - demand
      3. Record inv_level_end[t] = post_inv
      4. Shift pipeline:  pipeline[0] = post_inv + pipeline[1]; shift left
    """
    metadata = {"render_modes": []}

    def __init__(
            self,
            sku_id: int | None = None,
            args: argparse.Namespace = None,
            sku_whitelist: list = None,
            action_mode: str = "coeff",
            train: bool = True,
            deterministic_sku_cycle: bool = False,
            data_path: str = None,
    ):
        super().__init__()
        self.action_mode = action_mode
        self.train = train

        self.bp_default = getattr(args, 'bp', _GLOBAL_BP)
        self.max_lt = getattr(args, 'max_lead_time', _GLOBAL_LEAD_TIME)
        self.lost_sales = getattr(args, 'lost_sales_demand', _GLOBAL_LOST_SALES)
        self.max_replenish = getattr(args, 'max_replenish', _GLOBAL_MAX_REPLENISH)
        self.args = args
        self.np_random, _ = seeding.np_random(args.seed)
        np.random.seed(args.seed)

        self.norm_min = self.norm_max = None

        self._sku_meta = {}  # sku_id -> {lt_seq, bp, period_num}
        if data_path is not None:
            self.groups, self._sku_meta = parse_ds_csv(data_path)
        else:
            raise ValueError("data_path must be provided")

        if sku_whitelist is not None and data_path is not None:
            self.groups = {k: v for k, v in self.groups.items() if k in sku_whitelist}

        self.all_skus = list(self.groups.keys())
        if sku_id is not None and sku_id not in self.groups:
            raise ValueError(f"sku_id {sku_id} not found")
        self.fixed_sku = sku_id

        # Deterministic SKU cycling
        self.deterministic_sku_cycle = deterministic_sku_cycle
        self.sku_cycle_index = 0

        # state and action spaces
        obs_dim = len(FEATS) + (1 + self.max_lt)   # dim of features + pipeline length
        obs_low = np.full((1, obs_dim), -np.inf, dtype=np.float32)
        obs_high = np.full((1, obs_dim), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        if self.action_mode in ("rl_none_coeff", "rl_base_coeff"):
            self.action_space = spaces.Box(0, 1, shape=(1, 5), dtype=np.float32)
        elif self.action_mode in ("rl_none", "rl_base"):
            self.action_space = spaces.Box(0, self.max_replenish, shape=(1, 1), dtype=np.float32)

        # for each sku
        self.sku_df: pd.DataFrame | None = None
        self.pipeline: np.ndarray | None = None
        self.inv_level_end: list[float] = []
        self.lost_sales_seq: list[float] = []
        self.t: int = 0


    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        if self.fixed_sku is not None:
            sku = self.fixed_sku
        elif self.deterministic_sku_cycle:
            sku = self.all_skus[self.sku_cycle_index % len(self.all_skus)]
            self.sku_cycle_index += 1
        else:
            sku = self.np_random.choice(self.all_skus)
        self.sku_df = self.groups[sku].copy()

        if sku in self._sku_meta:
            meta = self._sku_meta[sku]
            self.lt_seq = meta['lt_seq']
            self.bp = meta['bp']
        else:
            self.lt_seq = [self.max_lt] * len(self.sku_df)
            self.bp = self.bp_default

        self.ignore_period_num = max(self.lt_seq[:len(self.sku_df)]) + self.bp

        init_stock = float(self.sku_df.at[0, "init_stock"])
        self.pipeline = np.zeros(1 + self.max_lt, dtype=np.float64)
        self.pipeline[0] = init_stock

        self.inv_level_end = []
        self.lost_sales_seq = []
        self.order_qty_seq = []
        self.inv_pos_seq = []
        self.t = 0
        self._step_overages = []
        self._step_underages = []

        tmp = self.sku_df[["demand"] + FEATS]
        self.norm_min = tmp.min()
        self.norm_max = tmp.max()

        if self.action_mode in ("rl_none_coeff", "rl_base_coeff",
                                "ds_none_coeff", "ds_base_coeff"):
            self.normalize = True
        else:
            self.normalize = False

        # Warmup: days 0..bp-1 with order=0 (matches DS: t < bp masked)
        self.inv_pos_seq.append(float(self.pipeline.sum()))
        self.order_qty_seq.append(0.0)
        for d in range(min(self.bp, len(self.sku_df))):
            self.simulate_one_day(order_qty=0.0)

        obs = self.build_obs(normalize=self.normalize)
        return obs, {}

    def step(self, action: np.ndarray):
        done = False
        if action.ndim == 3:
            action = action.squeeze(1)

        t_before = self.t
        lt_t = self.lt_seq[min(t_before, len(self.lt_seq) - 1)]

        qty = self.decode_action(action)
        self.inv_pos_seq.append(float(self.pipeline.sum()))
        self.order_qty_seq.append(qty)

        # --- Simulate bp days (real, permanent) ---
        remaining = len(self.sku_df) - self.t
        days_this_step = min(self.bp, remaining)
        for d in range(days_this_step):
            self.simulate_one_day(order_qty=qty if d == 0 else 0.0)

        # --- Lookahead lt days with order=0 to fill loss window ---
        # Correct because the next order arrives at t+bp+lt (window end),
        # so it does NOT affect inventory during [t+lt, t+bp+lt).
        saved_pipeline = self.pipeline.copy()
        saved_t = self.t
        saved_inv_len = len(self.inv_level_end)
        saved_lost_len = len(self.lost_sales_seq)

        lookahead = min(lt_t, len(self.sku_df) - self.t)
        for d in range(lookahead):
            self.simulate_one_day(order_qty=0.0)

        # Loss = [t_before + lt, t_before + bp + lt), always bp days
        loss_start = t_before + lt_t
        loss_end = min(t_before + self.bp + lt_t, len(self.inv_level_end))
        reward, underage, overage = self.calc_loss(loss_start, loss_end)

        # --- Rewind lookahead (restore state to after bp days) ---
        self.pipeline = saved_pipeline
        self.t = saved_t
        self.inv_level_end = self.inv_level_end[:saved_inv_len]
        self.lost_sales_seq = self.lost_sales_seq[:saved_lost_len]

        # --- Early termination: next order won't arrive before horizon ---
        if self.t >= len(self.sku_df):
            done = True
        else:
            next_lt = self.lt_seq[min(self.t, len(self.lt_seq) - 1)]
            if self.t + next_lt >= len(self.sku_df):
                # Simulate remaining days for complete episode metrics
                for d in range(len(self.sku_df) - self.t):
                    self.simulate_one_day(order_qty=0.0)
                done = True

        if done and not self.train:
            print('base vec:', np.array([round(a + b, 3) for a, b in
                                         zip(self.order_qty_seq, self.inv_pos_seq)]))

        obs = self.build_obs(self.normalize)

        return obs, reward, done, False, {
            'underage': underage,
            'overage': overage,
        }

    def simulate_one_day(self, order_qty: float):
        """Advance one day.  Replicates Simulator.transition() logic."""
        # Place order into pipeline slot [lt_t] (per-period lead time from data)
        lt_t = self.lt_seq[min(self.t, len(self.lt_seq) - 1)]
        self.pipeline[lt_t] += order_qty

        # Fulfill demand
        demand = float(self.sku_df.at[self.t, "demand"])
        post_inv = self.pipeline[0] - demand

        if self.lost_sales and post_inv < 0:
            self.lost_sales_seq.append(-post_inv)
            post_inv = 0.0
        else:
            self.lost_sales_seq.append(0.0)

        # Record ending inventory
        self.inv_level_end.append(post_inv)

        # Shift pipeline (arrival for next day)
        if len(self.pipeline) > 1:
            self.pipeline[0] = post_inv + self.pipeline[1]
            self.pipeline[1:-1] = self.pipeline[2:]
            self.pipeline[-1] = 0.0
        else:
            self.pipeline[0] = post_inv

        self.t += 1


    def decode_action(self, action):
        """Translate raw network output into an order quantity."""
        if self.action_mode in ('ds_none', 'rl_none'):
            return float(np.clip(float(action[0]), 0, self.max_replenish))

        elif self.action_mode in ('ds_none_coeff', 'rl_none_coeff'):
            feats = self.sku_df.loc[self.t, REG_COLS].to_numpy(np.float32)
            coeffs = np.clip(action[:, :4], 0, 1)
            bias = action[:, 4]
            tmp_qty = coeffs @ feats * 28 + bias
            return float(np.clip(tmp_qty[0], 0, self.max_replenish))

        elif self.action_mode in ('ds_base_coeff', 'rl_base_coeff'):
            feats = self.sku_df.loc[self.t, REG_COLS].to_numpy(np.float32)
            coeffs = np.clip(action[:, :4], 0, 1)
            bias = action[:, 4]
            target_inv = coeffs @ feats * 28 + bias
            inv_pos = float(self.pipeline.sum())
            return float(np.clip(target_inv[0] - inv_pos, 0, self.max_replenish))

        elif self.action_mode in ('ds_base', 'rl_base'):
            target_inv = float(action[0])
            inv_pos = float(self.pipeline.sum())
            return float(np.clip(target_inv - inv_pos, 0, self.max_replenish))

        raise ValueError(f"Unknown action_mode: {self.action_mode}")

    def build_obs(self, normalize: bool) -> np.ndarray:
        """
        None / Base:
            flatten([features_org[:, 0:1], inventory])
          = [feat1_raw, pipeline[0], pipeline[1], ..., pipeline[lt]]

        NoneCoeff / BaseCoeff:
            flatten([features_normalized, inventory / max_replenish])
          = [feat1_norm, ..., feat4_norm, pipe[0]/M, ..., pipe[lt]/M]
        """
        obs_day = min(self.t, len(self.sku_df) - 1)

        if normalize:
            feat_vals = self.sku_df.loc[obs_day, FEATS].to_numpy(np.float32)
            for i, c in enumerate(FEATS):
                feat_vals[i] = (feat_vals[i] - self.norm_min[c]) / (self.norm_max[c] - self.norm_min[c] + 1e-8)
            pipe_norm = self.pipeline.astype(np.float32) / self.max_replenish
            return np.concatenate([feat_vals, pipe_norm])
        else:
            feat1 = float(self.sku_df.at[obs_day, "feat1"])
            return np.concatenate([[feat1], self.pipeline.astype(np.float32)])


    def roll_with_agent(self, agent, deterministic: bool = True):
        """Run a full episode (RL interface). Returns metrics tuple."""
        obs, _ = self.reset()
        done = False
        while not done:
            if hasattr(agent, "predict"):
                if obs.ndim == 1:
                    obs = obs.reshape(1, 1, -1)
                act, _ = agent.predict(obs, deterministic=deterministic)
            else:
                act, _ = agent(obs, None)
            obs, _, done, _, _ = self.step(act)
        return self.compute_episode_metrics()

    def run_full_episode(self, policy, deterministic: bool = True):
        """Run a full episode (DS interface). Returns metrics tuple."""
        obs, _ = self.reset()
        done = False
        while not done:
            if hasattr(policy, "predict"):
                action, _ = policy.predict(obs, deterministic=deterministic)
            else:
                action, _ = policy(obs, None)
            obs, _, done, _, _ = self.step(action)
        return self.compute_episode_metrics()

    def compute_episode_metrics(self):
        """Compute episode-level/horizon metrics from inv_level_end."""
        skip = self.ignore_period_num
        inv_tr = np.asarray(self.inv_level_end[skip:])
        demand_tr = self.sku_df["demand"].to_numpy()[skip:len(self.inv_level_end)]
        lost_tr = np.asarray(self.lost_sales_seq[skip:])
        sales_tr = demand_tr - lost_tr

        length = len(inv_tr)

        # Flat average over all valid days
        overage = np.maximum(inv_tr, 0).sum() / length if length > 0 else 0.0
        if self.lost_sales:
            underage = lost_tr.sum() / length if length > 0 else 0.0
        else:
            underage = np.maximum(-inv_tr, 0).sum() / length if length > 0 else 0.0

        service = 1.0 - np.mean(inv_tr <= 0) if length > 0 else 0.0
        avg_inv = float(inv_tr.mean()) if length > 0 else 0.0
        avg_sales = float(sales_tr.mean()) if length > 0 else 0.0

        weight_ratio = self.args.loss_alpha
        aligned_reward = -(weight_ratio * overage + (1 - weight_ratio) * underage)

        return (float(service), avg_inv, avg_sales,
                float(overage), float(underage),
                float(aligned_reward), inv_tr)

    def calc_loss(self, window_start=None, window_end=None):
        """Per-step reward over an explicit inventory window.

        Parameters
        ----------
        window_start : int, optional
            First index (inclusive) into inv_level_end.
            Defaults to ignore_period_num.
        window_end : int, optional
            Last index (exclusive) into inv_level_end.
            Defaults to len(inv_level_end).
        """
        skip = self.ignore_period_num
        if window_start is None:
            window_start = skip
        if window_end is None:
            window_end = len(self.inv_level_end)

        start = max(skip, window_start)
        end = min(window_end, len(self.inv_level_end))

        if start >= end:
            return 0.0, 0.0, 0.0

        inv_arr = np.asarray(self.inv_level_end[start:end])
        window_len = end - start
        overage = np.maximum(inv_arr, 0).sum() / window_len

        if self.lost_sales:
            underage = np.asarray(self.lost_sales_seq[start:end]).sum() / window_len
        else:
            underage = np.maximum(-inv_arr, 0).sum() / window_len

        reward = -(self.args.loss_alpha * overage + (1 - self.args.loss_alpha) * underage)
        return reward, underage, overage


    def seed(self, seed: int | None = None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass

    @staticmethod
    def predict_wrapper(torch_model, env, device="cpu"):
        """Wrap a PyTorch DS model for use with the env."""
        torch_model.eval()

        class _Wrapper:
            def __init__(self, model):
                self.model = model.to(device)

            def predict(self, obs, deterministic=True):
                obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
                if obs_t.ndim == 1:
                    obs_t = obs_t.unsqueeze(0)  # (D,) -> (1, D)

                with torch.no_grad():
                    out = self.model.net(obs_t)
                return out.cpu().numpy(), None

        return _Wrapper(torch_model)
