#!/usr/bin/env python3

DATA_DIR = None

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import yaml
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from collections import defaultdict
from pathlib import Path


def load_global_config():
    global_path = Path(__file__).parent / "config" / "global.yaml"
    with open(global_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    g = cfg.get('global', {})
    return g

def parse_syn_full(csv_path: str) -> Tuple[Dict[int, dict], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    sku_data = {}

    for _, row in df.iterrows():
        sid = int(row['sku_id'])
        period_num = int(row['period_num'])
        bp = int(row['bp'])
        demands = list(map(float, str(row['demand']).split(',')))
        lt_seq = list(map(int, map(float, str(row['lt']).split(','))))
        stocks = list(map(float, str(row['stock']).split(',')))
        init_stock = stocks[0] if stocks else 0.0

        sku_data[sid] = {
            'demands': demands,
            'lt_seq': lt_seq[:len(demands)],
            'bp': bp,
            'period_num': period_num,
            'init_stock': init_stock,
        }
    return sku_data, df

def load_sku_splits(data_dir: str) -> Tuple[List[int], List[int], List[int]]:
    splits = {}
    for name in ('train', 'val', 'test'):
        path = os.path.join(data_dir, f'{name}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                splits[name] = json.load(f)
        else:
            splits[name] = []
    return splits['train'], splits['val'], splits['test']


def compute_empirical_demand_distribution(
    sku_data: Dict[int, dict],
    train_skus: List[int],
    T: int,
) -> List[Dict[int, float]]:
    demand_distributions = []

    for t in range(T):
        day_demands = []
        for sid in train_skus:
            if sid in sku_data and t < len(sku_data[sid]['demands']):
                day_demands.append(sku_data[sid]['demands'][t])

        if day_demands:
            demand_counts = defaultdict(int)
            for d in day_demands:
                demand_counts[int(round(d))] += 1
            total = len(day_demands)
            demand_distributions.append(
                {d: cnt / total for d, cnt in demand_counts.items()}
            )
        else:
            demand_distributions.append({0: 1.0})

    return demand_distributions


class InventoryDP:
    def __init__(
        self,
        demand_distributions: List[Dict[int, float]],
        bp: int,
        max_lt: int,
        h: float,
        p: float,
        lost_sales: bool,
        I_MIN: int = -100,
        I_MAX: int = 200,
        A_MAX: int = 100,
    ):
        self.demands = demand_distributions
        self.T = len(demand_distributions)
        self.bp = bp
        self.max_lt = max_lt
        self.h = h
        self.p = p
        self.lost_sales = lost_sales
        self.I_MIN = I_MIN
        self.I_MAX = I_MAX
        self.A_MAX = A_MAX
        self.skip = max_lt + bp

        self.review_periods = set(range(bp, self.T, bp))

        n_states = I_MAX - I_MIN + 1
        self.V = [np.full(n_states, 0.0) for _ in range(self.T + 1)]
        self.Pi = [np.zeros(n_states, dtype=int) for _ in range(self.T)]

    def idx(self, i: int) -> int:
        return i - self.I_MIN

    def expected_period_cost(self, t: int, i: int) -> float:
        if t < self.skip:
            return 0.0
        dpmf = self.demands[t]
        cost = 0.0
        for d, prob in dpmf.items():
            end_inv = i - d
            if self.lost_sales and end_inv < 0:
                cost += prob * self.p * (-end_inv)
            elif end_inv >= 0:
                cost += prob * self.h * end_inv
            else:
                cost += prob * self.p * (-end_inv)
        return cost

    def next_state_distribution(self, t: int, i: int, u: int) -> Dict[int, float]:
        dpmf = self.demands[t]
        ns = defaultdict(float)
        for d, prob in dpmf.items():
            ip1 = i - d + u
            if self.lost_sales and (i - d) < 0:
                ip1 = 0 + u
            ip1 = max(self.I_MIN, min(self.I_MAX, ip1))
            ns[ip1] += prob
        return dict(ns)

    def solve(self) -> float:
        self.V[self.T][:] = 0.0

        for t in range(self.T - 1, -1, -1):
            n_states = self.I_MAX - self.I_MIN + 1
            Vt = np.full(n_states, np.inf)
            Pit = np.zeros(n_states, dtype=int)
            order_allowed = t in self.review_periods

            for i in range(self.I_MIN, self.I_MAX + 1):
                base_cost = self.expected_period_cost(t, i)

                if not order_allowed:
                    ns = self.next_state_distribution(t, i, 0)
                    cont = sum(prob * self.V[t + 1][self.idx(ip1)]
                               for ip1, prob in ns.items())
                    Vt[self.idx(i)] = base_cost + cont
                    Pit[self.idx(i)] = 0
                else:
                    best_val = np.inf
                    best_u = 0
                    for u in range(0, self.A_MAX + 1):
                        ns = self.next_state_distribution(t, i, u)
                        cont = sum(prob * self.V[t + 1][self.idx(ip1)]
                                   for ip1, prob in ns.items())
                        total = base_cost + cont
                        if total < best_val:
                            best_val = total
                            best_u = u
                    Vt[self.idx(i)] = best_val
                    Pit[self.idx(i)] = best_u

            self.V[t] = Vt
            self.Pi[t] = Pit

        return self.V[0][self.idx(0)]


def simulate_sku(
    sku_info: dict,
    policy: list,
    dp: InventoryDP,
) -> Tuple[float, float, float]:
    demands = sku_info['demands']
    T = len(demands)
    bp = sku_info['bp']
    lt_seq = sku_info['lt_seq']
    max_lt = max(lt_seq) if lt_seq else 1
    skip = max_lt + bp

    pipeline = np.zeros(1 + max_lt, dtype=np.float64)
    pipeline[0] = sku_info['init_stock']

    inv_level_end = []
    lost_sales_log = []

    for t in range(T):
        if t % bp == 0 and t >= bp and t < len(policy):
            inv = pipeline[0]
            clamped_inv = max(dp.I_MIN, min(dp.I_MAX, int(round(inv))))
            order = int(policy[t][dp.idx(clamped_inv)])
        else:
            order = 0

        lt_t = lt_seq[min(t, len(lt_seq) - 1)]
        pipeline[lt_t] += order

        demand = demands[t]
        post_inv = pipeline[0] - demand

        if dp.lost_sales and post_inv < 0:
            lost_sales_log.append(-post_inv)
            post_inv = 0.0
        else:
            lost_sales_log.append(0.0)

        inv_level_end.append(post_inv)

        if len(pipeline) > 1:
            pipeline[0] = post_inv + pipeline[1]
            pipeline[1:-1] = pipeline[2:]
            pipeline[-1] = 0.0
        else:
            pipeline[0] = post_inv

    inv_tr = np.asarray(inv_level_end[skip:])
    demand_tr = np.asarray(demands[skip:len(inv_level_end)])
    lost_tr = np.asarray(lost_sales_log[skip:])

    length = len(inv_tr)
    if length == 0:
        return 0.0, 0.0, 0.0

    overage = np.maximum(inv_tr, 0).sum() / length
    if dp.lost_sales:
        underage = lost_tr.sum() / length
    else:
        underage = np.maximum(-inv_tr, 0).sum() / length

    return overage, underage, length


def evaluate_policy(
    sku_data: Dict[int, dict],
    sku_list: List[int],
    policy: list,
    dp: InventoryDP,
    loss_alpha: float,
) -> Dict[str, float]:
    overages, underages, losses = [], [], []

    for sid in sku_list:
        if sid not in sku_data:
            continue
        overage, underage, length = simulate_sku(sku_data[sid], policy, dp)
        if length == 0:
            continue
        overages.append(overage)
        underages.append(underage)
        loss = loss_alpha * overage + (1 - loss_alpha) * underage
        losses.append(loss)

    if not losses:
        return {'n': 0, 'overage': 0, 'underage': 0, 'loss': 0}

    return {
        'n': len(losses),
        'overage': np.mean(overages),
        'underage': np.mean(underages),
        'loss': np.mean(losses),
    }


def main():
    parser = argparse.ArgumentParser(description='DP Inventory Control')
    parser.add_argument('--data-dir', type=str, default=None)
    args = parser.parse_args()

    gcfg = load_global_config()
    loss_alpha = gcfg.get('loss_alpha', 0.1)
    lost_sales = gcfg.get('lost_sales_demand', False)
    max_lt = gcfg.get('max_lead_time', 1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or DATA_DIR or gcfg.get('data_directory')
    if data_dir is None:
        print("Error: No data directory specified.")
        sys.exit(1)
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(base_dir, data_dir)

    csv_path = os.path.join(data_dir, 'syn_full.csv')
    print(f"Data directory: {data_dir}")
    print(f"CSV path: {csv_path}")

    print("Loading data...")
    sku_data, df = parse_syn_full(csv_path)
    train_skus, val_skus, test_skus = load_sku_splits(data_dir)
    print(f"Loaded {len(sku_data)} SKUs: {len(train_skus)} train, "
          f"{len(val_skus)} val, {len(test_skus)} test")

    sample_sid = train_skus[0] if train_skus else list(sku_data.keys())[0]
    sample = sku_data[sample_sid]
    T = len(sample['demands'])
    bp = sample['bp']
    print(f"Periods T={T}, review period bp={bp}, max lead time={max_lt}, "
          f"lost_sales={lost_sales}")

    print("Computing empirical demand distributions from training data...")
    demand_dists = compute_empirical_demand_distribution(sku_data, train_skus, T)

    if len(demand_dists) > max_lt:
        sample_t = max_lt
        dist = demand_dists[sample_t]
        print(f"\nSample demand distribution for day {sample_t}:")
        for d, prob in sorted(dist.items())[:10]:
            print(f"  Demand {d}: probability {prob:.3f}")

    print("\nSolving dynamic program...")
    dp = InventoryDP(
        demand_distributions=demand_dists,
        bp=bp,
        max_lt=max_lt,
        h=loss_alpha,
        p=1.0 - loss_alpha,
        lost_sales=lost_sales,
    )
    optimal_value = dp.solve()
    cost_periods = T - dp.skip
    print(f"Optimal expected cost from DP: {optimal_value:.4f} "
          f"(avg per period: {optimal_value / cost_periods:.4f})")

    print("\nOptimal order quantities at review periods (inventory -> order):")
    review_days = sorted(dp.review_periods)[:10]
    print("Day | " + " ".join([f"i={i:+d}" for i in range(-10, 21, 5)]))
    print("-" * 60)
    for day in review_days:
        if day < len(dp.Pi):
            row = [f"{day:3d} |"]
            for i in range(-10, 21, 5):
                if dp.I_MIN <= i <= dp.I_MAX:
                    order = dp.Pi[day][dp.idx(i)]
                    row.append(f"{order:5d}")
            print(" ".join(row))

    for name, sku_list in [('Train', train_skus), ('Val', val_skus), ('Test', test_skus)]:
        if not sku_list:
            continue
        print(f"\nEvaluating on {name} SKUs...")
        results = evaluate_policy(sku_data, sku_list, dp.Pi, dp, loss_alpha)
        print(f"  SKUs evaluated: {results['n']}")
        print(f"  Avg overage: {results['overage']:.4f}")
        print(f"  Avg underage: {results['underage']:.4f}")
        print(f"  Avg loss: {results['loss']:.4f}")


if __name__ == "__main__":
    main()
