"""
Generate IID normal-demand synthetic data AND convert to sequence format.
Produces a complete data4dl_* directory with:
  - syn_full.csv (one row per SKU, sequence columns)
  - train.json / val.json / test.json (SKU ID lists)
"""

import pandas as pd
import numpy as np
import random
import json
import os
import sys
from pathlib import Path

# ========================= Configuration =========================
NO_FEATURES = False
PERFECT_FEATURES = True

BP = 2
LEAD_TIME = 1
RANDOM_DEMAND = True

HORIZON = 2 # 2, 8, 16, 32, 64
TARGETS_DAYS = (BP + LEAD_TIME) + BP * HORIZON

SEED = 42
np.random.seed(SEED)

NUM_SIGMA = 5  # training sample size = val sample size
TEST_REPEAT = 100  # testing sample size = TEST_REPEAT * NUM_SIGMA; each sigma has TEST_REPEAT testing samples

MEAN = 100
STDS_1 = np.random.uniform(5, 20, size=NUM_SIGMA)
STDS_2 = STDS_1
STDS_3 = np.repeat(STDS_1, TEST_REPEAT)
STDS = np.concatenate([STDS_1, STDS_2, STDS_3])

NUM_SKUS = len(STDS)
print('#std', NUM_SIGMA, '; total # SKU', NUM_SKUS, '\nSTDS', STDS[:NUM_SIGMA])

OUTPUT_DIR_NAME = f"data_ds_iid_perffeat_T{TARGETS_DAYS}_train{NUM_SIGMA}"

# ========================= Demand generation =========================

def sample_random_D_true(mean, std: float, rng: np.random.Generator) -> float:
    demand = rng.normal(loc=mean, scale=std)
    return max(0, demand)


def get_demands_for_sku(sku_idx, sku_stds, mean, total_days, rng, random_flag=True):
    sku_std = sku_stds[sku_idx]
    demands = []
    for t in range(total_days):
        if t >= 2 and t % 2 == 1:
            if random_flag:
                demands.append(sample_random_D_true(mean, sku_std, rng))
            else:
                demands.append(0.0)
        else:
            demands.append(0.0)
    return demands


def calculate_inventory_loss(base_stock_levels, all_data, num_skus, total_days):
    holding_cost = 0.1
    backlog_cost = 0.9
    losses = []

    for sku_idx in range(num_skus):
        base_stock = base_stock_levels[sku_idx]
        inventory = base_stock
        total_cost = 0.0
        cost_periods = 0
        for t in range(3, total_days):
            if t >= 2 and t % 2 == 1:
                demand = all_data[sku_idx].loc[all_data[sku_idx]['day'] == t, 'demand'].values[0]
                inventory -= demand
                if inventory >= 0:
                    period_cost = holding_cost * inventory
                else:
                    period_cost = backlog_cost * (-inventory)
                total_cost += period_cost
                cost_periods += 1
                inventory = max(inventory, base_stock)
        avg_cost = total_cost / cost_periods if cost_periods > 0 else 0.0
        losses.append(avg_cost)

    return losses


# ========================= Sequence conversion =========================

def make_seq(arr):
    return ','.join(map(str, arr.tolist()))


def seq_convert(panel_df, sku_list, out_csv, T, lead_time):
    rows = []
    sku_to_group = dict(tuple(panel_df.groupby("sku_id")))
    for sku in sku_list:
        if sku not in sku_to_group:
            continue
        df = sku_to_group[sku].sort_values("day")

        lt_seq = np.full(T, lead_time, dtype=int)
        demand_seq = np.concatenate([df["demand"].to_numpy(), np.zeros(T - len(df))])
        stock_seq = np.concatenate([df["init_stock"].to_numpy(), np.zeros(T - len(df))])
        feat1_seq = np.concatenate([df["feat1"].to_numpy(dtype=float), np.zeros(T - len(df))])
        feat2_seq = np.concatenate([df["feat2"].to_numpy(dtype=float), np.zeros(T - len(df))])
        feat3_seq = np.concatenate([df["feat3"].to_numpy(dtype=float), np.zeros(T - len(df))])
        feat4_seq = np.concatenate([df["feat4"].to_numpy(dtype=float), np.zeros(T - len(df))])

        reg1_seq = np.concatenate([df["reg_input1"].to_numpy(dtype=float), np.zeros(T - len(df))])
        reg2_seq = np.concatenate([df["reg_input2"].to_numpy(dtype=float), np.zeros(T - len(df))])
        reg3_seq = np.concatenate([df["reg_input3"].to_numpy(dtype=float), np.zeros(T - len(df))])
        reg4_seq = np.concatenate([df["reg_input4"].to_numpy(dtype=float), np.zeros(T - len(df))])

        rows.append({
            "sku_id": sku,
            "period_num": T,
            "bp": BP,
            "lt": make_seq(lt_seq),
            "demand": make_seq(demand_seq),
            "stock": make_seq(stock_seq),
            "feature": make_seq(feat1_seq) + ";" + make_seq(feat2_seq) + ";" + make_seq(feat3_seq) + ";" + make_seq(feat4_seq),
            "reg_input": make_seq(reg1_seq) + ";" + make_seq(reg2_seq) + ";" + make_seq(reg3_seq) + ";" + make_seq(reg4_seq)
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  saved {out_csv} ({len(rows)} skus)")


# ========================= Main =========================

def main():
    np.random.seed(SEED)
    random.seed(SEED)
    rng = np.random.default_rng(SEED)

    num_skus = NUM_SKUS
    target_days = TARGETS_DAYS
    warmup_days = 0
    total_days = target_days + warmup_days

    sku_stds = STDS
    mean = MEAN

    # Critical fractile analysis
    critical_fractiles_90 = []
    for i, std in enumerate(sku_stds):
        fractile_90 = mean + 1.282 * std
        critical_fractiles_90.append(fractile_90)

    all_data = []
    empirical_critical_fractiles_90 = []

    for sku_idx in range(num_skus):
        demands = get_demands_for_sku(sku_idx, sku_stds, mean, total_days, rng, True)

        empirical_demands = [demands[t] for t in range(total_days) if t >= 2 and t % 2 == 1]
        empirical_fractile_90 = np.percentile(empirical_demands, 90) if empirical_demands else 0.0
        empirical_critical_fractiles_90.append(empirical_fractile_90)

        df = pd.DataFrame({
            "sku_id": sku_idx + 1,
            "day": list(range(total_days)),
            "demand": demands
        })
        all_data.append(df)

    # Empirical critical fractiles from training data
    empirical_critical_fractiles_train = []
    for i in range(NUM_SIGMA):
        combined_demands = []
        for src_idx in [i, NUM_SIGMA + i]:
            for t in range(total_days):
                if t >= 2 and t % 2 == 1:
                    combined_demands.append(all_data[src_idx].loc[all_data[src_idx]['day'] == t, 'demand'].values[0])
        empirical_fractile_90 = np.percentile(combined_demands, 90) if combined_demands else 0.0
        empirical_critical_fractiles_train.append(empirical_fractile_90)

    # Test base-stock levels from empirical fractiles
    empirical_base_stocks_test = [empirical_critical_fractiles_train[i // TEST_REPEAT] for i in range(len(STDS_3))]

    stds3_start_idx = 2 * NUM_SIGMA
    all_data_stds3 = all_data[stds3_start_idx:]
    empirical_losses = calculate_inventory_loss(empirical_base_stocks_test, all_data_stds3, len(STDS_3), total_days)

    theoretical_critical_fractiles_test = [mean + 1.282 * std for std in STDS_3]
    theoretical_losses = calculate_inventory_loss(theoretical_critical_fractiles_test, all_data_stds3, len(STDS_3), total_days)

    # Concatenate all SKUs
    final_df = pd.concat(all_data, ignore_index=True)

    # Print analysis
    print(f"Empirical critical fractiles (from STDS_1 & STDS_2): {[round(float(x), 5) for x in empirical_critical_fractiles_train]}")
    print(f"Empirical losses on STDS_3: {[round(float(x), 5) for x in empirical_losses]}\n")
    print(f"Theoretical losses on STDS_3: {[round(float(x), 5) for x in theoretical_losses]}\n")
    print(f"Mean empirical loss on STDS_3: {np.mean(empirical_losses):.5f}")
    print(f"Mean theoretical loss on STDS_3: {np.mean(theoretical_losses):.5f}\n")

    print("Mean losses by original std pattern:")
    print("Std\t\tEmpirical Loss\tTheoretical Loss")
    print("-" * 50)
    for i in range(NUM_SIGMA):
        start_idx = i * TEST_REPEAT
        end_idx = (i + 1) * TEST_REPEAT
        emp_mean = np.mean(empirical_losses[start_idx:end_idx])
        theo_mean = np.mean(theoretical_losses[start_idx:end_idx])
        print(f"{STDS_1[i]:.3f}\t\t{emp_mean:.5f}\t\t{theo_mean:.5f}")

    # Feature columns
    if NO_FEATURES:
        final_df["feat1"] = 0.0
        final_df["feat2"] = 0.0
        final_df["feat3"] = 0.0
        final_df["feat4"] = 0.0
    elif PERFECT_FEATURES:
        sku_mean_map = {i + 1: sku_stds[i] for i in range(num_skus)}
        final_df["feat1"] = final_df["sku_id"].map(sku_mean_map)
        final_df["feat2"] = 0.0
        final_df["feat3"] = 0.0
        final_df["feat4"] = 0.0
    else:
        final_df["feat1"] = final_df.groupby("sku_id")["demand"].transform(
            lambda x: x.rolling(window=28, min_periods=1).mean().shift(1))
        final_df["feat2"] = final_df.groupby("sku_id")["demand"].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean().shift(1))
        final_df["feat3"] = final_df.groupby("sku_id")["demand"].transform(
            lambda x: x.shift(-1).rolling(window=7, min_periods=1).mean() + np.random.normal(0, 3, size=len(x)))
        final_df["feat4"] = final_df.groupby("sku_id")["demand"].transform(
            lambda x: x.shift(-1).rolling(window=28, min_periods=1).mean() + np.random.normal(0, 3, size=len(x)))

    # Coeff inputs
    final_df["reg_input1"] = final_df.groupby("sku_id")["demand"].transform(
        lambda x: x.rolling(window=28, min_periods=1).mean().shift(1))
    final_df["reg_input2"] = final_df.groupby("sku_id")["demand"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean().shift(1))
    final_df["reg_input3"] = final_df.groupby("sku_id")["demand"].transform(
        lambda x: x.shift(-1).rolling(window=7, min_periods=1).mean() + np.random.normal(0, 3, size=len(x)))
    final_df["reg_input4"] = final_df.groupby("sku_id")["demand"].transform(
        lambda x: x.shift(-1).rolling(window=28, min_periods=1).mean() + np.random.normal(0, 3, size=len(x)))

    # Drop warmup days
    if warmup_days > 0:
        final_df = final_df.groupby("sku_id").apply(lambda x: x.iloc[warmup_days:]).reset_index(drop=True)

    # init_stock
    final_df["init_stock"] = 0

    final_df = final_df[[
        "sku_id", "day", "demand", "init_stock",
        "feat1", "feat2", "feat3", "feat4",
        "reg_input1", "reg_input2", "reg_input3", "reg_input4"
    ]]

    print(final_df[["sku_id", "demand", "feat1", "reg_input4"]].head(30))

    # ========================= Convert to sequence format =========================
    data_dir = Path(__file__).parent
    project_root = data_dir.parent
    sys.path.insert(0, str(project_root))

    out_dir = str(data_dir / OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # Split: first NUM_SIGMA -> train, next NUM_SIGMA -> val, rest -> test
    all_skus = sorted(final_df["sku_id"].unique())
    train_skus = all_skus[:NUM_SIGMA]
    val_skus = all_skus[NUM_SIGMA:2 * NUM_SIGMA]
    test_skus = all_skus[2 * NUM_SIGMA:]

    print(f"#total SKUs: {len(all_skus)}")
    for name, lst in zip(("train", "val", "test"), (train_skus, val_skus, test_skus)):
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump([int(x) for x in lst], f)
        print(f"  {name:5s}: {len(lst)} skus -> {path}")

    # Convert to syn_full.csv
    seq_convert(final_df, train_skus + val_skus + test_skus,
                os.path.join(out_dir, "syn_full.csv"),
                T=target_days, lead_time=LEAD_TIME)

    print(f"\nDone! Output directory: {out_dir}")


if __name__ == "__main__":
    main()
