"""
Generate independent-demand synthetic data AND convert to sequence format.
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
RANDOM_DEMAND = True
TARGETS_DAYS = 31
TRAIN_NUM = 10
TEST_NUM = 100
NUM_SKUS = 2 * TRAIN_NUM + TEST_NUM  # TRAIN_NUM = VAL_NUM

SEED = 42
LEAD_TIME = 1
SPAN = 5

OUTPUT_DIR_NAME = f"data_ind_perffeat_T{TARGETS_DAYS}_train{TRAIN_NUM}"

# ========================= Demand generation =========================

np.random.seed(SEED)
LOW_DEMAND = np.clip(np.random.normal(loc=10, scale=4, size=TARGETS_DAYS), a_min=0, a_max=100).astype(int)
print('LOW_DEMAND:', LOW_DEMAND)


def sample_random_D_true(t: int, rng: np.random.Generator, span: int) -> int:
    low = LOW_DEMAND[t]
    return int(rng.integers(low, low + span))


def make_span_scenarios(T: int, span: int) -> np.ndarray:
    scen = []
    for k in range(span):
        path = []
        for t in range(T + 1):
            low = t % span
            path.append(low + k)
        scen.append(path)
    return np.array(scen, dtype=float)


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

    span_paths = make_span_scenarios(total_days - 1, SPAN)

    def get_demands_for_sku(sku_idx, random_flag=RANDOM_DEMAND):
        if random_flag:
            return [float(sample_random_D_true(t, rng, SPAN)) for t in range(total_days)]
        else:
            k = sku_idx % SPAN
            return list(span_paths[k])

    all_data = []
    for sku_idx in range(num_skus):
        demands = get_demands_for_sku(sku_idx, True)
        df = pd.DataFrame({
            "sku_id": sku_idx + 1,
            "day": list(range(total_days)),
            "demand": demands
        })
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)

    # Feature columns
    if NO_FEATURES:
        final_df["feat1"] = 0.0
        final_df["feat2"] = 0.0
        final_df["feat3"] = 0.0
        final_df["feat4"] = 0.0
    elif PERFECT_FEATURES:
        final_df["feat1"] = final_df["day"].astype(int) // BP
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

    # inputs for Coeff
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

    print(final_df[["sku_id", "demand", "feat1", "reg_input4"]].head(20))

    # ========================= Convert to sequence format =========================
    data_dir = Path(__file__).parent
    project_root = data_dir.parent
    sys.path.insert(0, str(project_root))

    out_dir = str(data_dir / OUTPUT_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # Split SKUs into train / val / test
    all_skus = sorted(final_df["sku_id"].unique())
    train_skus = all_skus[:TRAIN_NUM]
    val_skus = all_skus[TRAIN_NUM:2 * TRAIN_NUM]
    test_skus = all_skus[2 * TRAIN_NUM:]

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
