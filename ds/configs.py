import sys
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_global_config():
    global_path = project_root / "config/global.yaml"
    with open(global_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    g = cfg.get('global', {})
    return (g.get('max_period_num', 31), g.get('max_lead_time', 1),
            g.get('lost_sales_demand', False), g.get('max_replenish', 100),
            g.get('max_turnover', 28), g.get('data_directory', ''))

DATE_NUM, LEAD_TIME, LOST_SALES_DEMAND, MAX_REPLENISH, MAX_TURNOVER, DATA_DIRECTORY = load_global_config()

# Resolve data directory and set global data paths
from paths import set_data_paths, get_data_file_path
_data_dir = str(project_root / DATA_DIRECTORY) if DATA_DIRECTORY else None
set_data_paths(_data_dir)


class DLConfig:

    PROBLEM_PARAMS = {
        "lost_sales_demand": LOST_SALES_DEMAND,
        "holding_cost": 1.0,
        "backlog_cost": 9.0,
        "max_period_num": DATE_NUM,
        "max_lead_time": LEAD_TIME,
        "max_replenish": MAX_REPLENISH,
        "max_turnover": MAX_TURNOVER,
    }

    DATASET_PARAMS = {
        "file_location": str(get_data_file_path('syn_full.csv')),
        "period_num_col": ["period_num"],
        "lead_time_col": ["lt"],
        "review_period_col": ["bp"],
        "demand_seq_col": ["demand"],
        "initial_inventory_col": ["stock"],
        "features_col": ["feature"],
        "reg_input_col": ["reg_input"],
    }

    STATE_PARAMS = {
        "period_num": True,
        "ignore_period_num": True,
        "period_state": {
            "demands": True,
            "features": True,
            "features_org": True,
            "lead_times": True,
            "reg_input": True,
        },
        "static_state": {
            "holding_cost": True,
            "backlog_cost": True,
            "review_period": True,
        },
        "past_state": {
            "past_arrivals": 4,
            "past_orders": 4,
        },
    }
