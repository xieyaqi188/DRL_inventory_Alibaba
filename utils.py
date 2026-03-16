import os, sys, random, numpy as np, torch
from datetime import datetime
from stable_baselines3.common.monitor import Monitor
from rl.inv_env import InventorySimEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines3.common.utils import set_random_seed


class TeeStdout:
    """Duplicate stdout to both console and a file."""
    def __init__(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def setup_stdout_log(subdir, config_stem, log_folder="logs_single_run"):
    """Start teeing stdout to <project_root>/{subdir}/{log_folder}/{config_stem}_{datetime}.txt.
    Returns the TeeStdout object (call .close() when done)."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    log_path = os.path.join(PROJECT_ROOT, subdir, log_folder, f"{config_stem}_{timestamp}.txt")
    tee = TeeStdout(log_path)
    print(f"Logging to {log_path}")
    return tee


def fix_seed(seed):
    """Fix all random seeds for deterministic training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        torch.set_num_threads(1)

    set_random_seed(seed)
    print(f"Fixed all random seeds to {seed} for deterministic training")


def make_env(sku_subset, action_mode, args, trainenv=True, deterministic_sku_cycle=False):

    from paths import get_data_file_path
    data_path = str(get_data_file_path('syn_full.csv'))
    def f():
        return Monitor(
            InventorySimEnv(sku_id=None, args=args, sku_whitelist=sku_subset,
                          action_mode=action_mode, train=trainenv,
                          deterministic_sku_cycle=deterministic_sku_cycle,
                          data_path=data_path),
            filename=None
        )
    return f

def add_action_noise(action_noise, train_env):
    if action_noise not in ('ou', 'norm'):
        return None
    n_actions = train_env.action_space.shape[-1]
    mean = np.zeros(n_actions)
    if action_noise == 'ou':
        return OrnsteinUhlenbeckActionNoise(mean=mean, sigma=0.1 * np.ones(n_actions))
    return NormalActionNoise(mean=mean, sigma=0.3 * np.ones(n_actions))

def to_metrics_dict(weighted_loss, underage_list, overage_list,
                    service_level_list, turnover_list, avg_inv_list, avg_sales_list):
    """Convert evaluation lists to a metrics dict."""
    return {
        'weighted_loss': float(weighted_loss),
        'underage': float(np.mean(underage_list)),
        'overage': float(np.mean(overage_list)),
        'service_level_mean': float(np.mean(service_level_list)),
        'turnover_mean': float(np.mean(turnover_list)),
        'avg_inventory': float(np.mean(avg_inv_list)),
        'avg_sales': float(np.mean(avg_sales_list)),
    }
