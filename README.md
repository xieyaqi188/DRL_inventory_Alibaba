# DeepStock: Reinforcement Learning with Policy Regularizations for Inventory Management

Code for synthetic experiments in **Yaqi Xie, Xinru Hao, Jiaxi Liu, Will Ma, Linwei Xin, Lei Cao, and Yidong Zhang. "Deepstock: Reinforcement learning with policy regularizations for inventory management."**

## Citation

The BibTeX entry for citing our paper:

```bibtex
@article{xie2025deepstock,
  title={DeepStock: Reinforcement Learning with Policy Regularizations for Inventory Management},
  author={Xie, Yaqi and Hao, Xinru and Liu, Jiaxi and Ma, Will and Xin, Linwei and Cao, Lei and Zhang, Yidong},
  journal={Available at SSRN 5784782},
  year={2025}
}
```

## Project Outline

```
config/                         # Configuration files
├── global.yaml                 # Global settings (e.g., max_replenish, lead time)
├── hyperparameter_registry.py  # Hyperparameter search ranges for Ray Tune
├── ds/                         # DS model configs (ds_base.yaml, ds_none.yaml, etc.)
├── rl/                         # RL model configs (ddpg_*.yaml, ppo_*.yaml)
└── search_engine.py            # Universal hyperparameter search engine

synthetic_data/                 # Data generation scripts and generated datasets
├── gene_syn_data_ind.py        # Setting 1: independent demand
├── gene_syn_data_ar1.py        # Settings 2 & 3: AR(1) demand
└── gene_syn_data_iid.py        # Setting 4: i.i.d. demand

ds/                             # Direct Supervision (DS) model
├── main_ds.py                  # Training entry point
├── hyperparameter_sweep_ds.py  # Hyperparameter tuning
├── trainer.py                  # Training loop
├── neural_network.py           # Network architecture
├── simulation.py               # DS environment
├── configs.py                  # DS configuration loader
└── dataset.py                  # Dataset utilities

rl/                             # Reinforcement Learning (DDPG & PPO)
├── main_rl.py                  # Training entry point
├── hyperparameter_sweep_ddpg.py  # DDPG hyperparameter tuning
├── hyperparameter_sweep_ppo.py   # PPO hyperparameter tuning
├── trainer.py                  # Training loop
├── trainer_ddpg_critic.py      # DDPG critic training
└── inv_env.py                  # Gym environment
```

## Data Generation

Scripts are in `synthetic_data/`:

- **Setting 1** (independent demand): `gene_syn_data_ind.py`
- **Settings 2 & 3** (AR(1) demand): `gene_syn_data_ar1.py`
- **Setting 4** (IID demand): `gene_syn_data_iid.py`

## Configuration

- `config/global.yaml`: global parameters (lead time, review period, max replenishment, etc.)
- `config/ds/*.yaml` and `config/rl/*.yaml`: hyperparameters specific to each DRL method and regularization approach. Configurations with `_base` use BASE regularization; configs with `_none` use no regularization; configs with `_coeff_base` and `_coeff_none` use BOTH and COEFF regularizations, respectively.
  - The current `ds_none`, `ds_base`, `ddpg_none`, `ddpg_base`, `ppo_none`, and `ppo_base` configs include a corresponding set of tuned hyperparameters for Setting 1.
- `config/hyperparameter_registry.py`: search ranges for Ray Tune.

## Training

### Single Run

```bash
# DS
python ds/main_ds.py --configs config/ds/ds_none.yaml

# DDPG
python rl/main_rl.py --configs config/rl/ddpg_none.yaml

# PPO
python rl/main_rl.py --configs config/rl/ppo_none.yaml
```

### Hyperparameter Sweep

```bash
# DS
python ds/hyperparameter_sweep_ds.py --config config/ds/ds_none.yaml

# DDPG
python rl/hyperparameter_sweep_ddpg.py --config config/rl/ddpg_none.yaml

# PPO
python rl/hyperparameter_sweep_ppo.py --config config/rl/ppo_none.yaml
```

## Benchmarks

- **Setting 1**: empirical dynamic programming, computed in `dp_inventory.py`.
- **Settings 2 & 3**: approximated by running DS with testing samples.
- **Setting 4**: computed in `synthetic_data/gene_syn_data_iid.py` along with data generation.

## Requirements

- Python >= 3.8
- numpy
- pandas
- PyYAML
- torch (PyTorch)
- stable-baselines3
- gymnasium
- ray[tune]
- tqdm
- matplotlib

Install all dependencies:

```bash
pip install numpy pandas pyyaml torch stable-baselines3 gymnasium "ray[tune]" tqdm matplotlib
```
