#!/usr/bin/env python3
"""Centralized hyperparameter registry for all algorithms."""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import ray.tune as tune


@dataclass
class HyperparameterConfig:
    """Hyperparameter config for a specific algorithm."""
    grid_search: Dict[str, List[Any]]
    ray_tune_search: Optional[Dict[str, Any]] = None
    description: str = ""


def create_ds_config() -> HyperparameterConfig:
    """DS hyperparameters."""
    return HyperparameterConfig(
        description="Deep Learning (Neural Network) hyperparameters",
        grid_search={
        },
        ray_tune_search={
            'batch_size': tune.choice([5, 10]),
            'learning_rate': tune.loguniform(4e-4, 1e-1),
            'initial_action_bias': tune.uniform(0.2, 0.7),
            'neurons_per_hidden_layer': tune.choice([[32, 32], [64, 64], [128, 128]]),
            'scheduler_factor': tune.uniform(0.5, 0.95),
        }
    )


def create_ppo_config() -> HyperparameterConfig:
    """PPO hyperparameters."""
    return HyperparameterConfig(
        description="PPO (Proximal Policy Optimization) hyperparameters",
        grid_search={
        },
        ray_tune_search={
            'eval_freq': tune.choice([512, 1024, 2048]),
            'learning_rate': tune.loguniform(5e-5, 9e-3),
            'clip_range': tune.uniform(0.1, 0.3),
            'batch_size': tune.choice([128, 256, 512]),
            'n_steps': tune.choice([1024, 2048, 4096]),
            'n_epochs': tune.choice([1, 3, 5, 10]),
            'ent_coef': tune.uniform(1e-4, 5e-1),
            'lr_min': tune.loguniform(1e-5, 5e-5),
            'initial_action_bias': tune.uniform(0.2, 0.7),
            'gamma': tune.uniform(0.98, 1.0),
            'gae_lambda': tune.uniform(0.5, 1.0),
            'net_arch': tune.choice([
                {'pi': [256, 256], 'vf': [256, 256]},
            ])
        }
    )


def create_ddpg_config() -> HyperparameterConfig:
    """DDPG hyperparameters."""
    return HyperparameterConfig(
        description="DDPG (Deep Deterministic Policy Gradient) hyperparameters",
        grid_search={
        },
        ray_tune_search={
            'eval_freq': tune.choice([512, 1024]),
            'learning_rate': tune.loguniform(3e-4, 9e-3),
            'tau': tune.loguniform(1e-3, 1e-2),
            'batch_size': tune.choice([128, 256, 512]),
            'buffer_size': tune.choice([20_000, 50_000, 100_000]),
            'train_freq': tune.choice([1, 14]),
            'learning_starts': tune.choice([100, 500, 1000]),
            'lr_min': tune.loguniform(6e-5, 3e-4),
            'initial_action_bias': tune.uniform(0.2, 0.7),
            'gamma': tune.uniform(0.98, 1.0),
            'net_arch': tune.choice([
                {'pi': [64, 64], 'qf': [64, 64]},
            ])
        }
    )


HYPERPARAMETER_REGISTRY: Dict[str, HyperparameterConfig] = {
    'ds': create_ds_config(),
    'ppo': create_ppo_config(),
    'ddpg': create_ddpg_config(),
}


def get_hyperparameters(algorithm: str) -> HyperparameterConfig:
    """Get hyperparameter config for a given algorithm."""
    algorithm = algorithm.lower()
    if algorithm not in HYPERPARAMETER_REGISTRY:
        available = list(HYPERPARAMETER_REGISTRY.keys())
        raise ValueError(f"Algorithm '{algorithm}' not found. Available: {available}")
    
    return HYPERPARAMETER_REGISTRY[algorithm]


def list_algorithms() -> List[str]:
    """List all available algorithms."""
    return list(HYPERPARAMETER_REGISTRY.keys())


def get_parameter_mapping() -> Dict[str, str]:
    """Map old parameter names to standardized names."""
    return {
        'lr': 'learning_rate',
        'n_step': 'n_steps',
    }


def standardize_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert old parameter names to standardized names."""
    mapping = get_parameter_mapping()
    standardized = {}
    
    for key, value in params.items():
        new_key = mapping.get(key, key)
        standardized[new_key] = value
    
    return standardized


if __name__ == "__main__":
    print("Available algorithms:", list_algorithms())
    print()
    
    for algo in ['ds', 'ppo', 'ddpg']:
        config = get_hyperparameters(algo)
        print(f"{algo.upper()} Configuration:")
        print(f"  Description: {config.description}")
        print(f"  Grid search options: {len(config.grid_search)} parameters")
        print(f"  Ray Tune support: {config.ray_tune_search is not None}")
        print(f"  Tunable parameters: {', '.join(config.grid_search.keys())}")
        print()