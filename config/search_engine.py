#!/usr/bin/env python3
"""Modular search engine supporting Ray Tune and exhaustive grid search."""

import os
import sys
import subprocess
import tempfile
import shutil
import time
import itertools
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass

# Set Ray environment variables before importing Ray
os.environ["RAY_DEDUP_LOGS"] = "0"

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
import yaml

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.hyperparameter_registry import get_hyperparameters, standardize_parameters


@dataclass
class SearchResult:
    """Result from a hyperparameter search."""
    best_config: Dict[str, Any]
    best_metric: float
    all_results: List[Dict[str, Any]]
    total_trials: int
    successful_trials: int
    elapsed_time: float
    search_type: str
    analysis: Optional[Any] = None  # Ray Tune analysis object for intelligent search


class UniversalSearchEngine:
    """Universal hyperparameter search engine supporting Ray Tune and grid search."""
    
    def __init__(self, 
                 algorithm: str,
                 base_config_path: str,
                 training_script_path: str,
                 metric_parser: Callable[[str], Dict[str, float]],
                 metric_name: str = "reward_mean",
                 metric_mode: str = "max",
                 num_seeds: int = 3,
                 custom_params: Optional[Dict[str, Any]] = None,
                 extra_cmd_args: Optional[List[str]] = None):
        """Initialize the search engine."""
        self.algorithm = algorithm.lower()
        self.base_config_path = base_config_path
        self.training_script_path = training_script_path
        self.metric_parser = metric_parser
        self.metric_name = metric_name
        self.metric_mode = metric_mode
        self.num_seeds = num_seeds
        self.custom_params = custom_params or {}
        self.extra_cmd_args = extra_cmd_args or []
        self.project_root = project_root
        
        # Ensure custom_params is properly initialized
        if not hasattr(self, 'custom_params'):
            self.custom_params = {}
        
        # Get hyperparameter configuration from registry
        self.config = get_hyperparameters(self.algorithm)

        # Seeds to use
        self.seeds = [42, 123, 456] if num_seeds == 3 else list(range(42, 42 + num_seeds))

        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init()
    
    def create_config_file(self, params: Dict[str, Any], temp_dir: str) -> str:
        """Create a temp config file with the given parameters."""
        # Load base config - ensure path is relative to project root
        config_path = self.project_root / self.base_config_path if not os.path.isabs(self.base_config_path) else self.base_config_path
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load global config and merge it
        global_config_path = self.project_root / "config" / "global.yaml"
        with open(global_config_path, 'r') as f:
            global_config = yaml.safe_load(f)
        
        # Merge global config
        if 'global' in global_config:
            config.update(global_config['global'])
        
        # Remove the includes section since we've already merged global config
        if 'includes' in config:
            del config['includes']
        
        # Add custom parameters first
        custom_params = getattr(self, 'custom_params', {})
        for param_name, param_value in custom_params.items():
            config['setup'][param_name] = param_value
        
        # Standardize and update parameters (these can override custom params)
        standardized_params = standardize_parameters(params)
        for param_name, param_value in standardized_params.items():
            if param_name != 'seed':  # Skip seed, it's handled separately
                config['setup'][param_name] = param_value
        
        # Point trial log_dir_base into the temp dir (cleaned up after trial)
        if 'log_dir_base' in config.get('setup', {}):
            config['setup']['log_dir_base'] = temp_dir
        elif 'log_dir_base' in config:
            config['log_dir_base'] = temp_dir
        else:
            config['log_dir_base'] = temp_dir
        
        # Write to temporary file (preserve base config name for log directory derivation)
        temp_config_path = os.path.join(temp_dir, Path(self.base_config_path).name)
        with open(temp_config_path, 'w') as f:
            yaml.safe_dump(config, f)
            
        return temp_config_path
    
    def run_single_trial(self, config: Dict[str, Any], trial_num: int = 0, total_trials: int = 0) -> Dict[str, float]:
        """Run a single trial with given hyperparameters and seed."""
        trial_start_time = time.time()
        trial_info = f"{self.algorithm.upper()} Trial {trial_num}/{total_trials}" if total_trials > 0 else f"{self.algorithm.upper()} Trial"
        print(f"🚀 STARTING {trial_info} {config}")
        
        # Create temporary directory for this trial
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create config file for this trial
            temp_config_path = self.create_config_file(config, temp_dir)
            
            # Prepare command
            cmd = [
                sys.executable, 
                str(self.project_root / self.training_script_path),
                "--configs", temp_config_path,
                "--seed", str(config['seed'])
            ]
            
            # Add any extra command-line arguments
            cmd.extend(self.extra_cmd_args)
            
            # For DS and RL algorithms, add skip-full-eval flag during hyperparameter search
            if self.algorithm in ['ds', 'ppo', 'ddpg']:
                cmd.append("--skip-full-eval")
            
            # Run training
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=str(self.project_root),
                env=env
            )
            
            trial_duration = time.time() - trial_start_time
            
            if result.returncode != 0:
                print(f"❌ FAILED {trial_info} {config} (took {trial_duration:.1f}s)")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                
                # Return worst possible metric value
                worst_metric = float('-inf') if self.metric_mode == 'max' else float('inf')
                return {self.metric_name: worst_metric, "success": False, "duration": trial_duration}
            
            # Parse metrics from output
            metrics = self.metric_parser(result.stdout)
            # print(result.stdout)

            metrics['success'] = True
            metrics['duration'] = trial_duration
            
            metric_value = metrics.get(self.metric_name, float('-inf') if self.metric_mode == 'max' else float('inf'))
            
            # For DS algorithm, also display training loss alongside validation loss
            if self.algorithm == 'ds':
                train_loss = metrics.get('train_loss', float('inf'))
                print(f"✅ COMPLETED {trial_info} {config} -> train: {train_loss:.4f}, val: {metric_value:.4f} (took {trial_duration:.1f}s)")
            # For RL algorithms, display reward_mean with additional context
            elif self.algorithm in ['ppo', 'ddpg']:
                total_loss = metrics.get('total_loss', float('nan'))
                print(f"✅ COMPLETED {trial_info} {config} -> reward: {metric_value:.4f} (loss: {total_loss:.4f}) (took {trial_duration:.1f}s)")
            else:
                print(f"✅ COMPLETED {trial_info} {config} -> {self.metric_name}: {metric_value:.4f} (took {trial_duration:.1f}s)")
            
            return metrics
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def run_intelligent_search(self, num_samples: int = 50, max_concurrent: int = 4, n_startup_trials: int = 10) -> SearchResult:
        """Run intelligent search using Ray Tune with Optuna."""
        print(f"Starting {self.algorithm.upper()} intelligent search with Ray Tune...")
        print(f"Max samples: {num_samples}, Max concurrent: {max_concurrent}")
        
        # Define the training function for Ray Tune
        def train_model(config):
            """Training function that Ray Tune will call with different configs."""
            # Add a fixed seed for this trial to ensure reproducibility
            config_with_seed = config.copy()
            if 'seed' not in config_with_seed:
                # Use a deterministic seed based on config hash for reproducibility
                config_str = str(sorted(config.items()))
                import hashlib
                seed_hash = int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16) % 10000
                config_with_seed['seed'] = seed_hash
            
            # Run the trial
            result = self.run_single_trial(config_with_seed)
            
            # Return the primary metric (Ray Tune can receive metrics this way too)
            primary_metric_value = result.get(self.metric_name, float('-inf') if self.metric_mode == 'max' else float('inf'))
            # Include the seed and other important metrics in the result
            # Exclude the primary metric from the spread to avoid duplication in Ray Tune table
            return {
                "metric_value": primary_metric_value, 
                "seed": config_with_seed['seed'],
                "success": result.get('success', False),
                "duration": result.get('duration', 0),
                **{k: v for k, v in result.items() if k not in ['metric_value', 'seed', 'success', 'duration', self.metric_name]}
            }
        
        # Set up Optuna search algorithm with better configuration
        import optuna
        search_alg = OptunaSearch(
            metric="metric_value",  # Simple metric name for returned values
            mode=self.metric_mode,
            sampler=optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)
        )
        print(f"🎲 Random startup trials: {n_startup_trials}")
        
        # For single-iteration trials (like our RL training), ASHA scheduler is not needed
        # and can interfere with OptunaSearch. Use basic FIFOScheduler instead.
        scheduler = None
        
        # Run the hyperparameter search
        print(f"🔍 Starting intelligent search with OptunaSearch")
        print(f"📊 Search space: {list(self.config.ray_tune_search.keys())}")
        start_time = time.time()
        
        # Configure progress reporter to show best trials
        from ray.tune.progress_reporter import CLIReporter
        reporter = CLIReporter(
            metric_columns=["metric_value"],
            max_progress_rows=5,  # Keep default 5 rows
            max_error_rows=5,
            metric="metric_value",  # Specify which metric to sort by
            mode=self.metric_mode,  # Specify whether to sort ascending or descending
            sort_by_metric=True,  # This ensures we see the 5 BEST configurations
            print_intermediate_tables=False  # Disable intermediate table updates
        )
        
        def short_dirname(trial):
            return f"trial_{trial.trial_id}"

        analysis = tune.run(
            train_model,
            config=self.config.ray_tune_search,
            num_samples=num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={"cpu": 1},
            max_concurrent_trials=max_concurrent,
            verbose=1,
            progress_reporter=reporter,
            raise_on_failed_trial=False,
            checkpoint_config=tune.CheckpointConfig(checkpoint_at_end=False),
            trial_dirname_creator=short_dirname,
        )
        
        elapsed_time = time.time() - start_time
        
        # Get best trial results
        best_trial = analysis.get_best_trial("metric_value", self.metric_mode)
        
        if best_trial is None:
            print("❌ No successful trials found!")
            return SearchResult(
                best_config=None,
                best_metric=float('-inf') if self.metric_mode == 'max' else float('inf'),
                all_results=[],
                total_trials=0,
                successful_trials=0,
                elapsed_time=elapsed_time,
                search_type="intelligent",
                analysis=analysis
            )
        
        # Process all trial results
        successful_trials = 0
        all_results = []
        total_training_time = 0
        
        for trial in analysis.trials:
            if trial.status == "TERMINATED" and trial.last_result:
                if trial.last_result.get("success", False):
                    successful_trials += 1
                    total_training_time += trial.last_result.get("duration", 0)
                    all_results.append({
                        **trial.config,
                        **trial.last_result
                    })
        
        best_metric = best_trial.last_result["metric_value"]
        
        # Create best config including the seed from the best trial
        best_config_with_seed = best_trial.config.copy()
        best_config_with_seed['seed'] = best_trial.last_result.get('seed', 42)  # Default seed if not found
        
        print(f"\n📈 {self.algorithm.upper()} INTELLIGENT SEARCH SUMMARY:")
        print(f"✅ Successful: {successful_trials}, ❌ Failed: {len(analysis.trials) - successful_trials}")
        print(f"⏱️  Total training time: {total_training_time:.1f}s, Avg per trial: {total_training_time/max(successful_trials,1):.1f}s")
        print(f"🏆 Best trial: {best_trial.trial_id}")
        print(f"🎯 Best {self.metric_name}: {best_metric:.4f}")
        print(f"⚙️  Best config: {best_config_with_seed}")
        print(f"⏱️  Search completed in {elapsed_time:.2f} seconds")
        
        return SearchResult(
            best_config=best_config_with_seed,
            best_metric=best_metric,
            all_results=all_results,
            total_trials=len(analysis.trials),
            successful_trials=successful_trials,
            elapsed_time=elapsed_time,
            search_type="intelligent",
            analysis=analysis
        )
    
    def run_exhaustive_search(self) -> SearchResult:
        """
        Run exhaustive grid search testing all parameter combinations.
        
        Returns:
            SearchResult object with grid search results
        """
        # Calculate total number of trials
        total_combinations = 1
        for param_values in self.config.grid_search.values():
            total_combinations *= len(param_values)
        total_trials = total_combinations * self.num_seeds
        
        print(f"Starting {self.algorithm.upper()} exhaustive grid search...")
        print(f"Total trials: {total_trials} ({total_combinations} combinations × {self.num_seeds} seeds)")
        
        # Create all parameter combinations
        trials = []
        param_names = list(self.config.grid_search.keys())
        param_values = [self.config.grid_search[name] for name in param_names]
        
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            for seed in self.seeds:
                trial_config = {**param_dict, 'seed': seed}
                trials.append(trial_config)
        
        # For exhaustive search, just run trials sequentially to avoid Ray serialization issues
        # This is actually more reliable for large grid searches
        start_time = time.time()
        trial_results = []
        for i, trial_config in enumerate(trials):
            result = self.run_single_trial(trial_config, i+1, len(trials))
            trial_results.append(result)
        
        # Process results
        successful_trials = 0
        all_results = []
        total_training_time = 0
        best_metric = float('-inf') if self.metric_mode == 'max' else float('inf')
        best_config = None
        
        for i, (trial_config, result) in enumerate(zip(trials, trial_results)):
            if result['success']:
                successful_trials += 1
                total_training_time += result.get('duration', 0)
                full_result = {**trial_config, **result}
                all_results.append(full_result)
                
                # Track best configuration
                metric_value = result[self.metric_name]
                is_better = (self.metric_mode == 'max' and metric_value > best_metric) or \
                           (self.metric_mode == 'min' and metric_value < best_metric)
                
                if is_better:
                    best_metric = metric_value
                    best_config = trial_config.copy()
                    if self.algorithm == 'ds':
                        train_loss = result.get('train_loss', float('inf'))
                        print(f"🏆 NEW BEST {self.algorithm.upper()}! Trial {i+1}/{len(trials)} - train: {train_loss:.4f}, val: {metric_value:.4f}, Config: {trial_config}")
                    elif self.algorithm in ['ppo', 'ddpg']:
                        total_loss = result.get('total_loss', float('nan'))
                        print(f"🏆 NEW BEST {self.algorithm.upper()}! Trial {i+1}/{len(trials)} - reward: {metric_value:.4f} (loss: {total_loss:.4f}), Config: {trial_config}")
                    else:
                        print(f"🏆 NEW BEST {self.algorithm.upper()}! Trial {i+1}/{len(trials)} - {self.metric_name}: {metric_value:.4f}, Config: {trial_config}")
                else:
                    if self.algorithm == 'ds':
                        train_loss = result.get('train_loss', float('inf'))
                        print(f"📊 {self.algorithm.upper()} Trial {i+1}/{len(trials)} - train: {train_loss:.4f}, val: {metric_value:.4f}, Config: {trial_config}")
                    elif self.algorithm in ['ppo', 'ddpg']:
                        total_loss = result.get('total_loss', float('nan'))
                        print(f"📊 {self.algorithm.upper()} Trial {i+1}/{len(trials)} - reward: {metric_value:.4f} (loss: {total_loss:.4f}), Config: {trial_config}")
                    else:
                        print(f"📊 {self.algorithm.upper()} Trial {i+1}/{len(trials)} - {self.metric_name}: {metric_value:.4f}, Config: {trial_config}")
            else:
                print(f"💥 {self.algorithm.upper()} Trial {i+1}/{len(trials)} - FAILED, Config: {trial_config}")
        
        elapsed_time = time.time() - start_time
        failed_trials = len(trials) - successful_trials
        
        print(f"\n📈 {self.algorithm.upper()} EXHAUSTIVE SEARCH SUMMARY:")
        print(f"✅ Successful: {successful_trials}, ❌ Failed: {failed_trials}")
        print(f"⏱️  Total training time: {total_training_time:.1f}s, Avg per trial: {total_training_time/max(successful_trials,1):.1f}s")
        print(f"🎯 Best {self.metric_name}: {best_metric:.4f}")
        print(f"⚙️  Best config: {best_config}")
        print(f"⏱️  Search completed in {elapsed_time:.2f} seconds")
        
        return SearchResult(
            best_config=best_config,
            best_metric=best_metric,
            all_results=all_results,
            total_trials=len(trials),
            successful_trials=successful_trials,
            elapsed_time=elapsed_time,
            search_type="exhaustive"
        )
    
    def run_search(self,
                   search_type: str = "intelligent",
                   num_samples: int = 50,
                   max_concurrent: int = 4,
                   n_startup_trials: int = 10) -> SearchResult:
        """
        Run hyperparameter search with the specified strategy.
        
        Args:
            search_type: 'intelligent' for Ray Tune optimization or 'exhaustive' for grid search
            num_samples: Number of samples for intelligent search (ignored for exhaustive)
            max_concurrent: Maximum concurrent trials (ignored for exhaustive)
            
        Returns:
            SearchResult object
        """
        if search_type.lower() == "intelligent":
            return self.run_intelligent_search(num_samples, max_concurrent, n_startup_trials)
        elif search_type.lower() == "exhaustive":
            return self.run_exhaustive_search()
        else:
            raise ValueError(f"Unknown search type: {search_type}. Use 'intelligent' or 'exhaustive'")
    
    def evaluate_best_model(self, search_result: SearchResult) -> Dict[str, Any]:
        """Train and evaluate the best model from search results."""
        if search_result.best_config is None:
            raise ValueError("No best configuration found in search results.")
        
        print(f"\nTraining and evaluating best {self.algorithm.upper()} model with config: {search_result.best_config}")
        
        # Create temporary directory for best model training
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create config file for best configuration
            temp_config_path = self.create_config_file(search_result.best_config, temp_dir)
            
            # Prepare command
            cmd = [
                sys.executable,
                str(self.project_root / self.training_script_path),
                "--configs", temp_config_path,
                "--seed", str(search_result.best_config['seed'])
            ]
            
            # Add any extra command-line arguments
            cmd.extend(self.extra_cmd_args)
            
            # Run training and evaluation
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(self.project_root),
                env=env
            )
            
            if result.returncode != 0:
                print(f"Best {self.algorithm.upper()} model training failed!")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return {"success": False}
            
            print(f"Best {self.algorithm.upper()} model training completed successfully!")
            
            # Parse results from output
            test_results = self.metric_parser(result.stdout)
            test_results['success'] = True
            test_results['best_config'] = search_result.best_config
            test_results['search_type'] = search_result.search_type
            test_results['stdout'] = result.stdout

            return test_results
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
