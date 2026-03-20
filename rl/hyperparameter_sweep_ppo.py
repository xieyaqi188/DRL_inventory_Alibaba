import sys
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.search_engine import UniversalSearchEngine
from utils import setup_stdout_log

GAE_LAMBDA = 0.95
CONFIG_YAML = "config/rl/ppo_base.yaml"


def parse_ppo_metrics(stdout: str) -> Dict[str, float]:
    """
    Parse PPO training metrics from output.
    
    Parses both EvalCallback episode_reward and detailed evaluation metrics
    (eval_train_loss, eval_val_loss, eval_test_loss).
    """
    lines = stdout.split('\n')
    metrics = {}
    
    # Parse EvalCallback metrics - look for the best (highest) episode reward
    best_episode_reward = float('-inf')
    found_eval_callback = False
    
    # Variables for detailed evaluation parsing
    found_detailed_eval = False
    current_set = None
    
    for line in lines:
        line = line.strip()
        
        # Parse EvalCallback output: "Eval num_timesteps=X, episode_reward=Y.Y +/- Z.Z"
        if 'Eval num_timesteps=' in line and 'episode_reward=' in line and '+/-' in line:
            try:
                # Extract the reward value before the "+/-"
                # Split by episode_reward= and take everything before +/-
                after_reward = line.split('episode_reward=')[1]
                reward_part = after_reward.split('+/-')[0].strip()
                episode_reward = float(reward_part)
                
                # Keep track of the best (highest) episode reward during training
                best_episode_reward = max(best_episode_reward, episode_reward)
                found_eval_callback = True
            except (IndexError, ValueError):
                continue
        
        # === Detailed evaluation parsing (copied from DDPG) ===
        # Track which dataset is being evaluated
        if '=== Start evaluating on' in line:
            found_detailed_eval = True
            if 'train-set' in line:
                current_set = 'train'
            elif 'validation-set' in line:
                current_set = 'val'
            elif 'test-set' in line:
                current_set = 'test'
        
        # Parse service level
        if 'service level' in line and '=' in line:
            try:
                metrics['service_level'] = float(line.split('=')[1].strip())
            except (IndexError, ValueError):
                continue
        
        # Parse turnover days
        elif 'turnover days' in line and '=' in line:
            try:
                # turnover days = 16.98
                metrics['turnover_days'] = float(line.split('=')[1].strip())
            except (IndexError, ValueError):
                continue
        
        # Parse underage
        elif 'underage' in line and '=' in line and 'ratio' not in line:
            try:
                # underage                = 0.050
                metrics['underage'] = float(line.split('=')[1].strip())
            except (IndexError, ValueError):
                continue

        # Parse overage
        elif 'overage' in line and '=' in line and 'ratio' not in line:
            try:
                # overage                 = 0.612
                metrics['overage'] = float(line.split('=')[1].strip())
            except (IndexError, ValueError):
                continue

        # Parse ratio_overage
        elif 'ratio_overage' in line and '=' in line:
            try:
                # ratio_overage           = 0.1
                metrics['ratio_overage'] = float(line.split('=')[1].strip())
            except (IndexError, ValueError):
                continue

        # Parse loss and capture per-set losses
        elif 'loss' in line and '=' in line and 'ratio' not in line and 'underage' not in line and 'overage' not in line:
            try:
                # loss                    = 0.567
                loss_value = float(line.split('=')[1].strip())
                metrics['total_loss'] = loss_value
                # Also capture set-specific loss
                if current_set:
                    metrics[f'eval_{current_set}_loss'] = loss_value
            except (IndexError, ValueError):
                continue

    # If we found detailed evaluation metrics, calculate reward_mean and return
    if found_detailed_eval and 'total_loss' in metrics:
        metrics['reward_mean'] = -metrics['total_loss']
        return metrics
    elif found_detailed_eval and ('underage' in metrics and 'overage' in metrics):
        # Fallback calculation for detailed eval
        ratio_overage = metrics.get('ratio_overage', 0.1)
        combined_loss = metrics['overage'] * ratio_overage + metrics['underage'] * (1 - ratio_overage)
        metrics['reward_mean'] = -combined_loss
        return metrics
    
    # Fallback to EvalCallback reward if found
    if found_eval_callback:
        metrics['reward_mean'] = best_episode_reward
        return metrics
    
    # No metrics found - return a clear failure value
    return {'reward_mean': float('-inf')}


class PPOHyperparameterSweep:
    def __init__(self,
                 base_config_path: str = "config/rl/ppo_base.yaml",
                 num_seeds: int = 3,
                 gae_lambda: float = 0.95,
                 extra_cmd_args: Optional[List[str]] = None):
        """
        Initialize PPO hyperparameter sweep using the universal search engine.
        
        Args:
            base_config_path: Path to base YAML configuration file
            num_seeds: Number of random seeds to run for each configuration
            gae_lambda: GAE lambda parameter for bias-variance tradeoff
            extra_cmd_args: Optional list of extra command-line arguments to pass to training script
        """
        self.search_engine = UniversalSearchEngine(
            algorithm="ppo",
            base_config_path=base_config_path,
            training_script_path="rl/main_rl.py",
            metric_parser=parse_ppo_metrics,
            metric_name="reward_mean",
            metric_mode="max",
            num_seeds=num_seeds,
            extra_cmd_args=extra_cmd_args,
            custom_params={"gae_lambda": gae_lambda}
        )
        self.metric_history = []  # Track all configuration metric values
        
    def run_sweep(self,
                  search_type: str = "intelligent",
                  num_samples: int = 50,
                  max_concurrent: int = 4,
                  top_k: int = 5,
                  stage2_seeds: int = 5,
                  n_startup_trials: int = 10):
        """
        Run hyperparameter sweep with specified search strategy.

        For intelligent search, implements two-stage approach:
        - Stage 1: Quick single-seed search to find top-K configurations
        - Stage 2: Re-evaluate top-K with multiple seeds

        Args:
            search_type: 'intelligent' for Ray Tune optimization or 'exhaustive' for grid search
            num_samples: Number of samples for intelligent search (ignored for exhaustive)
            max_concurrent: Maximum concurrent trials (ignored for exhaustive)
            top_k: Number of top configurations to re-evaluate with multiple seeds (intelligent only)
            stage2_seeds: Number of seeds to use in stage 2 evaluation (intelligent only)
            n_startup_trials: Number of random trials before Bayesian optimization (default: 10)

        Returns:
            SearchResult object
        """
        # Run initial search
        search_result = self.search_engine.run_search(search_type, num_samples, max_concurrent, n_startup_trials)
        
        # Record initial metrics
        self.record_metrics(search_result)
        self.print_metric_summary()

        # For intelligent search, run stage 2 with multiple seeds
        if search_type == "intelligent" and search_result.successful_trials > 0:
            search_result = self.run_stage2_evaluation(search_result, top_k, stage2_seeds)
        
        return search_result
    
    def run_stage2_evaluation(self, stage1_result, top_k: int, num_seeds: int):
        """
        Stage 2: Re-evaluate top-K configurations with multiple seeds in parallel.
        
        Args:
            stage1_result: SearchResult from stage 1
            top_k: Number of top configurations to evaluate
            num_seeds: Number of seeds per configuration
            
        Returns:
            Updated SearchResult with multi-seed evaluation
        """
        print(f"\n🔬 STAGE 2: Re-evaluating top-{top_k} configurations with {num_seeds} seeds each...")
        
        # Sort configurations by metric value (Ray Tune stores as 'metric_value')
        sorted_configs = sorted(
            stage1_result.all_results,
            key=lambda x: x.get('metric_value', float('-inf')),
            reverse=(self.search_engine.metric_mode == 'max')
        )
        
        # Take top-K configurations
        top_configs = sorted_configs[:min(top_k, len(sorted_configs))]
        
        # Import ray for parallel execution
        import ray
        
        # Define a Ray remote function for running trials
        @ray.remote
        def run_trial_remote(search_engine, config_with_seed, trial_num, total_trials):
            return search_engine.run_single_trial(config_with_seed, trial_num, total_trials)
        
        # Prepare all trials for parallel execution
        all_trials = []
        trial_configs = []
        
        for i, config_result in enumerate(top_configs):
            # Extract just the hyperparameters (exclude seed and results)
            config = {k: v for k, v in config_result.items() 
                     if k not in ['seed', 'success', 'duration', 'metric_value'] 
                     and not (k.startswith('eval_') and k != 'eval_freq') and k != self.search_engine.metric_name}
            
            # Create trials for all seeds of this config
            config_trials = []
            for seed in range(42, 42 + num_seeds):
                config_with_seed = config.copy()
                config_with_seed['seed'] = seed
                trial_num = i * num_seeds + (seed - 42) + 1
                
                # Submit trial to Ray
                future = run_trial_remote.remote(
                    self.search_engine,
                    config_with_seed,
                    trial_num,
                    len(top_configs) * num_seeds
                )
                config_trials.append(future)
                trial_configs.append((i, config, seed, config_with_seed))
            
            all_trials.extend(config_trials)
        
        print(f"📡 Submitted {len(all_trials)} trials for parallel execution...")
        
        # Wait for all trials to complete and organize results by config
        config_results = {i: {'config': config, 'seed_results': [], 'seed_metrics': []} 
                         for i, config in enumerate([{k: v for k, v in cfg.items() 
                                                     if k not in ['seed', 'success', 'duration', 'metric_value'] 
                                                     and not (k.startswith('eval_') and k != 'eval_freq') and k != self.search_engine.metric_name} 
                                                    for cfg in top_configs])}
        
        # Process results as they complete
        completed = 0
        for future, (config_idx, config, seed, config_with_seed) in zip(all_trials, trial_configs):
            result = ray.get(future)
            completed += 1
            
            if completed % 10 == 0 or completed == len(all_trials):
                print(f"  Progress: {completed}/{len(all_trials)} trials completed...")
            
            if result['success']:
                # Store result with seed info
                result_with_seed = (seed, result)
                config_results[config_idx]['seed_results'].append(result_with_seed)
                config_results[config_idx]['seed_metrics'].append((seed, result[self.search_engine.metric_name]))
        
        # Calculate statistics and find best config
        stage2_results = []
        best_avg_metric = float('-inf') if self.search_engine.metric_mode == 'max' else float('inf')
        best_config = None
        best_config_details = None
        
        for i, cfg_data in config_results.items():
            config = cfg_data['config']
            seed_results = cfg_data['seed_results']
            seed_metrics = cfg_data['seed_metrics']
            
            print(f"\n📊 Config {i+1}/{len(top_configs)}: {config}")
            
            # Calculate statistics
            if seed_metrics:
                # Extract just the metric values for statistics
                metric_values = [m[1] for m in seed_metrics]
                avg_metric = np.mean(metric_values)
                std_metric = np.std(metric_values)
                min_metric = np.min(metric_values)
                max_metric = np.max(metric_values)
                
                # Convert seed_results to list of dicts for JSON serialization
                seed_results_list = [result for seed, result in seed_results]
                
                config_summary = {
                    **config,
                    'avg_' + self.search_engine.metric_name: avg_metric,
                    'std_' + self.search_engine.metric_name: std_metric,
                    'min_' + self.search_engine.metric_name: min_metric,
                    'max_' + self.search_engine.metric_name: max_metric,
                    'num_successful_seeds': len(seed_metrics),
                    'seed_results': seed_results_list
                }
                
                stage2_results.append(config_summary)
                
                print(f"  → Avg {self.search_engine.metric_name}: {avg_metric:.4f} ± {std_metric:.4f} "
                      f"(min: {min_metric:.4f}, max: {max_metric:.4f})")
                
                # Track best configuration based on average performance
                is_better = (self.search_engine.metric_mode == 'max' and avg_metric > best_avg_metric) or \
                           (self.search_engine.metric_mode == 'min' and avg_metric < best_avg_metric)
                
                if is_better:
                    best_avg_metric = avg_metric
                    best_config = config.copy()
                    # Use the seed that achieved the median performance
                    # seed_metrics is now a list of (seed, metric) tuples
                    sorted_by_metric = sorted(seed_metrics, key=lambda x: x[1])
                    median_idx = len(sorted_by_metric) // 2
                    best_config['seed'] = sorted_by_metric[median_idx][0]  # Get the seed from the tuple
                    best_config_details = config_summary
        
        # Create updated SearchResult
        print(f"\n🏆 STAGE 2 BEST: {best_config} with metric loss {sorted_by_metric[median_idx][1]}; avg {self.search_engine.metric_name}: {best_avg_metric:.4f}")
        
        # Print train and val loss for best configuration using sorted_by_metric
        if sorted_by_metric:
            print(f"\n📊 BEST CONFIG DETAILED RESULTS:")
            for i, (seed, metric_value) in enumerate(sorted_by_metric):
                print(f"  Seed {seed}: metric loss={metric_value}")
        
        # Update the search result with stage 2 findings
        stage1_result.best_config = best_config
        stage1_result.best_metric = best_avg_metric
        stage1_result.stage2_results = stage2_results
        stage1_result.stage2_best_details = best_config_details
        
        return stage1_result
    
    def record_metrics(self, search_result):
        """Record all configuration metric values."""
        if hasattr(search_result, 'all_results') and search_result.all_results:
            for result in search_result.all_results:
                if isinstance(result, dict) and 'metric_value' in result:
                    metric_value = result['metric_value']
                    # Extract config info for better tracking
                    config_summary = {k: v for k, v in result.items() 
                                    if k not in ['metric_value', 'success', 'duration'] 
                                    and not (k.startswith('eval_') and k != 'eval_freq')}
                    self.metric_history.append({
                        'config': config_summary,
                        'metric_value': metric_value
                    })
    
    def print_metric_summary(self):
        """Print summary of all recorded metric values."""
        if not self.metric_history:
            print("\n📊 No metric values recorded.")
            return
            
        print(f"\n📊 STAGE-1 METRIC SUMMARY: Recorded {len(self.metric_history)} configurations")
        print("=" * 80)
        
        # Sort by metric value (descending for reward)
        sorted_metrics = sorted(self.metric_history, 
                              key=lambda x: x['metric_value'], 
                              reverse=True)
        
        # Print top 10 results
        top_n = min(10, len(sorted_metrics))
        print(f"🏆 TOP {top_n} STAGE-1 CONFIGURATIONS:")
        for i, entry in enumerate(sorted_metrics[:top_n], 1):
            config_str = ", ".join([f"{k}={v}" for k, v in entry['config'].items() 
                                  if k != 'seed'])
            print(f"  {i:2d}. Metric: {entry['metric_value']:.4f} | Config: {config_str}")
        
        # Print statistics
        metric_values = [entry['metric_value'] for entry in self.metric_history]
        print(f"\n📈 STAGE-1 STATISTICS:")
        print(f"  Best:  {max(metric_values):.4f}")
        print(f"  Worst: {min(metric_values):.4f}")
        print(f"  Mean:  {sum(metric_values)/len(metric_values):.4f}")
        print(f"  Count: {len(metric_values)}")
        
        # Print raw metric values list (order of execution)
        print(f"\n📋 STAGE-1 RAW METRIC VALUES (execution order):")
        print(f"  {metric_values}")
        print("=" * 80)

    def evaluate_best_model(self, search_result):
        """
        Train and evaluate the best model from search results using Ray remote execution for consistency.
        
        Args:
            search_result: SearchResult object from run_sweep
            
        Returns:
            Dictionary with test results
        """
        print(f"\n🎯 FINAL EVALUATION: Running best config through Ray infrastructure for identical execution environment")

        # Wait for all current Ray tasks to complete to clear the environment
        print(f"🧹 Waiting for all current Ray tasks to complete before final evaluation...")
        import ray
        ray.wait([ray.remote(lambda: None).remote()], num_returns=1)
        
        # Force garbage collection in Ray to clear any residual state
        import gc
        gc.collect()

        # Define a Ray remote function that calls the search engine's evaluate_best_model
        @ray.remote
        def run_evaluate_remote(search_engine, search_result):
            return search_engine.evaluate_best_model(search_result)

        print(f"📡 Running final evaluation with config: {search_result.best_config}")

        # Submit the evaluation as a Ray remote task
        future = run_evaluate_remote.remote(self.search_engine, search_result)

        # Wait for completion and get results
        result = ray.get(future)

        if result and 'success' in result and result['success']:
            print(f"✅ Final evaluation completed successfully")
            # Print the evaluation stdout in main process so it gets captured by tee logger
            if 'stdout' in result:
                print("\n" + "=" * 80)
                print("BEST MODEL EVALUATION OUTPUT:")
                print("=" * 80)
                print(result.pop('stdout'))
                print("=" * 80)
            return result
        else:
            print(f"❌ Final evaluation failed")
            # Return a failure result
            return {"success": False, "best_config": search_result.best_config}
    


def main():
    """Main function to run PPO hyperparameter sweep."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for PPO inventory model with intelligent and exhaustive search options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Search Type Details:
  intelligent: Uses Ray Tune with Optuna optimization (requires --num-samples, --max-concurrent)
  exhaustive:  Tests all parameter combinations from grid (requires --num-seeds, ignores num-samples/max-concurrent)

Examples:
  # Intelligent search with 30 trials, up to 2 concurrent
  python %(prog)s --search-type intelligent --num-samples 30 --max-concurrent 2
  
  # Exhaustive grid search with 5 seeds per configuration
  python %(prog)s --search-type exhaustive --num-seeds 5
        """)
    
    parser.add_argument("--config", default=CONFIG_YAML,
                       help=f"Base configuration file (default: {CONFIG_YAML})")
    parser.add_argument("--search-type", choices=["intelligent", "exhaustive"], default="intelligent",
                       help="Search strategy: 'intelligent' (Ray Tune) or 'exhaustive' (grid search) (default: intelligent)")
    parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA,
                       help=f"GAE lambda parameter for bias-variance tradeoff (default: {GAE_LAMBDA})")
    parser.add_argument("--data-dir", default=None,
                       help="Data directory containing train.json, val.json, test.json")
    # Intelligent search options
    intel_group = parser.add_argument_group('Intelligent Search Options (--search-type=intelligent)')
    intel_group.add_argument("--num-samples", type=int, default=50,
                            help="Number of hyperparameter configurations to try (default: 50)")
    intel_group.add_argument("--max-concurrent", type=int, default=5,
                            help="Maximum number of concurrent trials (default: 5)")
    intel_group.add_argument("--top-k", type=int, default=5,
                            help="Number of top configurations to re-evaluate with multiple seeds (default: 5)")
    intel_group.add_argument("--n-startup-trials", type=int, default=10,
                       help="Number of random trials before Bayesian optimization (default: 10)")
    intel_group.add_argument("--stage2-seeds", type=int, default=5,
                            help="Number of seeds per configuration in stage 2 evaluation (default: 5)")
    
    # Exhaustive search options  
    exhaust_group = parser.add_argument_group('Exhaustive Search Options (--search-type=exhaustive)')
    exhaust_group.add_argument("--num-seeds", type=int, default=3,
                              help="Number of random seeds per configuration (default: 3)")
    
    args = parser.parse_args()

    config_stem = Path(args.config).stem
    tee = setup_stdout_log("rl", config_stem, log_folder="logs_hp_search")

    # Set data paths if provided
    if args.data_dir:
        from paths import set_data_paths
        set_data_paths(args.data_dir)
        # Update args to use the new paths
    
    # Validate arguments based on search type
    if args.search_type == "intelligent":
        if not hasattr(args, 'num_samples') or not hasattr(args, 'max_concurrent'):
            parser.error("Intelligent search requires --num-samples and --max-concurrent")
        num_seeds = 1  # Ray Tune handles randomness internally
    else:  # exhaustive
        if not hasattr(args, 'num_seeds'):
            parser.error("Exhaustive search requires --num-seeds")
        num_seeds = args.num_seeds
    
    # Prepare extra command-line arguments to pass to training script
    extra_cmd_args = []
    if args.data_dir:
        extra_cmd_args.extend(["--data-dir", args.data_dir])
    
    # Print configuration
    print(f"Using CONFIG YAML: {args.config}")
    print(f"Using GAE lambda: {args.gae_lambda}\n")
    
    # Run sweep
    sweep = PPOHyperparameterSweep(
        base_config_path=args.config,
        num_seeds=num_seeds,
        gae_lambda=args.gae_lambda,
        extra_cmd_args=extra_cmd_args
    )
    
    # Run hyperparameter sweep with specified search strategy
    search_result = sweep.run_sweep(
        search_type=args.search_type,
        num_samples=args.num_samples,
        max_concurrent=args.max_concurrent,
        top_k=args.top_k if args.search_type == "intelligent" else 0,
        stage2_seeds=args.stage2_seeds if args.search_type == "intelligent" else 0,
        n_startup_trials=args.n_startup_trials
    )
    
    # Evaluate best model
    test_results = sweep.evaluate_best_model(search_result)

    print(f"\nPPO hyperparameter sweep completed!")
    print(f"Search type: {search_result.search_type}")

    # Show stage 2 information if available
    if hasattr(search_result, 'stage2_best_details') and search_result.stage2_best_details:
        details = search_result.stage2_best_details
        print(f"Stage 2 evaluation: Top-{len(search_result.stage2_results)} configs evaluated with {details['num_successful_seeds']} seeds each")
        print(f"Best configuration: {search_result.best_config}")
        print(f"Best avg reward: {details['avg_reward_mean']:.4f} ± {details['std_reward_mean']:.4f}")
        print(f"  (min: {details['min_reward_mean']:.4f}, max: {details['max_reward_mean']:.4f})")
    else:
        print(f"Best configuration: {search_result.best_config}")
        print(f"Best reward: {search_result.best_metric:.4f}")

    tee.close()


if __name__ == "__main__":
    if len(sys.argv) == 1:  # no CLI args → use defaults
        sys.argv += [
            "--search-type", "intelligent",      # Bayesian optimization via Ray Tune + Optuna
            "--num-samples", "50",               # total stage-1 trials
            "--max-concurrent", "5",             # parallel trials
            "--top-k", "5",                      # best configs promoted to stage-2
            "--n-startup-trials", "10",          # random trials before Bayesian kicks in
            "--stage2-seeds", "5",               # seeds per config in stage-2
        ]
    main()