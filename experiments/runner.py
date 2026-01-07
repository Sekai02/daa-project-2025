"""
Main stress testing framework runner.

Orchestrates instance generation, validation, solution execution,
metrics collection, and result serialization.
"""

import sys
import os
import time
import json
import csv
import traceback
import importlib.util
import multiprocessing
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Callable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.generators.generator import generate_instance, GeneratorConfig, Instance
from utils.validators import validate_instance, ValidationResult
from experiments.config import load_experiment_config, ExperimentConfig, BatchConfig
from experiments.metrics_collector import MetricsCollector


@dataclass
class InstanceResult:
    """Result for a single solution on a single instance."""

    solution_name: str
    instance_id: str
    selected_edges: Optional[List[int]]
    objective_value: Optional[int]
    runtime_seconds: float
    memory_mb: float
    status: str
    is_correct: Optional[bool]
    error_message: Optional[str]


@dataclass
class BatchResult:
    """Results for all solutions on all instances in a batch."""

    batch_name: str
    batch_config: BatchConfig
    instance_results: List[InstanceResult]


@dataclass
class ExperimentResult:
    """Complete experiment results."""

    experiment_name: str
    timestamp: str
    config: ExperimentConfig
    batch_results: List[BatchResult]
    validation_failures: int


def _timeout_wrapper(queue: multiprocessing.Queue, solution_name: str, solutions_dir: str, func_args: Tuple):
    """Wrapper to run function and collect metrics in subprocess."""
    try:
        import tracemalloc
        import importlib.util

        # Handle solution name to file name mapping
        SOLUTION_FILE_MAP = {
            'grasp_solution': 'grasp_swap_local_search_solution',
        }
        file_name = SOLUTION_FILE_MAP.get(solution_name, solution_name)
        file_path = os.path.join(solutions_dir, f'{file_name}.py')
        spec = importlib.util.spec_from_file_location(file_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module spec for {solution_name}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, solution_name)

        tracemalloc.start()

        start = time.perf_counter()
        result = func(*func_args)
        end = time.perf_counter()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        queue.put({
            'status': 'SUCCESS',
            'result': result,
            'runtime': end - start,
            'memory_mb': peak / (1024 * 1024)
        })
    except Exception as e:
        queue.put({
            'status': 'ERROR',
            'error': str(e),
            'traceback': traceback.format_exc()
        })


def _run_with_timeout(solution_name: str, solutions_dir: str, args: Tuple, timeout: int) -> Tuple[
    Optional[List[int]],
    Optional[int],
    float,
    float,
    str,
    Optional[str]
]:
    """
    Run solution function in a separate process with timeout.

    Args:
        solution_name: Name of the solution function/module
        solutions_dir: Directory containing solution files
        args: Tuple of arguments to pass to solution
        timeout: Timeout in seconds

    Returns:
        Tuple of (edges, objective, runtime, memory_mb, status, error_msg)
        status: "SUCCESS", "TIMEOUT", or "ERROR"
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_timeout_wrapper, args=(queue, solution_name, solutions_dir, args))

    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return None, None, float(timeout), 0.0, "TIMEOUT", None

    if not queue.empty():
        result = queue.get()
        if result['status'] == 'SUCCESS':
            edges, obj = result['result']
            return edges, obj, result['runtime'], result['memory_mb'], 'SUCCESS', None
        else:
            return None, None, 0.0, 0.0, 'ERROR', result.get('traceback', result['error'])

    return None, None, 0.0, 0.0, 'ERROR', 'Unknown error - no result from subprocess'


class StressTestRunner:
    """Main stress testing framework orchestrator."""

    SOLUTION_FILE_MAP = {
        'grasp_solution': 'grasp_swap_local_search_solution',
    }

    def __init__(self, config: ExperimentConfig):
        """
        Initialize stress test runner.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.solutions: Dict[str, Callable] = {}
        self.reference_solution: Optional[Callable] = None
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    def load_solutions(self) -> None:
        """Dynamically import all solution functions from solutions/ directory."""
        solutions_dir = os.path.join(self.project_root, 'solutions')

        print(f"\n[1/3] Loading {len(self.config.solutions_to_test)} solutions...")

        for solution_name in self.config.solutions_to_test:
            file_name = self.SOLUTION_FILE_MAP.get(solution_name, solution_name)
            file_path = os.path.join(solutions_dir, f'{file_name}.py')

            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Solution file not found: {file_path}\n"
                    f"Expected solution: {solution_name}"
                )

            spec = importlib.util.spec_from_file_location(solution_name, file_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load module spec for {solution_name}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, solution_name):
                raise AttributeError(
                    f"Module {solution_name} does not have function '{solution_name}'\n"
                    f"Available attributes: {dir(module)}"
                )

            func = getattr(module, solution_name)
            self.solutions[solution_name] = func
            print(f"  ✓ {solution_name}")

        self.reference_solution = self.solutions.get(self.config.reference_solution)
        if self.reference_solution is None:
            raise ValueError(f"Reference solution '{self.config.reference_solution}' not loaded")

    def run_experiment(self) -> ExperimentResult:
        """
        Run the entire experiment.

        Returns:
            ExperimentResult containing all batch results
        """
        print("\n" + "=" * 80)
        print(" Stress Testing Framework - Experiment Runner")
        print("=" * 80)
        print(f"\nExperiment: {self.config.experiment_name}")
        print(f"Description: {self.config.description}")
        print(f"Timeout per instance: {self.config.timeout_per_instance}s")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.load_solutions()

        print(f"\n[2/3] Running {len(self.config.batches)} batches...")

        batch_results = []
        total_validation_failures = 0

        for batch_idx, batch_config in enumerate(self.config.batches, 1):
            print(f"\nBatch {batch_idx}/{len(self.config.batches)}: {batch_config.batch_name} ({batch_config.count} instances)")
            print(f"  Description: {batch_config.description}")

            batch_result, validation_failures = self.run_batch(batch_config)
            batch_results.append(batch_result)
            total_validation_failures += validation_failures

            successful = sum(1 for ir in batch_result.instance_results if ir.status == 'SUCCESS')
            timeouts = sum(1 for ir in batch_result.instance_results if ir.status == 'TIMEOUT')
            errors = sum(1 for ir in batch_result.instance_results if ir.status == 'ERROR')
            total_runs = len(batch_result.instance_results)

            print(f"  Batch complete. Success: {successful}/{total_runs}, Timeouts: {timeouts}/{total_runs}, Errors: {errors}/{total_runs}")

        experiment_result = ExperimentResult(
            experiment_name=self.config.experiment_name,
            timestamp=timestamp,
            config=self.config,
            batch_results=batch_results,
            validation_failures=total_validation_failures
        )

        return experiment_result

    def run_batch(self, batch_config: BatchConfig) -> Tuple[BatchResult, int]:
        """
        Run all solutions on all instances in a batch.

        Args:
            batch_config: Batch configuration

        Returns:
            Tuple of (BatchResult, validation_failures_count)
        """
        instance_results: List[InstanceResult] = []
        validation_failures = 0

        for instance_idx in range(batch_config.count):
            seed = batch_config.seed_start + instance_idx
            instance_id = f"{batch_config.batch_name}_{instance_idx}_seed{seed}"

            print(f"  Instance {instance_idx + 1}/{batch_config.count} (seed={seed}): ", end='', flush=True)

            gen_config = GeneratorConfig(**batch_config.generator_config)
            instance = generate_instance(gen_config, seed=seed)

            validation_result = validate_instance(instance)
            if not validation_result.is_valid:
                print(f"VALIDATION FAILED - skipping")
                validation_failures += 1
                for error in validation_result.critical_errors:
                    print(f"    {error}")
                continue

            args = instance.to_solver_args()
            reference_result = None
            solutions_dir = os.path.join(self.project_root, 'solutions')

            for solution_idx, solution_name in enumerate(self.config.solutions_to_test):
                edges, obj, runtime, memory, status, error_msg = _run_with_timeout(
                    solution_name,
                    solutions_dir,
                    args,
                    self.config.timeout_per_instance
                )

                if solution_name == self.config.reference_solution:
                    reference_result = (edges, obj, status)
                    is_correct = None
                else:
                    is_correct = None
                    if status == 'SUCCESS' and reference_result is not None:
                        ref_edges, ref_obj, ref_status = reference_result
                        if ref_status == 'SUCCESS':
                            is_correct = (obj == ref_obj)

                instance_results.append(InstanceResult(
                    solution_name=solution_name,
                    instance_id=instance_id,
                    selected_edges=edges,
                    objective_value=obj,
                    runtime_seconds=runtime,
                    memory_mb=memory,
                    status=status,
                    is_correct=is_correct,
                    error_message=error_msg
                ))

                if status == 'SUCCESS':
                    print('✓', end='', flush=True)
                elif status == 'TIMEOUT':
                    print('T', end='', flush=True)
                else:
                    print('E', end='', flush=True)

            print(f" [{len(self.config.solutions_to_test)}/{len(self.config.solutions_to_test)} solutions]")

        return BatchResult(
            batch_name=batch_config.batch_name,
            batch_config=batch_config,
            instance_results=instance_results
        ), validation_failures

    def save_results(self, experiment_result: ExperimentResult, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save experiment results to JSON and CSV files.

        Args:
            experiment_result: The experiment results to save
            output_dir: Optional output directory override

        Returns:
            Dictionary mapping format -> file path
        """
        if output_dir is None:
            output_dir = os.path.join(self.project_root, 'experiments', 'results')

        os.makedirs(output_dir, exist_ok=True)

        timestamp = experiment_result.timestamp
        name = experiment_result.experiment_name

        saved_files = {}

        if 'json' in self.config.output.get('formats', []):
            json_path = os.path.join(output_dir, f'{name}_{timestamp}.json')
            self._save_json(experiment_result, json_path)
            saved_files['json'] = json_path
            print(f"  ✓ Saved JSON: {json_path}")

        if 'csv' in self.config.output.get('formats', []):
            csv_path = os.path.join(output_dir, f'{name}_{timestamp}.csv')
            self._save_csv(experiment_result, csv_path)
            saved_files['csv'] = csv_path
            print(f"  ✓ Saved CSV: {csv_path}")

        return saved_files

    def _save_json(self, experiment_result: ExperimentResult, file_path: str) -> None:
        """Save results to JSON format."""
        data = {
            'experiment_name': experiment_result.experiment_name,
            'timestamp': experiment_result.timestamp,
            'config': {
                'experiment_name': experiment_result.config.experiment_name,
                'timeout_per_instance': experiment_result.config.timeout_per_instance,
                'solutions_to_test': experiment_result.config.solutions_to_test,
                'reference_solution': experiment_result.config.reference_solution,
            },
            'summary': {
                'total_batches': len(experiment_result.batch_results),
                'total_instances': sum(len(set(ir.instance_id for ir in br.instance_results))
                                       for br in experiment_result.batch_results),
                'validation_failures': experiment_result.validation_failures,
            },
            'batches': []
        }

        for batch_result in experiment_result.batch_results:
            instances_data: Dict[str, Dict] = {}

            for ir in batch_result.instance_results:
                if ir.instance_id not in instances_data:
                    instances_data[ir.instance_id] = {
                        'instance_id': ir.instance_id,
                        'solutions': {}
                    }

                instances_data[ir.instance_id]['solutions'][ir.solution_name] = {
                    'status': ir.status,
                    'runtime_seconds': ir.runtime_seconds,
                    'memory_mb': ir.memory_mb,
                    'objective_value': ir.objective_value,
                    'is_correct': ir.is_correct,
                    'error_message': ir.error_message
                }

            data['batches'].append({
                'batch_name': batch_result.batch_name,
                'instances': list(instances_data.values())
            })

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_csv(self, experiment_result: ExperimentResult, file_path: str) -> None:
        """Save results to CSV format."""
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'instance_id', 'batch', 'mode', 'solution_name', 'status',
                'runtime_seconds', 'memory_mb', 'objective_value', 'is_correct'
            ])

            for batch_result in experiment_result.batch_results:
                batch_name = batch_result.batch_name
                mode = batch_result.batch_config.generator_config.get('mode', 'unknown')

                for ir in batch_result.instance_results:
                    writer.writerow([
                        ir.instance_id,
                        batch_name,
                        mode,
                        ir.solution_name,
                        ir.status,
                        f"{ir.runtime_seconds:.6f}",
                        f"{ir.memory_mb:.2f}",
                        ir.objective_value if ir.objective_value is not None else '',
                        ir.is_correct if ir.is_correct is not None else ''
                    ])


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Stress Testing Framework for Power Grid Solutions"
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to experiment configuration (JSON)'
    )
    parser.add_argument(
        '--output-dir',
        help='Override output directory for results'
    )
    parser.add_argument(
        '--no-charts',
        action='store_true',
        help='Skip chart generation'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    runner = StressTestRunner(config)

    try:
        experiment_result = runner.run_experiment()
    except Exception as e:
        print(f"\nError running experiment: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1

    print("\n[3/3] Generating results and charts...")

    saved_files = runner.save_results(experiment_result, args.output_dir)

    if not args.no_charts and config.output.get('save_charts', True):
        try:
            from experiments.chart_generator import ChartGenerator

            graphs_dir = os.path.join(runner.project_root, 'experiments', 'graphs')
            os.makedirs(graphs_dir, exist_ok=True)

            metrics_collector = MetricsCollector(experiment_result)
            metrics = metrics_collector.compute_solution_metrics()

            chart_gen = ChartGenerator(experiment_result, metrics)
            chart_gen.generate_all_charts(graphs_dir, formats=['png'])

            print(f"  ✓ Generated charts: {graphs_dir}")
        except Exception as e:
            print(f"  ✗ Chart generation failed: {e}")
            if args.verbose:
                traceback.print_exc()

    print("\n" + "=" * 80)
    print(" Summary")
    print("=" * 80)

    total_instances = sum(
        len(set(ir.instance_id for ir in br.instance_results))
        for br in experiment_result.batch_results
    )
    total_runs = sum(len(br.instance_results) for br in experiment_result.batch_results)

    successful = sum(1 for br in experiment_result.batch_results
                     for ir in br.instance_results if ir.status == 'SUCCESS')
    timeouts = sum(1 for br in experiment_result.batch_results
                   for ir in br.instance_results if ir.status == 'TIMEOUT')
    errors = sum(1 for br in experiment_result.batch_results
                 for ir in br.instance_results if ir.status == 'ERROR')

    print(f"\nTotal instances: {total_instances}")
    print(f"Total solution runs: {total_runs}")
    print(f"Successful: {successful} ({100 * successful / total_runs:.1f}%)")
    print(f"Timeouts: {timeouts} ({100 * timeouts / total_runs:.1f}%)")
    print(f"Errors: {errors} ({100 * errors / total_runs:.1f}%)")
    print(f"Validation failures: {experiment_result.validation_failures}")

    if 'json' in saved_files:
        print(f"\nResults saved to: {saved_files.get('json', 'N/A')}")
    if 'csv' in saved_files:
        print(f"CSV saved to: {saved_files.get('csv', 'N/A')}")

    print("=" * 80)

    return 0


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    sys.exit(main())
