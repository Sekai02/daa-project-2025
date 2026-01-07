"""
Metrics collection and aggregation for stress testing results.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import statistics


@dataclass
class SolutionMetrics:
    """Aggregated metrics for a single solution across all instances."""

    solution_name: str
    total_instances: int
    successful_runs: int
    timeouts: int
    errors: int
    correct_count: int
    incorrect_count: int

    mean_runtime: float
    median_runtime: float
    min_runtime: float
    max_runtime: float
    std_runtime: float

    mean_memory_mb: float
    peak_memory_mb: float

    correctness_by_mode: Dict[str, float]

    def success_rate(self) -> float:
        """Percentage of runs that completed successfully."""
        if self.total_instances == 0:
            return 0.0
        return 100.0 * self.successful_runs / self.total_instances

    def timeout_rate(self) -> float:
        """Percentage of runs that timed out."""
        if self.total_instances == 0:
            return 0.0
        return 100.0 * self.timeouts / self.total_instances

    def error_rate(self) -> float:
        """Percentage of runs that errored."""
        if self.total_instances == 0:
            return 0.0
        return 100.0 * self.errors / self.total_instances

    def correctness_rate(self) -> float:
        """Percentage of successful runs that matched reference solution."""
        total_decided = self.correct_count + self.incorrect_count
        if total_decided == 0:
            return 0.0
        return 100.0 * self.correct_count / total_decided


class MetricsCollector:
    """Collect and aggregate metrics from experiment results."""

    def __init__(self, experiment_result: Any):
        """
        Initialize metrics collector.

        Args:
            experiment_result: ExperimentResult object containing all batch results
        """
        self.experiment_result = experiment_result

    def compute_solution_metrics(self) -> Dict[str, SolutionMetrics]:
        """
        Compute aggregated metrics for each solution.

        Returns:
            Dictionary mapping solution_name -> SolutionMetrics
        """
        solution_data: Dict[str, List[Dict[str, Any]]] = {}

        for batch_result in self.experiment_result.batch_results:
            batch_mode = batch_result.batch_config.generator_config.get('mode', 'unknown')

            for instance_result in batch_result.instance_results:
                solution_name = instance_result.solution_name

                if solution_name not in solution_data:
                    solution_data[solution_name] = []

                solution_data[solution_name].append({
                    'status': instance_result.status,
                    'runtime': instance_result.runtime_seconds,
                    'memory': instance_result.memory_mb,
                    'is_correct': instance_result.is_correct,
                    'mode': batch_mode
                })

        metrics = {}
        for solution_name, data_points in solution_data.items():
            metrics[solution_name] = self._aggregate_solution_data(solution_name, data_points)

        return metrics

    def _aggregate_solution_data(
        self,
        solution_name: str,
        data_points: List[Dict[str, Any]]
    ) -> SolutionMetrics:
        """Aggregate data points for a single solution."""
        total_instances = len(data_points)
        successful = [d for d in data_points if d['status'] == 'SUCCESS']
        timeouts = [d for d in data_points if d['status'] == 'TIMEOUT']
        errors = [d for d in data_points if d['status'] == 'ERROR']

        correct = [d for d in successful if d.get('is_correct') is True]
        incorrect = [d for d in successful if d.get('is_correct') is False]

        runtimes = [d['runtime'] for d in successful if d['runtime'] > 0]
        memories = [d['memory'] for d in successful if d['memory'] >= 0]

        mean_runtime = statistics.mean(runtimes) if runtimes else 0.0
        median_runtime = statistics.median(runtimes) if runtimes else 0.0
        min_runtime = min(runtimes) if runtimes else 0.0
        max_runtime = max(runtimes) if runtimes else 0.0
        std_runtime = statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0

        mean_memory = statistics.mean(memories) if memories else 0.0
        peak_memory = max(memories) if memories else 0.0

        correctness_by_mode = self._compute_correctness_by_mode(data_points)

        return SolutionMetrics(
            solution_name=solution_name,
            total_instances=total_instances,
            successful_runs=len(successful),
            timeouts=len(timeouts),
            errors=len(errors),
            correct_count=len(correct),
            incorrect_count=len(incorrect),
            mean_runtime=mean_runtime,
            median_runtime=median_runtime,
            min_runtime=min_runtime,
            max_runtime=max_runtime,
            std_runtime=std_runtime,
            mean_memory_mb=mean_memory,
            peak_memory_mb=peak_memory,
            correctness_by_mode=correctness_by_mode
        )

    def _compute_correctness_by_mode(
        self,
        data_points: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute correctness rate for each instance mode."""
        by_mode: Dict[str, List[bool]] = {}

        for dp in data_points:
            if dp['status'] == 'SUCCESS' and dp.get('is_correct') is not None:
                mode = dp.get('mode', 'unknown')
                if mode not in by_mode:
                    by_mode[mode] = []
                by_mode[mode].append(dp['is_correct'])

        correctness_rates = {}
        for mode, results in by_mode.items():
            if results:
                correctness_rates[mode] = 100.0 * sum(results) / len(results)
            else:
                correctness_rates[mode] = 0.0

        return correctness_rates

    def compute_correctness_matrix(self) -> List[List[Any]]:
        """
        Generate correctness matrix for tabular analysis.

        Returns:
            List of rows: [instance_id, solution1_result, solution2_result, ...]
            Result values: 1 (correct), 0 (incorrect), -1 (timeout/error), None (not run)
        """
        matrix = []

        for batch_result in self.experiment_result.batch_results:
            instances: Dict[str, Dict[str, int]] = {}

            for instance_result in batch_result.instance_results:
                instance_id = instance_result.instance_id
                solution_name = instance_result.solution_name

                if instance_id not in instances:
                    instances[instance_id] = {}

                if instance_result.status == 'SUCCESS':
                    if instance_result.is_correct is True:
                        instances[instance_id][solution_name] = 1
                    elif instance_result.is_correct is False:
                        instances[instance_id][solution_name] = 0
                    else:
                        instances[instance_id][solution_name] = None
                else:
                    instances[instance_id][solution_name] = -1

            for instance_id, results in instances.items():
                row = [instance_id]
                for solution_name in self.experiment_result.config.solutions_to_test:
                    row.append(results.get(solution_name, None))
                matrix.append(row)

        return matrix

    def compute_runtime_matrix(self) -> List[List[Any]]:
        """
        Generate runtime matrix for heatmap visualization.

        Returns:
            List of rows: [instance_id, solution1_runtime, solution2_runtime, ...]
            Runtime values in seconds, None for timeout/error
        """
        matrix = []

        for batch_result in self.experiment_result.batch_results:
            instances: Dict[str, Dict[str, Optional[float]]] = {}

            for instance_result in batch_result.instance_results:
                instance_id = instance_result.instance_id
                solution_name = instance_result.solution_name

                if instance_id not in instances:
                    instances[instance_id] = {}

                if instance_result.status == 'SUCCESS':
                    instances[instance_id][solution_name] = instance_result.runtime_seconds
                else:
                    instances[instance_id][solution_name] = None

            for instance_id, runtimes in instances.items():
                row = [instance_id]
                for solution_name in self.experiment_result.config.solutions_to_test:
                    row.append(runtimes.get(solution_name, None))
                matrix.append(row)

        return matrix
