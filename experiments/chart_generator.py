"""
Chart generation for stress testing results using matplotlib.
"""

import os
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class ChartGenerator:
    """Generate all experiment visualizations."""

    def __init__(self, experiment_result: Any, metrics: Dict[str, Any]):
        """
        Initialize chart generator.

        Args:
            experiment_result: ExperimentResult object
            metrics: Dictionary of solution_name -> SolutionMetrics
        """
        self.experiment_result = experiment_result
        self.metrics = metrics

    def generate_all_charts(self, output_dir: str, formats: List[str] = ['png']) -> None:
        """
        Generate all charts and save to output_dir.

        Args:
            output_dir: Directory to save charts
            formats: List of file formats (e.g., ['png', 'pdf'])
        """
        os.makedirs(output_dir, exist_ok=True)

        self.generate_runtime_comparison(output_dir, formats)
        self.generate_correctness_by_mode(output_dir, formats)
        self.generate_scalability_analysis(output_dir, formats)
        self.generate_memory_usage(output_dir, formats)

    def generate_runtime_comparison(self, output_dir: str, formats: List[str]) -> None:
        """
        Bar chart: Solutions vs Median Runtime.
        Shows median runtime with error bars (25th/75th percentile).
        """
        solutions = sorted(self.metrics.keys())
        medians = []
        q1_errors = []
        q3_errors = []

        for solution_name in solutions:
            m = self.metrics[solution_name]
            medians.append(m.median_runtime)

            runtimes = self._get_runtimes_for_solution(solution_name)
            if len(runtimes) >= 4:
                sorted_runtimes = sorted(runtimes)
                q1 = np.percentile(sorted_runtimes, 25)
                q3 = np.percentile(sorted_runtimes, 75)
                q1_errors.append(m.median_runtime - q1)
                q3_errors.append(q3 - m.median_runtime)
            else:
                q1_errors.append(0)
                q3_errors.append(0)

        fig, ax = plt.subplots(figsize=(12, 6))

        x_pos = np.arange(len(solutions))
        colors = plt.cm.Set3(np.linspace(0, 1, len(solutions)))

        ax.bar(x_pos, medians, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax.errorbar(x_pos, medians, yerr=[q1_errors, q3_errors], fmt='none',
                    ecolor='black', capsize=5, capthick=2)

        ax.set_xlabel('Solution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Runtime Comparison Across Solutions (Median with IQR)',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(solutions, rotation=45, ha='right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        plt.tight_layout()

        for fmt in formats:
            plt.savefig(os.path.join(output_dir, f'runtime_comparison.{fmt}'), dpi=300)
        plt.close()

    def generate_correctness_by_mode(self, output_dir: str, formats: List[str]) -> None:
        """
        Grouped bar chart: Solutions vs Correctness Rate %.
        Groups: random, two_layer, bottleneck modes.
        """
        solutions = sorted(self.metrics.keys())
        modes = self._get_unique_modes()

        if not modes:
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        x_pos = np.arange(len(solutions))
        width = 0.8 / len(modes)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

        for mode_idx, mode in enumerate(modes):
            correctness_rates = []
            for solution_name in solutions:
                m = self.metrics[solution_name]
                rate = m.correctness_by_mode.get(mode, 0.0)
                correctness_rates.append(rate)

            offset = (mode_idx - len(modes) / 2) * width + width / 2
            ax.bar(x_pos + offset, correctness_rates, width,
                   label=mode, color=colors[mode_idx % len(colors)],
                   alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Solution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Correctness Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Correctness Rate by Instance Type', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(solutions, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend(title='Instance Mode', loc='lower right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()

        for fmt in formats:
            plt.savefig(os.path.join(output_dir, f'correctness_by_mode.{fmt}'), dpi=300)
        plt.close()

    def generate_scalability_analysis(self, output_dir: str, formats: List[str]) -> None:
        """
        Scatter plot: Problem Size (n × m) vs Runtime.
        One series per solution with trend lines.
        """
        solutions = sorted(self.metrics.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(solutions)))

        fig, ax = plt.subplots(figsize=(12, 8))

        for solution_idx, solution_name in enumerate(solutions):
            sizes = []
            runtimes = []

            for batch_result in self.experiment_result.batch_results:
                n = batch_result.batch_config.generator_config.get('n', 0)
                m = batch_result.batch_config.generator_config.get('m_edges', 0)
                problem_size = n * m

                for ir in batch_result.instance_results:
                    if ir.solution_name == solution_name and ir.status == 'SUCCESS':
                        sizes.append(problem_size)
                        runtimes.append(ir.runtime_seconds)

            if not sizes:
                continue

            ax.scatter(sizes, runtimes, alpha=0.6, s=50,
                      color=colors[solution_idx], label=solution_name,
                      edgecolors='black', linewidth=0.5)

            if len(sizes) >= 2:
                try:
                    log_sizes = np.log10(np.array(sizes) + 1)
                    log_runtimes = np.log10(np.array(runtimes) + 1e-6)
                    z = np.polyfit(log_sizes, log_runtimes, 1)
                    p = np.poly1d(z)

                    size_range = np.linspace(min(sizes), max(sizes), 100)
                    log_size_range = np.log10(size_range + 1)
                    trend_runtimes = 10 ** p(log_size_range)

                    ax.plot(size_range, trend_runtimes, '--',
                           color=colors[solution_idx], alpha=0.7, linewidth=2)
                except:
                    pass

        ax.set_xlabel('Problem Size (n × m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax.set_title('Scalability Analysis: Runtime vs Problem Size',
                     fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        plt.tight_layout()

        for fmt in formats:
            plt.savefig(os.path.join(output_dir, f'scalability_analysis.{fmt}'), dpi=300)
        plt.close()

    def generate_memory_usage(self, output_dir: str, formats: List[str]) -> None:
        """
        Box plot: Solutions vs Memory Usage (MB).
        Shows distribution of memory consumption.
        """
        solutions = sorted(self.metrics.keys())
        memory_data = []

        for solution_name in solutions:
            memories = []
            for batch_result in self.experiment_result.batch_results:
                for ir in batch_result.instance_results:
                    if ir.solution_name == solution_name and ir.status == 'SUCCESS':
                        if ir.memory_mb > 0:
                            memories.append(ir.memory_mb)

            memory_data.append(memories if memories else [0])

        fig, ax = plt.subplots(figsize=(12, 6))

        bp = ax.boxplot(memory_data, labels=solutions, patch_artist=True,
                        showmeans=True, meanline=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(color='green', linewidth=2, linestyle='--'),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))

        ax.set_xlabel('Solution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
        ax.set_title('Memory Usage Distribution', fontsize=14, fontweight='bold')
        ax.set_xticklabels(solutions, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        red_line = plt.Line2D([0], [0], color='red', linewidth=2, label='Median')
        green_line = plt.Line2D([0], [0], color='green', linewidth=2,
                                linestyle='--', label='Mean')
        ax.legend(handles=[red_line, green_line], loc='upper right')

        plt.tight_layout()

        for fmt in formats:
            plt.savefig(os.path.join(output_dir, f'memory_usage.{fmt}'), dpi=300)
        plt.close()

    def _get_runtimes_for_solution(self, solution_name: str) -> List[float]:
        """Get all successful runtimes for a solution."""
        runtimes = []
        for batch_result in self.experiment_result.batch_results:
            for ir in batch_result.instance_results:
                if ir.solution_name == solution_name and ir.status == 'SUCCESS':
                    runtimes.append(ir.runtime_seconds)
        return runtimes

    def _get_unique_modes(self) -> List[str]:
        """Get list of unique instance modes from batches."""
        modes = set()
        for batch_result in self.experiment_result.batch_results:
            mode = batch_result.batch_config.generator_config.get('mode', 'unknown')
            modes.add(mode)
        return sorted(modes)
