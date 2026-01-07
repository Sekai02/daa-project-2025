"""
Configuration system for experiment batches.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


@dataclass
class BatchConfig:
    """Configuration for a single batch of test instances."""

    batch_name: str
    description: str
    count: int
    generator_config: Dict[str, Any]
    seed_start: int

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'BatchConfig':
        return BatchConfig(
            batch_name=data['batch_name'],
            description=data.get('description', ''),
            count=data['count'],
            generator_config=data['generator_config'],
            seed_start=data['seed_start']
        )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    experiment_name: str
    description: str
    timeout_per_instance: int
    random_seed: int
    batches: List[BatchConfig]
    solutions_to_test: List[str]
    reference_solution: str
    output: Dict[str, Any]

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ExperimentConfig':
        batches = [BatchConfig.from_dict(b) for b in data['batches']]

        return ExperimentConfig(
            experiment_name=data['experiment_name'],
            description=data.get('description', ''),
            timeout_per_instance=data.get('timeout_per_instance', 60),
            random_seed=data.get('random_seed', 42),
            batches=batches,
            solutions_to_test=data['solutions_to_test'],
            reference_solution=data['reference_solution'],
            output=data.get('output', {'save_results': True, 'save_charts': True, 'formats': ['json', 'csv']})
        )

    def validate(self) -> List[str]:
        """Validate configuration for common errors."""
        errors = []

        if not self.batches:
            errors.append("No batches defined in configuration")

        if not self.solutions_to_test:
            errors.append("No solutions specified in solutions_to_test")

        if self.reference_solution not in self.solutions_to_test:
            errors.append(
                f"Reference solution '{self.reference_solution}' "
                f"not found in solutions_to_test list"
            )

        if self.timeout_per_instance <= 0:
            errors.append("timeout_per_instance must be positive")

        for batch in self.batches:
            if batch.count <= 0:
                errors.append(f"Batch '{batch.batch_name}' has non-positive count")

        return errors


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """
    Load and parse experiment configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Parsed ExperimentConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
        ValueError: If config validation fails
    """
    with open(config_path, 'r') as f:
        data = json.load(f)

    config = ExperimentConfig.from_dict(data)

    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    return config
