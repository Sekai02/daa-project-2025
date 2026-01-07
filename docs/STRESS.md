# Stress Testing Framework

Comprehensive stress testing framework for power grid solutions.

## Overview

This framework allows you to:
- Generate test instances with configurable parameters
- Validate instances before testing
- Run all solutions with timeout protection
- Track performance metrics (runtime, memory, correctness)
- Generate visualizations comparing solutions
- Export results to JSON and CSV

## Quick Start

### Basic Usage

```bash
# Run the small correctness test
python3 experiments/runner.py --config experiments/config/small_instances.json

# Run the full stress test suite
python3 experiments/runner.py --config experiments/config/stress_test_suite.json

# Run without generating charts
python3 experiments/runner.py --config experiments/config/small_instances.json --no-charts
```

### Requirements

**Core Framework:**
- Python 3.7+ (only standard library)

**Optional (for charts):**
- matplotlib (for visualization generation)

Install matplotlib:
```bash
pip install matplotlib
```

## Configuration

Experiments are configured using JSON files in `experiments/config/`.

### Example Configuration

```json
{
  "experiment_name": "my_experiment",
  "description": "Test description",
  "timeout_per_instance": 60,
  "random_seed": 42,
  "batches": [
    {
      "batch_name": "small_random",
      "description": "Small random instances",
      "count": 10,
      "generator_config": {
        "n": 10,
        "num_plants": 2,
        "num_consumers": 3,
        "m_edges": 30,
        "mode": "random",
        "budget_ratio": 0.25
      },
      "seed_start": 1000
    }
  ],
  "solutions_to_test": [
    "brute_force_solution",
    "greedy_solution",
    "pure_greedy_solution"
  ],
  "reference_solution": "brute_force_solution",
  "output": {
    "save_results": true,
    "save_charts": true,
    "formats": ["json", "csv"]
  }
}
```

### Configuration Fields

- **experiment_name**: Name for the experiment (used in output files)
- **description**: Human-readable description
- **timeout_per_instance**: Timeout in seconds for each solution per instance
- **random_seed**: Global random seed
- **batches**: List of test batches (see Batch Configuration below)
- **solutions_to_test**: List of solution function names
- **reference_solution**: Solution to use for correctness comparison (must be in solutions_to_test)
- **output**: Output configuration

### Batch Configuration

Each batch can have different parameters:

- **batch_name**: Unique name for the batch
- **description**: Human-readable description
- **count**: Number of instances to generate
- **generator_config**: Parameters for instance generation (see Generator Config below)
- **seed_start**: Starting seed for instance generation

### Generator Config

Parameters for instance generation:

- **n**: Number of nodes
- **num_plants**: Number of power plants
- **num_consumers**: Number of consumers
- **m_edges**: Number of candidate edges
- **mode**: Instance mode - "random", "two_layer", or "bottleneck"
- **budget_ratio**: Budget as fraction of total edge costs (e.g., 0.25 = 25%)

See `utils/generators/generator.py` for full list of available parameters.

## Output

### Results Directory Structure

```
experiments/
├── results/
│   ├── {experiment_name}_{timestamp}.json
│   └── {experiment_name}_{timestamp}.csv
└── graphs/
    ├── runtime_comparison.png
    ├── correctness_by_mode.png
    ├── scalability_analysis.png
    └── memory_usage.png
```

### JSON Results

Detailed results including:
- Per-instance, per-solution metrics
- Runtime, memory usage, objective values
- Correctness flags (compared to reference solution)
- Error messages for failed runs

### CSV Results

Tabular format with columns:
- instance_id, batch, mode
- solution_name, status
- runtime_seconds, memory_mb
- objective_value, is_correct

### Charts (requires matplotlib)

1. **Runtime Comparison**: Bar chart comparing median runtimes across solutions
2. **Correctness by Mode**: Grouped bar chart showing correctness rates by instance type
3. **Scalability Analysis**: Scatter plot of problem size vs runtime
4. **Memory Usage**: Box plot of memory distribution

## Adding New Solutions

To test a new solution:

1. Add solution file to `solutions/` directory
2. Implement function with standard signature:
   ```python
   def my_solution(n, edges, plants, consumers, kappa, u, g, d, B):
       # Your implementation
       return (selected_edges, objective_value)
   ```
3. Add solution name to `solutions_to_test` in config
4. Run experiment - framework auto-discovers it

## Instance Validation

All generated instances are validated before testing:

### Validation Checks

1. **Basic constraints**:
   - List lengths match
   - All values non-negative
   - No overlap between plants and consumers
   - Edge endpoints in valid range

2. **Graph connectivity**:
   - At least one plant can reach at least one consumer (BFS-based check)

3. **Budget feasibility**:
   - At least one edge is affordable within budget (warning if not)

Invalid instances are skipped and counted in validation_failures.

## Performance Metrics

For each solution on each instance:

- **Runtime**: Execution time in seconds
- **Memory**: Peak memory usage in MB
- **Correctness**: Whether objective value matches reference solution
- **Status**: SUCCESS, TIMEOUT, or ERROR

## Advanced Usage

### Custom Output Directory

```bash
python3 experiments/runner.py --config my_config.json --output-dir /path/to/output
```

### Verbose Mode

```bash
python3 experiments/runner.py --config my_config.json --verbose
```

### Skip Chart Generation

```bash
python3 experiments/runner.py --config my_config.json --no-charts
```

## Troubleshooting

### Solutions Timing Out

- Increase `timeout_per_instance` in config
- Reduce instance size (n, m_edges)
- Exclude slow solutions like brute_force for large instances

### Validation Failures

- Check generator_config parameters
- Ensure plants + consumers <= n
- Verify mode is one of: "random", "two_layer", "bottleneck"

### Import Errors

- Ensure solution files exist in `solutions/` directory
- Verify function names match file names (e.g., `greedy_solution.py` contains `greedy_solution()`)

## Example Workflows

### Quick Correctness Check

Use `small_instances.json` to quickly verify solutions work correctly:

```bash
python3 experiments/runner.py --config experiments/config/small_instances.json
```

### Full Performance Analysis

Use `stress_test_suite.json` for comprehensive testing across sizes and modes:

```bash
python3 experiments/runner.py --config experiments/config/stress_test_suite.json
```

### Compare Specific Solutions

Create custom config with only desired solutions:

```json
{
  "solutions_to_test": ["greedy_solution", "simulated_annealing_solution"],
  "reference_solution": "greedy_solution",
  ...
}
```

## Framework Architecture

```
experiments/
├── runner.py                 # Main orchestrator
├── metrics_collector.py      # Statistical aggregation
├── chart_generator.py        # Visualization generation
└── config/                   # Experiment configurations
    └── __init__.py          # Config loading

utils/
├── generators/
│   └── generator.py          # Instance generation
└── validators/
    └── validator.py          # Instance validation
```

## Contact & Issues

For questions or issues with the stress testing framework, refer to the main project documentation or contact the development team.
