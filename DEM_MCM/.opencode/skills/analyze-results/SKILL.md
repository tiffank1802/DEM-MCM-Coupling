---
name: analyze-results
description: Analyze and visualize Markov chain experimental results, compare DEM vs Markov mixing metrics
license: MIT
compatibility: opencode
metadata:
  workflow: analysis
  module: analyze_results
---

## What I do
- Load experimental results from HuggingFace Hub (BIG/SMALL/Experiments buckets)
- Compute RSD (Relative Standard Deviation) mixing curves
- Compare DEM ground truth vs Markov predictions
- Generate comparison plots between partitioner methods
- Compute mixing metrics: t50, t90, spectral gap, eigenvalue analysis

## When to use me
Use this when you need to:
- Analyze results from completed experiments
- Compare DEM vs Markov mixing kinetics
- Generate comparison plots between partitioner methods
- Compute mixing times and quality metrics

## Key classes and functions

| Function | Description |
|----------|-------------|
| `MarkovAnalyzer.load_all()` | Load all experiments from all buckets |
| `MarkovAnalyzer.load_method(method)` | Load all experiments for a specific method |
| `MarkovAnalyzer.load_single(folder_name)` | Load a single experiment |
| `MarkovAnalyzer.simulate_mixing()` | Evolve concentration via P matrix |
| `MarkovAnalyzer.compute_rsd()` | Markov-predicted RSD curve |
| `MarkovAnalyzer.compute_dem_rsd()` | DEM-calculated RSD (ground truth) |
| `MarkovAnalyzer.compare_dem_vs_markov()` | Full DEM vs Markov comparison |
| `MarkovAnalyzer.compare_methods()` | Compare multiple methods side-by-side |

## How to use

### Basic analysis:
```python
from src.analyze_results import MarkovAnalyzer
ma = MarkovAnalyzer()
ma.load_all()
ma.compare_dem_vs_markov(method="cylindrical", method_kwargs={})
```

### Run comparison scripts:
```bash
# Tau comparison (cylindrical, SMALL)
python runs/run_tau_comparison.py

# Tau comparison (physics, both sizes)
python runs/run_tau_comparison_physics.py --diameter 0.004

# Tau comparison (voronoi, SMALL)
python runs/run_tau_comparison_voronoi.py

# Velocity weight comparison
python runs/run_velocity_weight_comparison.py --diameter 0.004
```

## Metrics reference
- **RSD**: Relative Standard Deviation of concentration (primary mixing metric)
- **t50/t90**: Steps to reach 50%/90% reduction in RSD
- **Spectral gap**: λ1 - λ2, measures mixing rate
- **P(stay)**: Diagonal mean of P matrix
- **Entropy**: Shannon entropy of state occupancy distribution
