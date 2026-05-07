---
name: run-tau-study
description: Run tau (Markov timestep) sensitivity studies for cylindrical, physics, and voronoi partitioners
license: MIT
compatibility: opencode
metadata:
  workflow: experiment
  module: tau_study
---

## What I do
- Run tau sensitivity studies to analyze how the Markov timestep affects mixing kinetics
- Support multiple partitioner types: cylindrical, physics, voronoi
- Handle both SMALL (d=0.004) and BIG (d=0.008) particle diameters
- Compare results across different tau values and partitioner methods

## When to use me
Use this when you need to:
- Study how the Markov timestep (tau) affects prediction quality
- Compare tau sensitivity across partitioner methods
- Determine optimal tau for a given partitioner
- Validate that the Markov model is robust to timestep choice

## Tau study scripts

| Script | Partitioner | Diameters |
|--------|-------------|-----------|
| `configs/configs_tau_study.py` | Cylindrical (nr=2, ntheta=4, nz=4) | SMALL (0.004) |
| `configs/config_physics_tau.py` | Physics (n_cells=30, vw=0.5) | SMALL + BIG |
| `runs/run_tau_comparison.py` | Cylindrical comparison plots | SMALL |
| `runs/run_tau_comparison_BIG.py` | Cylindrical comparison plots | BIG |
| `runs/run_tau_comparison_physics.py` | Physics comparison plots | SMALL + BIG |
| `runs/run_tau_comparison_voronoi.py` | Voronoi comparison plots | SMALL |
| `runs/run_tau_comparison_voronoi_BIG.py` | Voronoi comparison plots | BIG |

## How to use

### Run experiments:
```bash
# Cylindrical tau study (SMALL)
python configs/configs_tau_study.py

# Physics tau study (both diameters)
python configs/config_physics_tau.py
```

### Generate comparison plots:
```bash
# Cylindrical comparison
python runs/run_tau_comparison.py --diameter 0.004

# Physics comparison
python runs/run_tau_comparison_physics.py --diameter 0.004

# Voronoi comparison
python runs/run_tau_comparison_voronoi.py --diameter 0.004
```

## Typical tau sweep values
```python
tau_values = [10, 25, 50, 100, 150, 200, 300, 500, 750, 1000]
```
