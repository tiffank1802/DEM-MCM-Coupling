---
name: run-experiment
description: Run Markov chain experiments with spatial partitioners on DEM particle mixing data
license: MIT
compatibility: opencode
metadata:
  workflow: experiment
  module: run_sweep
---

## What I do
- Run a full Markov chain experiment sweep using `src/run_sweep.py`
- Support all partitioner methods: cartesian, cylindrical, voronoi, quantile, octree, physics, adaptive, multizone
- Handle both SMALL (d=0.004) and BIG (d=0.008) particle diameters
- Upload results to HuggingFace Hub automatically

## When to use me
Use this when you need to:
- Run a new Markov experiment with a specific partitioner
- Sweep over spatial or temporal parameters
- Generate transition matrices (P matrices) from DEM data
- Compare different partitioner configurations

## How to use

### Run with default configs:
```bash
python src/run_sweep.py --method voronoi --diameter 0.004
```

### List available configurations:
```bash
python src/run_sweep.py --method cylindrical --diameter 0.004 --list
```

### Run a specific configuration:
```bash
python src/run_sweep.py --method cartesian --diameter 0.008
```

### Key parameters:
- `--method`: Partitioning method (cartesian, cylindrical, voronoi, quantile, octree, physics, adaptive, multizone, single_cell)
- `--diameter`: Particle diameter (0.004 for SMALL, 0.008 for BIG)
- `--list`: List generated configs without running

### Available config scripts:
```bash
# Physics partitioner tau study
python configs/config_physics_tau.py

# Physics partitioner velocity weight study
python configs/config_physics_velocity_weight.py

# Cylindrical partitioner tau study
python configs/configs_tau_study.py
```

## Project structure
```
src/run_sweep.py          — Main experiment runner
src/partitioners.py       — All partitioner classes
src/bucket_io.py           — HuggingFace Hub I/O
configs/                   — Pre-defined config scripts
ExperimentConfig dataclass — Defines: method, method_kwargs, nlt, tau, step, dt, start_index, particle_diameter
```
