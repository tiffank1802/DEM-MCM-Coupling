---
name: data-io
description: Manage HuggingFace Hub data I/O for DEM MCM experiments
license: MIT
compatibility: opencode
metadata:
  workflow: data
  module: bucket_io
---

## What I do
- Save experiment results (transition matrices, configs, stats, images) to HuggingFace Hub
- Load experiments from HuggingFace Hub buckets
- List and search available experiments
- Auto-detect correct bucket based on particle diameter

## When to use me
Use this when you need to:
- Upload new experimental results
- Load existing results for analysis
- List available experiments by method, diameter, or parameters
- Delete or manage stored experiments
- Debug data storage/retrieval issues

## Key functions

| Function | Description |
|----------|-------------|
| `save_experiment_to_bucket(...)` | Upload P matrix, config, stats, images to HF Hub |
| `load_experiment_from_bucket(folder_name)` | Load experiment by folder name (auto-detects bucket) |
| `list_experiments(prefix)` | List all experiments, optionally filtered by prefix |
| `load_all_experiments()` | Load all experiments across all buckets |
| `get_fs()` | Get HfFileSystem singleton |
| `get_api()` | Get HfApi singleton |

## Bucket structure

| Bucket | Particle diameter | Prefix |
|--------|-------------------|--------|
| `ktongue/DEM_MCM/BIG/` | 0.008 | `BIG/` |
| `ktongue/DEM_MCM/SMALL/` | 0.004 | `SMALL/` |
| `ktongue/DEM_MCM/Experiments/` | All diameters | `Experiments/` |

## How to use

```python
from src.bucket_io import (
    save_experiment_to_bucket,
    load_experiment_from_bucket,
    list_experiments,
    load_all_experiments
)

# List experiments
experiments = list_experiments(prefix="SMALL")

# Load a specific experiment
data = load_experiment_from_bucket("SMALL/cylindrical_nr2_nth4_nz4_tau50")

# Load all experiments
all_data = load_all_experiments()

# Save an experiment
save_experiment_to_bucket(
    folder_name="SMALL/my_experiment",
    P_matrix=P,
    stats=stats,
    config=config,
    partitioner=partitioner,
    images=["image1.png"]
)
```
