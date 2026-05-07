---
name: visualize-partitions
description: Create 2D and 3D visualizations of spatial partitioner methods
license: MIT
compatibility: opencode
metadata:
  workflow: visualization
  module: visualize_partitioning
---

## What I do
- Create 3D scatter plots showing all partitioner methods side-by-side
- Generate 2D cross-section slice views of partitions
- Render resolution sweeps for individual methods
- Draw grid overlays: cartesian grids, cylindrical sectors, voronoi centroids, octree bounding boxes, adaptive splits

## When to use me
Use this when you need to:
- Visually inspect how different partitioners divide the spatial domain
- Compare partition structures across methods
- Create publication-quality figures of the partitioning schemes
- Debug partitioner behavior

## Key class

`PartitionVisualizer` in `src/visualize_partitioning.py`:
- `show_all_methods()` — 3D grid comparing all partitioners
- `show_all_methods_2d()` — 2D cross-section slices
- `show_resolution_sweep()` — Resolution sweep for a single method

Helper methods for grid overlays:
- `_draw_cartesian_grid()`, `_draw_cylindrical_grid()`
- `_draw_voronoi_centroids()`, `_draw_octree_boxes()`
- `_draw_adaptive_split()`

## How to use

```python
from src.visualize_partitioning import PartitionVisualizer
pv = PartitionVisualizer()

# Show all methods in 3D
pv.show_all_methods()

# Show all methods in 2D slices
pv.show_all_methods_2d()

# Resolution sweep for cylindrical partitioner
pv.show_resolution_sweep(method="cylindrical")
```

Output images are saved to the `images/` directory.

## Available partitioner methods
- **cartesian**: Regular nx * ny * nz grid
- **cylindrical**: Grid in (r, theta, z) with equal_dr or equal_area radial mode
- **voronoi**: K-means clustering (reference MCM method)
- **quantile**: Grid edges at data quantiles
- **octree**: Recursive 8-way spatial subdivision
- **physics**: K-means on [x, y, z, vx*w, vy*w, vz*w]
- **adaptive**: Y-split with coarse top, fine bottom
- **multizone**: N arbitrary zones with different partitioners
- **single_cell**: Single state for entire domain
