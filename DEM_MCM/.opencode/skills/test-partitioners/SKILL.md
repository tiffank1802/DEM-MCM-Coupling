---
name: test-partitioners
description: Run tests for spatial partitioners and Markov matrix properties
license: MIT
compatibility: opencode
metadata:
  workflow: testing
  module: tests
---

## What I do
- Run unit tests for spatial partitioner state assignments
- Validate transition matrix properties (column sums = 1)
- Test temporal refinement logic
- Test bucket I/O integration
- Run adaptive partitioner tests with real data

## When to use me
Use this when you need to:
- Verify partitioner correctness after changes
- Validate that transition matrices are stochastic
- Test new partitioner implementations
- Run the full test suite or specific test files

## Test files

| Test file | What it tests |
|-----------|---------------|
| `tests/partitionnement/cartesian_tests.py` | Cartesian state numbering: ix + iy*nx + iz*nx*ny |
| `tests/partitionnement/cylindrical_tests.py` | Cylindrical state numbering: ir + itheta*nr + iz*nr*ntheta |
| `tests/partitionnement/quantile_tests.py` | Quantile state numbering |
| `tests/test_adaptive.py` | AdaptiveZPartitioner creation, fitting, diagnostics, visualization |
| `tests/test_bucket_integration.py` | Bucket I/O functions with real data |
| `tests/dem_state_matrix/construction.py` | DEM snapshot loading |
| `tests/P_matrix/sum_colonne_test.py` | Matrix column sum = 1 (stochastic property) |
| `tests/raffinagetemporel/test_raffinagetemporel.py` | Temporal refinement logic |

## How to use

### Run individual test files:
```bash
python tests/partitionnement/cartesian_tests.py
python tests/partitionnement/cylindrical_tests.py
python tests/partitionnement/quantile_tests.py
python tests/test_adaptive.py
python tests/test_bucket_integration.py
python tests/P_matrix/sum_colonne_test.py
python tests/raffinagetemporel/test_raffinagetemporel.py
python tests/dem_state_matrix/construction.py
```

### Run with pytest:
```bash
python -m pytest tests/ -v
python -m pytest tests/partitionnement/ -v
python -m pytest tests/test_adaptive.py -v
```

### Quick matrix verification:
```bash
python verif_matrix_sum.py
```
