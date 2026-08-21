"""Physical validation of the experiments stored in the Hugging Face bucket.

Every experiment of the bucket is loaded and checked against the physics of
granular mixing:

* transition probabilities are non-negative;
* visited rows of the transition matrix are stochastic (the legacy
  column-stochastic convention of older experiments is auto-detected and
  transposed — see :func:`postprocessing.metrics.standardize_transition_matrix`);
* the particle mass is conserved during a Markov propagation;
* the matrix admits a valid stationary distribution;
* the concentration RSD stays in ``[0, 1]`` and decays over time (mixing
  cannot un-mix).

Usage::

    python -m postprocessing.validate_bucket --method voronoi --max 5
    python -m postprocessing.validate_bucket --folder voronoi_125cells_...
    python -m postprocessing.validate_bucket --synthetic   # offline demo

Requires a Hugging Face token with read access to ``ktongue/DEM_MCM``
(``huggingface-cli login`` or the ``HF_TOKEN`` environment variable).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from postprocessing.metrics import validate_experiment


def _synthetic_experiments() -> dict[str, dict]:
    """Build synthetic experiments in the two stored conventions.

    * ``"new_convention"`` — row-stochastic matrix (current format);
    * ``"legacy_convention"`` — the same matrix stored transposed
      (column-stochastic, as produced by the old pipeline).

    Both must validate identically thanks to the automatic standardisation.
    """
    rng = np.random.RandomState(42)
    n_states = 12
    n_timesteps = 120

    # Row-stochastic matrix with strong diagonal (particles mostly stay).
    P = rng.rand(n_states, n_states) * 0.02
    P[np.arange(n_states), np.arange(n_states)] += 1.0
    P /= P.sum(axis=1, keepdims=True)

    def _build(P_stored: np.ndarray) -> dict:
        S_small = np.zeros((n_timesteps, n_states))
        S_large = np.zeros((n_timesteps, n_states))
        # Initially segregated: the small species occupies the left half.
        S_small[0, : n_states // 2] = 20.0
        S_large[0, n_states // 2 :] = 20.0
        for t in range(1, n_timesteps):
            S_small[t] = S_small[t - 1] @ P
            S_large[t] = S_large[t - 1] @ P

        return {
            "config": {"tau": 50, "step": 100, "nlt": 2, "start_index": 250},
            "stats": {"species_list": ["small", "large"]},
            "species": {
                "small": {
                    "P": P_stored,
                    "S_matrix": S_small,
                    "times": np.arange(250, 250 + n_timesteps),
                },
                "large": {
                    "P": P_stored,
                    "S_matrix": S_large,
                    "times": np.arange(250, 250 + n_timesteps),
                },
            },
            "inhomogeneous": False,
            "inhomogeneous_metadata": None,
        }

    return {
        "synthetic_new_convention": _build(P),
        "synthetic_legacy_convention": _build(P.T),
    }


def validate_from_bucket(
    method: str | None = None,
    folder: str | None = None,
    max_experiments: int | None = None,
) -> int:
    """Validate experiments of the bucket; return the number of failures.

    Args:
        method: Optional method filter (folder-name prefix).
        folder: Optional exact folder name.
        max_experiments: Maximum number of experiments to check.

    Returns:
        Number of experiments whose report failed.
    """
    from dem_mcm_coupling.bucket_io import (
        BUCKET_BASE,
        list_experiments,
        load_experiment_from_bucket,
    )

    try:
        experiments = [folder] if folder else list_experiments()
    except Exception as exc:  # network/credentials error
        print(
            "❌ Could not reach the Hugging Face bucket.\n"
            f"   Error: {exc}\n"
            "   - Check your network access to huggingface.co;\n"
            "   - authenticate with `huggingface-cli login` (or set HF_TOKEN)\n"
            "     using an account with read access to ktongue/DEM_MCM;\n"
            "   - or run the offline demo with --synthetic."
        )
        return 1
    if method:
        experiments = [e for e in experiments if e.startswith(method)]
    if max_experiments is not None:
        experiments = experiments[:max_experiments]

    if not experiments:
        print("❌ No experiment found (check the method/folder filter).")
        return 1

    failures = 0
    print(f"Validating {len(experiments)} experiment(s) from {BUCKET_BASE}\n")
    for name in experiments:
        try:
            exp = load_experiment_from_bucket(name)
        except Exception as exc:  # report and continue
            print(f"❌ {name} — could not be loaded: {exc}\n")
            failures += 1
            continue
        report = validate_experiment(name, exp)
        print(report)
        if not report.passed:
            failures += 1
        print()

    print(f"Summary: {len(experiments) - failures}/{len(experiments)} passed")
    return failures


def main() -> None:
    """Command-line entry point of the bucket validation."""
    parser = argparse.ArgumentParser(
        description="Physical validation of the experiments of the DEM/MCM bucket"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Restrict to folders starting with this method name (e.g. voronoi)",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Validate a single experiment folder",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Maximum number of experiments to validate",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Offline demo: validate synthetic experiments in both stored "
        "conventions (no bucket access required)",
    )
    args = parser.parse_args()

    if args.synthetic:
        print("Offline demo — synthetic experiments (no bucket access).\n")
        failures = 0
        for name, exp in _synthetic_experiments().items():
            report = validate_experiment(name, exp)
            print(report)
            if not report.passed:
                failures += 1
            print()
        sys.exit(1 if failures else 0)

    sys.exit(validate_from_bucket(args.method, args.folder, args.max))


if __name__ == "__main__":
    main()
