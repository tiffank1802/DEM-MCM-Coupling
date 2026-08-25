"""Post-processing tools for the DEM/MCM coupling experiments.

Submodules:

* :mod:`postprocessing.metrics` — physics of the mixing (propagation,
  transition-matrix convention, segregation metrics, validation);
* :mod:`postprocessing.style` — global colour code and scientific matplotlib
  style;
* :mod:`postprocessing.figures` — publication-oriented annotated figures;
* :mod:`postprocessing.postprocess` — homogeneous post-processing pipeline
  and CLI;
* :mod:`postprocessing.postprocess_inhomogeneous` — inhomogeneous (P_blocks)
  pipeline and CLI;
* :mod:`postprocessing.validate_bucket` — physical validation of the bucket
  experiments;
* :mod:`postprocessing.tools` — one-off maintenance scripts.

Note: this package is intentionally **not** shipped in the
``dem-mcm-coupling`` PyPI distribution; it is part of the repository tooling.
"""

from postprocessing.metrics import (
    clean_transition_matrix,
    concentration_from_S,
    detect_convention,
    entropy_concentration,
    entropy_from_S,
    intensity_of_segregation,
    mixing_times,
    propagate_markov,
    propagate_markov_inhomogeneous,
    rsd_concentration,
    rsd_from_S,
    standardize_transition_matrix,
    stationary_distribution,
    validate_experiment,
)

__all__ = [
    "clean_transition_matrix",
    "concentration_from_S",
    "detect_convention",
    "entropy_concentration",
    "entropy_from_S",
    "intensity_of_segregation",
    "mixing_times",
    "propagate_markov",
    "propagate_markov_inhomogeneous",
    "rsd_concentration",
    "rsd_from_S",
    "standardize_transition_matrix",
    "stationary_distribution",
    "validate_experiment",
]
