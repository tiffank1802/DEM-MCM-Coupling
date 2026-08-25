"""Physics of the Markov/DEM mixing — metrics and propagation.

This module is the **single source of truth** for the mixing physics used by
the post-processing pipeline:

* the transition-matrix convention (row-stochastic, ``phi_next = phi @ P``);
* the propagation of homogeneous and inhomogeneous chains;
* the segregation metrics: RSD (global and concentration-based), entropy,
  intensity of segregation (Danckwerts), mixing times;
* the physical validation of an experiment.

Transition-matrix convention
----------------------------
``P[i, j]`` is the probability to jump from state ``i`` to state ``j``:

* rows are stochastic — ``P.sum(axis=1) == 1`` for every visited state;
* a state vector evolves by **right multiplication** — ``phi_next = phi @ P``;
* the stationary distribution is the dominant **left** eigenvector of ``P``.

Compatibility with legacy data
------------------------------
Experiments stored before this convention was unified follow the
**transposed** convention (column-stochastic matrices, evolved as
``phi_next = P.T @ phi``). :func:`standardize_transition_matrix` detects the
stored convention automatically and returns a row-stochastic matrix in every
case, so that old bucket data keep their physical meaning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CONVENTION STANDARDISATION
# =============================================================================


def detect_convention(P: np.ndarray, rtol: float = 1e-3) -> str:
    """Detect the stochasticity convention of a transition matrix.

    Args:
        P: Transition matrix of shape ``(n_states, n_states)``.
        rtol: Relative tolerance of the sums comparison.

    Returns:
        ``"row"`` when the rows sum to one, ``"column"`` when the columns
        sum to one, ``"none"`` otherwise.
    """
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    active_rows = row_sums > 0
    active_cols = col_sums > 0

    rows_ok = bool(active_rows.any()) and np.allclose(
        row_sums[active_rows], 1.0, rtol=rtol
    )
    cols_ok = bool(active_cols.any()) and np.allclose(
        col_sums[active_cols], 1.0, rtol=rtol
    )

    if rows_ok and not cols_ok:
        return "row"
    if cols_ok and not rows_ok:
        return "column"
    if rows_ok and cols_ok:
        return "row"  # doubly stochastic → keep as-is
    return "none"


def standardize_transition_matrix(
    P: np.ndarray, warn: bool = True
) -> tuple[np.ndarray, bool]:
    """Return a row-stochastic version of a transition matrix.

    Legacy column-stochastic matrices are transposed (their entries keep the
    meaning ``P[i, j] = P(i → j)``); matrices whose rows do not sum to one
    are renormalised row-wise. Unvisited (all-zero) rows stay zero.

    Args:
        P: Transition matrix of shape ``(n_states, n_states)``.
        warn: When ``True``, log a warning each time the stored convention
            differs from the row convention.

    Returns:
        Tuple ``(P_row, transposed)`` where ``P_row`` is row-stochastic and
        ``transposed`` tells whether the matrix had to be transposed.
    """
    P = np.asarray(P, dtype=np.float64)
    convention = detect_convention(P)

    if convention == "column":
        if warn:
            logger.warning(
                "Legacy column-stochastic matrix detected — transposing to "
                "the row convention (P[i, j] = P(i → j))."
            )
        return P.T.copy(), True

    if convention == "none":
        row_sums = P.sum(axis=1)
        P_out = P.copy()
        safe = np.where(row_sums > 0, row_sums, 1.0)
        P_out = P_out / safe[:, np.newaxis]
        if warn:
            logger.warning(
                "Transition matrix is not stochastic — rows renormalised "
                "(mean row sum was %.3f).",
                float(np.mean(row_sums)),
            )
        return P_out, False

    return P.copy(), False


# =============================================================================
# CLEANING AND PROPAGATION
# =============================================================================


def clean_transition_matrix(
    P: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Clean a transition matrix: standardise and renormalise.

    Drops the unvisited states and renormalises the remaining rows.

    Convention (row-stochastic): ``P[i, j] = P(i → j)``; a state vector
    evolves as ``S_next = S @ P``.

    Args:
        P: Raw transition matrix of shape ``(n_states, n_states)`` (any
            legacy convention is auto-detected and standardised).
        threshold: Rows with a total outgoing mass below this threshold are
            considered unvisited and deactivated.

    Returns:
        Tuple ``(P_clean, activated)`` — ``P_clean`` has zero rows for the
        deactivated states and renormalised rows elsewhere; ``activated`` is
        the boolean mask of the kept states.
    """
    P_clean, _transposed = standardize_transition_matrix(P)
    row_sums = P_clean.sum(axis=1)
    activated = row_sums >= threshold
    P_clean[~activated, :] = 0.0
    safe = row_sums.copy()
    safe[~activated] = 1.0
    P_clean = P_clean / safe[:, np.newaxis]
    return P_clean, activated


def propagate_markov(
    S0: np.ndarray,
    P: np.ndarray,
    times: np.ndarray,
    start_idx: int,
    tau: int,
    activated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a state vector with a row-stochastic transition matrix.

    ``S_next = S @ P`` — the total particle count is conserved at every step
    when ``P`` is row-stochastic (up to the machine precision).

    Args:
        S0: Initial state vector, shape ``(n_states,)``.
        P: Transition matrix, shape ``(n_states, n_states)`` (any stored
            convention is standardised first).
        times: DEM timestep indices.
        start_idx: First timestep used.
        tau: Markov step (in timesteps).
        activated: Boolean mask of the activated states; deactivated states
            are zeroed in the initial vector.

    Returns:
        Tuple ``(trajectory, times_markov)`` — trajectory of shape
        ``(n_steps + 1, n_states)``.
    """
    P_std, _ = standardize_transition_matrix(P)

    row_start = np.searchsorted(times, start_idx)
    times_full = times[row_start:]
    markov_idx = np.arange(0, len(times_full), tau)
    times_markov = times_full[markov_idx]

    S = np.asarray(S0, dtype=float).copy()
    S[~activated] = 0.0
    traj = [S.copy()]
    for _ in range(1, len(markov_idx)):
        S = S @ P_std
        traj.append(S.copy())
    return np.array(traj), times_markov


def propagate_markov_inhomogeneous(
    S0: np.ndarray,
    P_blocks: np.ndarray,
    times: np.ndarray,
    start_idx: int,
    tau: int,
    activated: np.ndarray,
    step: int | None = None,
    nlt: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a state vector with time-varying transition matrices.

    Each matrix ``P_k`` is applied on the real temporal span of its block:
    block ``k`` covers
    ``[start_idx + k*(step+tau), start_idx + (k+1)*(step+tau))``.

    Convention (row-stochastic): ``S_next = S @ P_k``.

    Args:
        S0: Initial state vector, shape ``(n_states,)``.
        P_blocks: Transition matrices, shape ``(n_blocks, n_states,
            n_states)``.
        times: DEM timestep indices.
        start_idx: First timestep used.
        tau: Markov step (in timesteps).
        activated: Boolean mask of the activated states.
        step: Optional distance between consecutive block starts (falls back
            to a uniform division when ``None``).
        nlt: Optional number of blocks.

    Returns:
        Tuple ``(trajectory, times_markov)`` — trajectory of shape
        ``(n_steps + 1, n_states)``.
    """
    # Standardise every block to the row convention.
    P_blocks = np.asarray(P_blocks, dtype=np.float64)
    P_std = np.empty_like(P_blocks)
    for k in range(len(P_blocks)):
        P_std[k], _ = standardize_transition_matrix(P_blocks[k], warn=k == 0)
    P_blocks = P_std

    row_start = np.searchsorted(times, start_idx)
    times_full = times[row_start:]
    markov_idx = np.arange(0, len(times_full), tau)
    times_markov = times_full[markov_idx]
    n_steps = len(markov_idx) - 1
    n_blocks = len(P_blocks)

    S = np.asarray(S0, dtype=float).copy()
    S[~activated] = 0.0
    traj = [S.copy()]

    if step is not None and nlt is not None:
        # Real temporal structure of the blocks.
        block_duration = step + tau
        for t in range(1, len(markov_idx)):
            time_curr = times_markov[t - 1]
            block_idx = int((time_curr - start_idx) / block_duration)
            block_idx = min(max(block_idx, 0), n_blocks - 1)
            S = S @ P_blocks[block_idx]
            traj.append(S.copy())
    else:
        # Fallback: uniform division of the trajectory over the blocks.
        block_size = max(1, n_steps // n_blocks) if n_blocks > 0 else 1
        for t in range(1, len(markov_idx)):
            block_idx = min((t - 1) // block_size, n_blocks - 1)
            S = S @ P_blocks[block_idx]
            traj.append(S.copy())

    return np.array(traj), times_markov


# =============================================================================
# SEGREGATION METRICS
# =============================================================================


def rsd_from_S(S: np.ndarray, activated: np.ndarray) -> np.ndarray:
    """Global RSD of the particle distribution per timestep.

    ``RSD(t) = std(S[t, active]) / mean(S[t, active])`` over the activated
    cells; 0 when the mean is zero (empty slice).

    Args:
        S: State matrix of shape ``(n_timesteps, n_states)``.
        activated: Boolean mask of the activated cells.

    Returns:
        RSD per timestep, shape ``(n_timesteps,)``.
    """
    S = np.asarray(S, dtype=float)
    S_a = S[:, activated]
    mean = S_a.mean(axis=1)
    std = S_a.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)


def concentration_from_S(S_small: np.ndarray, S_large: np.ndarray) -> np.ndarray:
    """Local concentration of the small species per cell and timestep.

    ``C(t, i) = S_small / (S_small + S_large)``; 0 where the cell is empty.

    Args:
        S_small: Small-species state matrix ``(n_timesteps, n_states)``.
        S_large: Large-species state matrix ``(n_timesteps, n_states)``.

    Returns:
        Concentration matrix of the same shape, values in ``[0, 1]``.
    """
    S_small = np.asarray(S_small, dtype=float)
    S_large = np.asarray(S_large, dtype=float)
    total = S_small + S_large
    return np.where(total > 0, S_small / np.where(total > 0, total, 1.0), 0.0)


def rsd_concentration(
    S_small: np.ndarray,
    S_large: np.ndarray,
    act_s: np.ndarray,
    act_l: np.ndarray,
) -> np.ndarray:
    """RSD of the small-species concentration across the active cells.

    The concentration RSD is the reference mixing index: it starts from a
    high value (segregated state) and decays towards 0 as the mixture
    homogenises.

    Args:
        S_small: Small-species state matrix ``(n_timesteps, n_states)``.
        S_large: Large-species state matrix ``(n_timesteps, n_states)``.
        act_s: Activated cells of the small species.
        act_l: Activated cells of the large species.

    Returns:
        Concentration RSD per timestep, shape ``(n_timesteps,)``.
    """
    act = np.asarray(act_s) & np.asarray(act_l)
    C = concentration_from_S(S_small[:, act], S_large[:, act])
    mean = C.mean(axis=1)
    std = C.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)


def entropy_from_S(S: np.ndarray, activated: np.ndarray) -> np.ndarray:
    """Shannon entropy of the particle distribution over the active cells.

    ``H(t) = -Σ p_i(t) log p_i(t)`` with ``p_i`` the fraction of particles in
    cell ``i``. A perfectly uniform distribution reaches ``log(n_active)``.

    Args:
        S: State matrix of shape ``(n_timesteps, n_states)``.
        activated: Boolean mask of the activated cells.

    Returns:
        Entropy per timestep, shape ``(n_timesteps,)``.
    """
    S = np.asarray(S, dtype=float)
    S_a = S[:, activated]
    totals = S_a.sum(axis=1, keepdims=True)
    p = S_a / np.where(totals > 0, totals, 1.0)
    safe = np.where(p > 0, p, 1.0)  # 0 * log(0) → 0
    return -np.sum(p * np.log(safe), axis=1)


def entropy_concentration(
    S_small: np.ndarray,
    S_large: np.ndarray,
    act_s: np.ndarray,
    act_l: np.ndarray,
    normalized: bool = True,
) -> np.ndarray:
    """Binary mixing entropy of the local concentration.

    ``H(t) = -Σ_i [C_i log C_i + (1-C_i) log(1-C_i)]`` summed over the
    active cells. When ``normalized``, the result is divided by
    ``n_active * log(2)`` — the maximum attainable value — so it lies in
    ``[0, 1]`` (1 = perfectly mixed, 0 = fully segregated).

    Args:
        S_small: Small-species state matrix ``(n_timesteps, n_states)``.
        S_large: Large-species state matrix ``(n_timesteps, n_states)``.
        act_s: Activated cells of the small species.
        act_l: Activated cells of the large species.
        normalized: When ``True`` (default), divide by the theoretical
            maximum ``n_active * log(2)``.

    Returns:
        Mixing entropy per timestep, shape ``(n_timesteps,)``.
    """
    act = np.asarray(act_s) & np.asarray(act_l)
    C = concentration_from_S(S_small[:, act], S_large[:, act])
    n_active = int(act.sum())

    H = np.zeros(len(C))
    for t in range(len(C)):
        Ct = C[t]
        valid = (Ct > 0) & (Ct < 1)
        if valid.any():
            Cv = Ct[valid]
            H[t] = -np.sum(Cv * np.log(Cv) + (1 - Cv) * np.log(1 - Cv))

    if normalized and n_active > 0:
        H = H / (n_active * np.log(2.0))
    return H


def intensity_of_segregation(
    S_small: np.ndarray,
    S_large: np.ndarray,
    act_s: np.ndarray,
    act_l: np.ndarray,
) -> np.ndarray:
    """Intensity of segregation (Danckwerts).

    ``I(t) = Var(C) / (C̄ (1 - C̄))`` over the active cells: 1 for a fully
    segregated mixture, 0 for a perfect mixture.

    Args:
        S_small: Small-species state matrix ``(n_timesteps, n_states)``.
        S_large: Large-species state matrix ``(n_timesteps, n_states)``.
        act_s: Activated cells of the small species.
        act_l: Activated cells of the large species.

    Returns:
        Intensity of segregation per timestep, shape ``(n_timesteps,)``.
    """
    act = np.asarray(act_s) & np.asarray(act_l)
    C = concentration_from_S(S_small[:, act], S_large[:, act])
    mean = C.mean(axis=1)
    var = C.var(axis=1)
    denom = mean * (1.0 - mean)
    return np.where(denom > 0, var / np.where(denom > 0, denom, 1.0), 0.0)


def mixing_times(
    rsd: np.ndarray,
    times: np.ndarray,
    fractions: tuple[float, ...] = (0.5, 0.1),
) -> dict[float, float | None]:
    """Mixing times of an RSD decay curve.

    The mixing time at fraction ``f`` is the first time where the RSD falls
    below ``f`` times its initial value (``f = 0.5`` → ``t50``, ``f = 0.1``
    → ``t90``).

    Args:
        rsd: RSD curve, shape ``(n_timesteps,)``.
        times: Time axis (raw timesteps, no unit conversion).
        fractions: Fractions of the initial RSD.

    Returns:
        Mapping ``fraction -> time in the same unit as ``times``` (``None``
        when never reached).
    """
    rsd = np.asarray(rsd, dtype=float)
    times = np.asarray(times, dtype=float)
    rsd_0 = rsd[0] if rsd[0] > 0 else 1.0

    out: dict[float, float | None] = {}
    for fraction in fractions:
        hit = np.where(rsd < fraction * rsd_0)[0]
        out[fraction] = float(times[hit[0]]) if len(hit) > 0 else None
    return out


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Stationary distribution ``π`` of a row-stochastic transition matrix.

    ``π`` solves ``π P = π`` — it is the dominant **left** eigenvector of
    ``P`` (computed here as the dominant right eigenvector of ``P.T``).

    Args:
        P: Row-stochastic transition matrix.

    Returns:
        Normalised stationary distribution, shape ``(n_states,)``.
    """
    P_std, _ = standardize_transition_matrix(P)
    eigvals, eigvecs = np.linalg.eig(P_std.T)
    dominant = int(np.argmax(np.abs(eigvals)))
    pi = np.abs(np.real(eigvecs[:, dominant]))
    pi = pi / pi.sum()
    return pi


def normalize_by_particle_count(S: np.ndarray) -> np.ndarray:
    """Normalise a state matrix by the particle count of each timestep.

    Each row is divided by its own total number of particles, so the result
    holds **fractions of particles** per cell (values in ``[0, 1]``). This is
    the homogenisation required before comparing DEM and Markov predictions
    coming from simulations with different particle populations.

    Args:
        S: State matrix of shape ``(n_timesteps, n_states)``.

    Returns:
        The normalised matrix of the same shape; empty rows stay zero.
    """
    S = np.asarray(S, dtype=float)
    totals = S.sum(axis=1, keepdims=True)
    return np.divide(S, totals, out=np.zeros_like(S), where=totals > 0)


@dataclass
class ProbabilityLawFit:
    """Interpolation law fitted on a transition probability ``p_ij(t)``.

    Attributes:
        degree: Degree of the selected polynomial law (0 = constant,
            1 = linear, 2 = quadratic, ...).
        coefficients: Polynomial coefficients ordered from the highest power
            to the constant term (``numpy.polyfit`` convention).
        rmse: Root-mean-square error of the fit on the data points.
        r2: Coefficient of determination of the fit.
        law_label: Human-readable equation of the fitted law.
    """

    degree: int
    coefficients: np.ndarray
    rmse: float
    r2: float
    law_label: str

    def predict(self, x: np.ndarray | float) -> np.ndarray:
        """Evaluate the fitted polynomial at the given abscissae.

        Args:
            x: Abscissae (times) at which to evaluate the law.

        Returns:
            The interpolated values, same shape as ``x``.
        """
        return np.polyval(self.coefficients, np.asarray(x, dtype=float))


def _polynomial_law_label(coefficients: np.ndarray, degree: int) -> str:
    """Format a polynomial law as a readable ``p = a t^k + ... + b`` string.

    Args:
        coefficients: Polynomial coefficients, highest power first.
        degree: Polynomial degree.

    Returns:
        The equation string, e.g. ``"p = -0.0012 t + 0.052"``.
    """
    terms = []
    for k, coef in zip(range(degree, -1, -1), coefficients):
        if abs(coef) < 1e-12:
            continue
        sign = "+" if (coef > 0 and terms) else ""
        if k == 0:
            terms.append(f"{sign}{coef:.4g}")
        elif k == 1:
            terms.append(f"{sign}{coef:.4g} t")
        else:
            terms.append(f"{sign}{coef:.4g} t^{k}")
    return "p = " + (" ".join(terms) if terms else "0")


def fit_probability_evolution(
    x: np.ndarray,
    y: np.ndarray,
    max_degree: int = 2,
) -> ProbabilityLawFit:
    """Fit the best polynomial law on a transition probability ``p_ij(t)``.

    Candidates are tested from degree 0 (constant) up to ``max_degree``
    (linear, quadratic, cubic, ...). The selection is parsimonious: among the
    candidates, the **lowest degree whose RMSE stays within 5% of the best
    RMSE** is kept, so a linear law is preferred whenever it is essentially
    as good as a quadratic one.

    Args:
        x: Abscissae (e.g. block times or timestep indices), shape ``(n,)``.
        y: Transition probabilities, shape ``(n,)``.
        max_degree: Maximum polynomial degree to consider (default 2).

    Returns:
        The fitted :class:`ProbabilityLawFit`.

    Raises:
        ValueError: If ``max_degree`` is negative or there are not enough
            data points to fit a single candidate.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if max_degree < 0:
        raise ValueError(f"max_degree must be >= 0, got {max_degree}")

    degrees = [d for d in range(max_degree + 1) if d <= len(x) - 1]
    if not degrees:
        raise ValueError(
            f"Not enough points ({len(x)}) to fit a polynomial of degree 0"
        )

    fits: dict[int, tuple[np.ndarray, float]] = {}
    best_rmse = float("inf")
    for degree in degrees:
        coefficients = np.polyfit(x, y, degree)
        y_hat = np.polyval(coefficients, x)
        rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
        fits[degree] = (coefficients, rmse)
        best_rmse = min(best_rmse, rmse)

    # Parsimonious selection: lowest degree within 5% of the best RMSE.
    chosen_degree = next(d for d in degrees if fits[d][1] <= best_rmse * 1.05 + 1e-12)

    coefficients = fits[chosen_degree][0]
    y_hat = np.polyval(coefficients, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else (1.0 if ss_res <= 1e-12 else 0.0)

    return ProbabilityLawFit(
        degree=chosen_degree,
        coefficients=coefficients,
        rmse=fits[chosen_degree][1],
        r2=r2,
        law_label=_polynomial_law_label(coefficients, chosen_degree),
    )


# =============================================================================
# PHYSICAL VALIDATION
# =============================================================================


@dataclass
class ValidationCheck:
    """Result of one physical check on an experiment.

    Attributes:
        name: Human-readable name of the check.
        passed: Whether the check passed.
        detail: Short quantitative detail (values, tolerances).
    """

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    """Full physical validation report of one experiment.

    Attributes:
        folder: Experiment name.
        checks: List of :class:`ValidationCheck`.
        passed: ``True`` when every check passed.
    """

    folder: str
    checks: list[ValidationCheck] = field(default_factory=list)
    passed: bool = False

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        """Append a check to the report."""
        self.checks.append(ValidationCheck(name=name, passed=passed, detail=detail))

    def __str__(self) -> str:
        lines = [f"Validation of '{self.folder}'"]
        for check in self.checks:
            marker = "✅" if check.passed else "❌"
            lines.append(f"  {marker} {check.name} — {check.detail}")
        lines.append(
            f"  → {'ALL CHECKS PASSED' if self.passed else 'SOME CHECKS FAILED'}"
        )
        return "\n".join(lines)


def validate_experiment(
    folder_name: str,
    exp: dict[str, Any],
    tau: int | None = None,
    rtol: float = 1e-3,
) -> ValidationReport:
    """Validate the physical coherence of a loaded experiment.

    Checks, per species:

    1. **transition probabilities** — non-negative entries;
    2. **stochasticity** — visited rows sum to one (after legacy-convention
       standardisation);
    3. **diagonal dominance** — particles mostly stay in place
       (``mean(diag(P))`` reasonably high, as expected at small ``tau``);
    4. **mass conservation** — the total particle count is preserved during
       a propagation of the (standardised) matrix;
    5. **stationarity** — the dominant eigenvalue is 1 and the stationary
       distribution is positive;
    6. **RSD bounds** — the concentration RSD stays in the physical range
       ``[0, 1]`` when both species are present.

    Args:
        folder_name: Experiment folder name.
        exp: Loaded experiment dictionary (same format as
            :func:`dem_mcm_coupling.bucket_io.load_experiment_from_bucket`).
        tau: Optional Markov step; read from ``exp["config"]`` when ``None``.
        rtol: Relative tolerance of the stochasticity check.

    Returns:
        The :class:`ValidationReport`.
    """
    report = ValidationReport(folder=folder_name)

    species = exp.get("species") or {}
    if not species:
        report.add("species present", False, "no per-species data found")
        report.passed = all(c.passed for c in report.checks)
        return report
    report.add("species present", True, f"{len(species)} species: {', '.join(species)}")

    config = exp.get("config") or {}
    if tau is None:
        tau = int(config.get("tau", 50))

    for sp, data in species.items():
        P_raw = data.get("P") if data.get("P") is not None else None
        if P_raw is None and data.get("P_blocks") is not None:
            P_raw = data["P_blocks"][0]
        if P_raw is None:
            report.add(f"{sp}: matrix present", False, "no P or P_blocks[0]")
            continue

        P_raw = np.asarray(P_raw, dtype=float)

        # 1. Non-negativity.
        negatives = int((P_raw < 0).sum())
        report.add(
            f"{sp}: non-negative entries",
            negatives == 0,
            f"{negatives} negative entries / {P_raw.size}",
        )

        # 2. Stochasticity after standardisation.
        P_std, transposed = standardize_transition_matrix(P_raw, warn=False)
        if transposed:
            report.add(
                f"{sp}: convention",
                True,
                "legacy column-stochastic matrix auto-transposed",
            )
        row_sums = P_std.sum(axis=1)
        active = row_sums > 0
        max_dev = float(np.max(np.abs(row_sums[active] - 1.0))) if active.any() else 1.0
        report.add(
            f"{sp}: row-stochastic after standardisation",
            bool(active.any() and max_dev <= rtol),
            f"max |row sum - 1| = {max_dev:.2e} over {int(active.sum())} visited rows",
        )

        # 3. Diagonal dominance (particles mostly stay at small tau).
        diag_mean = float(np.diag(P_std).mean())
        report.add(
            f"{sp}: diagonal dominance",
            diag_mean >= 0.1,
            f"mean(P(i → i)) = {diag_mean:.3f}",
        )

        # 4. Mass conservation during propagation.
        n_states = P_std.shape[0]
        S0 = np.ones(n_states)
        S0[~(row_sums > 0)] = 0.0
        S = S0.copy()
        conserved = True
        for _ in range(50):
            S = S @ P_std
            if not np.isclose(S.sum(), S0.sum(), rtol=1e-9):
                conserved = False
                break
        report.add(
            f"{sp}: mass conservation over 50 steps",
            conserved,
            f"total = {S0.sum():.1f} → {S.sum():.3f}",
        )

        # 5. Stationarity.
        eigvals = np.linalg.eigvals(P_std.T)
        dominant = float(np.max(np.abs(eigvals)))
        pi = stationary_distribution(P_std)
        report.add(
            f"{sp}: stationarity",
            bool(np.isclose(dominant, 1.0, rtol=1e-6)) and bool((pi >= 0).all()),
            f"|λ|max = {dominant:.6f}, π ∈ [0, 1]",
        )

    # 6. Concentration RSD physical bounds (cross-species).
    species_keys = list(species)
    if len(species_keys) >= 2:
        sp_a, sp_b = species_keys[0], species_keys[1]
        da, db = species[sp_a], species[sp_b]
        S_a = np.asarray(da.get("S_matrix", []))
        S_b = np.asarray(db.get("S_matrix", []))
        if S_a.size and S_b.size and S_a.shape == S_b.shape:
            n = min(len(S_a), len(S_b))
            act = np.ones(S_a.shape[1], dtype=bool)
            rsd = rsd_concentration(S_a[:n], S_b[:n], act, act)
            report.add(
                "concentration RSD in [0, 1]",
                bool((rsd >= -1e-12).all() and (rsd <= 1 + 1e-6).all()),
                f"min = {rsd.min():.4f}, max = {rsd.max():.4f}, final = {rsd[-1]:.4f}",
            )
            # Physical decay: the final RSD should not exceed the initial one
            # by more than a small tolerance (mixing cannot un-mix).
            decay_ok = rsd[-1] <= rsd[0] * 1.05 + 1e-9
            report.add(
                "concentration RSD decays (mixing)",
                bool(decay_ok),
                f"initial = {rsd[0]:.4f} → final = {rsd[-1]:.4f}",
            )

    report.passed = all(check.passed for check in report.checks)
    return report
