//! Research-oriented parametric sweep, session caching, and multi-law
//! approximation framework for `DEM-MCM-Coupling`.
//!
//! # Architecture
//! - [`session`] — Spatial/temporal decoupling (`PreparedMethodSession`,
//!   `ResearchSessionCache`) so `fit`, `states_matrix`, and `S_matrices` are
//!   computed once per spatial partitioning setup and reused across sweeps.
//! - [`laws`] — Competing mathematical law approximation families (`Constant`,
//!   `Linear`, `Quadratic`, `Cubic`, `ExponentialRelax`, `Logarithmic`,
//!   `PowerRelax`) with `RMSE`, `R^2`, `AICc`, and out-of-block validation.
//! - [`study`] — Declarative 1-D / 2-D dynamic study runner (`StudySpec`,
//!   `StudyAxisKind`, `run_study`, `StudyResultBundle`).
//! - [`presets`] — Ready-to-run scientific presets calibrated to synthetic and
//!   real DEM trajectories.

pub mod laws;
pub mod presets;
pub mod session;
pub mod study;

pub use laws::{
    compare_transition_laws, fit_law_family, LawCandidateFit, LawFamily, LawSelectionCriterion,
    TransitionLawComparison,
};
pub use presets::{
    build_preset_spec, catalog_study_presets, study_presets_json, StudyPresetDescriptor,
};
pub use session::{
    build_prepared_session, frobenius_relative_error, BedFrameBasis, PreparedMethodSession,
    ResearchSessionCache, SessionKey, TemporalEvalOutput, TemporalEvalParams,
};
pub use study::{
    run_study, AxisValue, ReferencePolicy, StudyAxisKind, StudyAxisSpec, StudyChainMode,
    StudyCurveSeries, StudyLawPointResult, StudyPointMetrics, StudyResultBundle, StudySpec,
};
