//! Visualization & CAD-style workbench core shared between `egui` (`gui-egui`)
//! and the interactive browser UI served by `dem-mcm-gui`.

pub mod camera3d;
pub mod matrix_evolution;
pub mod mesh3d;
pub mod metrics_eval;
pub mod simulation;
pub mod state;
pub mod web;

pub use camera3d::{Camera3D, CameraPreset, ProjectedPoint};
pub use matrix_evolution::{compute_matrix_evolution, MatrixEvolutionData, TransitionLawItem};
pub use mesh3d::{
    extract_mesh_3d, infer_domain, DrumCylinder3D, Mesh3DData, MeshCell3D, SplitPlane3D,
};
pub use metrics_eval::{ChainPredictionMetrics, MethodBenchmarkRow};
pub use simulation::{
    generate_synthetic_drum_frames, run_gui_comparison, run_method_benchmark, BlockSummary,
    ComparisonBundle, GuiDatasetCache, SnapshotFrame3D,
};
pub use state::{
    catalog_methods, ActiveSpecies, CellColorMode, CollapsiblePanelsState, EvolutionScope,
    EvolutionViewMode, MatrixEvolutionFilter, MethodDescriptor, SimulationRequest, ViewerAppState,
    WorkspaceTab,
};
