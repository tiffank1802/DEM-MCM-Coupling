//! Visualization & 3-D UI state core planned for `egui` (`gui-egui`) and
//! served interactively in the browser via the `dem-mcm-gui` binary.

pub mod camera3d;
pub mod mesh3d;
pub mod simulation;
pub mod state;
pub mod web_ui;

pub use camera3d::{Camera3D, CameraPreset, ProjectedPoint};
pub use mesh3d::{
    extract_mesh_3d, infer_domain, DrumCylinder3D, Mesh3DData, MeshCell3D, SplitPlane3D,
};
pub use simulation::{
    generate_synthetic_drum_frames, run_gui_comparison, run_method_benchmark, BlockSummary,
    ChainPredictionMetrics, ComparisonBundle, GuiDatasetCache, MethodBenchmarkRow, SnapshotFrame3D,
    TransitionLawItem,
};
pub use state::{
    catalog_methods, ActiveSpecies, CellColorMode, MethodDescriptor, SimulationRequest,
    ViewerAppState, WorkspaceTab,
};
