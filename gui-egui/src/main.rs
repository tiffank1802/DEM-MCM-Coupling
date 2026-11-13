//! Native `eframe` / `egui` CAD-style entry point for DEM-MCM Studio.

mod app;
mod feature_tree;
mod plots;
mod ribbon;
mod viewport3d;
mod views;

use app::DemMcmEguiApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "DEM-MCM Studio",
        options,
        Box::new(|cc| Ok(Box::new(DemMcmEguiApp::new(cc)))),
    )
}
