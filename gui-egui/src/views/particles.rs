//! Full-scale Particle Counts & Mass Conservation focus view.

use dem_mcm_coupling::gui::ComparisonBundle;
use egui::Ui;

use crate::plots::{draw_chain_curves, PlotMetric};

pub fn draw_particles_view(ui: &mut Ui, bundle: &ComparisonBundle) {
    let hm = &bundle.hom_metrics;
    let im = &bundle.inhom_metrics;
    ui.horizontal(|ui| {
        ui.strong("Conservation du nombre de particules");
        ui.separator();
        ui.label(format!(
            "Dérive Hom {:.3}% | Inhom {:.3}%",
            hm.mass_drift_pct, im.mass_drift_pct
        ));
    });
    let h = (ui.available_height() - 32.0).max(240.0);
    draw_chain_curves(
        ui,
        "focus_npart",
        bundle,
        PlotMetric::ParticleCount,
        true,
        h,
    );
}
