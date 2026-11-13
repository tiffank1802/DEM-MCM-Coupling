//! Full-scale L1 & RMS Discrepancy focus view.

use dem_mcm_coupling::gui::ComparisonBundle;
use egui::Ui;

use crate::plots::{draw_chain_curves, PlotMetric};

pub fn draw_errors_view(ui: &mut Ui, bundle: &ComparisonBundle) {
    let hm = &bundle.hom_metrics;
    let im = &bundle.inhom_metrics;
    ui.horizontal(|ui| {
        ui.strong("Erreurs L1 & Discrepancy RMS");
        ui.separator();
        ui.label(format!(
            "L1 moy Hom {:.4} / Inhom {:.4} | RMS disc Hom {:.4} / Inhom {:.4}",
            hm.l1_mean_target, im.l1_mean_target, hm.rms_discrepancy_mean, im.rms_discrepancy_mean
        ));
    });
    let h = ((ui.available_height() - 36.0) * 0.48).max(180.0);
    draw_chain_curves(ui, "focus_l1", bundle, PlotMetric::L1Error, true, h);
    ui.add_space(6.0);
    draw_chain_curves(
        ui,
        "focus_rms_disc",
        bundle,
        PlotMetric::RmsDiscrepancy,
        true,
        h,
    );
}
