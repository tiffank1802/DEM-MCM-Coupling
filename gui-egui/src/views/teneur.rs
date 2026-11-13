//! Full-scale Teneur & Mixing Indicators (RSD / Entropy / Danckwerts) focus view.

use dem_mcm_coupling::gui::ComparisonBundle;
use egui::Ui;

use crate::plots::{draw_chain_curves, PlotMetric};

pub fn draw_teneur_view(ui: &mut Ui, bundle: &ComparisonBundle) {
    let hm = &bundle.hom_metrics;
    let im = &bundle.inhom_metrics;
    ui.horizontal(|ui| {
        ui.strong(format!("Teneur ({})", bundle.target_species));
        ui.separator();
        ui.label(format!(
            "RMSE Hom {:.4} / Inhom {:.4} | MAE Hom {:.4} / Inhom {:.4}",
            hm.rmse_teneur, im.rmse_teneur, hm.mae_teneur, im.mae_teneur
        ));
    });
    let h = ((ui.available_height() - 36.0) * 0.48).max(180.0);
    draw_chain_curves(ui, "focus_rsd", bundle, PlotMetric::Rsd, true, h);
    ui.add_space(6.0);
    draw_chain_curves(ui, "focus_entropy", bundle, PlotMetric::Entropy, true, h);
}
