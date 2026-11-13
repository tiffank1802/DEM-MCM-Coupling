//! Compact Summary Overview with "⤢ En très grand" expand triggers.

use dem_mcm_coupling::gui::{ComparisonBundle, ViewerAppState, WorkspaceTab};
use egui::Ui;

use crate::plots::{
    draw_chain_curves, draw_matrix_evolution_overlay, filter_evolution_items, PlotMetric,
};
use crate::viewport3d::draw_viewport_3d;

pub fn draw_summary_view(ui: &mut Ui, bundle: &ComparisonBundle, state: &mut ViewerAppState) {
    let avail = ui.available_size();
    let row_h = ((avail.y - 40.0) * 0.46).max(160.0);

    ui.columns(3, |cols| {
        cols[0].group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("1. Mélangeur 3D");
                if ui.small_button("⤢ Agrandir").clicked() {
                    state.active_tab = WorkspaceTab::Mesh3D;
                }
            });
            let (rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width(), row_h),
                egui::Sense::hover(),
            );
            draw_viewport_3d(ui, rect, bundle, state);
        });

        cols[1].group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("2. Teneur & RSD");
                if ui.small_button("⤢ Agrandir").clicked() {
                    state.active_tab = WorkspaceTab::Teneur;
                }
            });
            draw_chain_curves(ui, "sum_rsd", bundle, PlotMetric::Rsd, true, row_h);
        });

        cols[2].group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("3. Particules (N)");
                if ui.small_button("⤢ Agrandir").clicked() {
                    state.active_tab = WorkspaceTab::Particles;
                }
            });
            draw_chain_curves(
                ui,
                "sum_npart",
                bundle,
                PlotMetric::ParticleCount,
                true,
                row_h,
            );
        });
    });

    ui.add_space(6.0);
    ui.columns(3, |cols| {
        cols[0].group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("4. Erreurs L1 & Discrepancy");
                if ui.small_button("⤢ Agrandir").clicked() {
                    state.active_tab = WorkspaceTab::Errors;
                }
            });
            draw_chain_curves(ui, "sum_l1", bundle, PlotMetric::L1Error, true, row_h);
        });

        cols[1].group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("5. Évolution P_inhom(t)");
                if ui.small_button("⤢ Agrandir").clicked() {
                    state.active_tab = WorkspaceTab::MatrixEvolution;
                }
            });
            let evol = if bundle.target_species == "large" {
                &bundle.matrix_evolution_large
            } else {
                &bundle.matrix_evolution_small
            };
            let items = filter_evolution_items(evol, &state.evol_filter);
            draw_matrix_evolution_overlay(ui, "sum_evol", evol, &items, &state.evol_filter, row_h);
        });

        cols[2].group(|ui| {
            ui.horizontal(|ui| {
                ui.strong("6. Synthèse Hom vs Inhom");
                if ui.small_button("⤢ Benchmark").clicked() {
                    state.active_tab = WorkspaceTab::Benchmark;
                }
            });
            let hm = &bundle.hom_metrics;
            let im = &bundle.inhom_metrics;
            egui::Grid::new("sum_metrics_grid")
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Métrique");
                    ui.strong("Hom");
                    ui.strong("Inhom");
                    ui.end_row();
                    ui.label("RMSE Teneur");
                    ui.label(format!("{:.4}", hm.rmse_teneur));
                    ui.label(format!("{:.4}", im.rmse_teneur));
                    ui.end_row();
                    ui.label("MAE Teneur");
                    ui.label(format!("{:.4}", hm.mae_teneur));
                    ui.label(format!("{:.4}", im.mae_teneur));
                    ui.end_row();
                    ui.label("L1 moy");
                    ui.label(format!("{:.4}", hm.l1_mean_target));
                    ui.label(format!("{:.4}", im.l1_mean_target));
                    ui.end_row();
                    ui.label("RMS disc");
                    ui.label(format!("{:.4}", hm.rms_discrepancy_mean));
                    ui.label(format!("{:.4}", im.rms_discrepancy_mean));
                    ui.end_row();
                    ui.label("Dérive N");
                    ui.label(format!("{:.3}%", hm.mass_drift_pct));
                    ui.label(format!("{:.3}%", im.mass_drift_pct));
                    ui.end_row();
                });
        });
    });
}
