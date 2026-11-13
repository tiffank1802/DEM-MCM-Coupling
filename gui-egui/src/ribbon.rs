//! Collapsible Top Command Strip & Ribbon with dropdown menus.

use dem_mcm_coupling::gui::{CellColorMode, ViewerAppState, WorkspaceTab};
use egui::Ui;

pub fn draw_top_strip_and_ribbon(
    ctx: &egui::Context,
    state: &mut ViewerAppState,
    computing: bool,
    trigger_run: &mut bool,
) {
    egui::TopBottomPanel::top("top_strip").show(ctx, |ui| {
        ui.horizontal(|ui| {
            ui.strong("DEM-MCM Studio");
            ui.separator();
            if ui
                .selectable_label(state.panels.feature_tree_expanded, "Arbre")
                .clicked()
            {
                state.panels.feature_tree_expanded = !state.panels.feature_tree_expanded;
            }
            if ui
                .selectable_label(state.panels.ribbon_expanded, "Ruban")
                .clicked()
            {
                state.panels.ribbon_expanded = !state.panels.ribbon_expanded;
            }
            if ui
                .selectable_label(state.panels.timeline_expanded, "Frise")
                .clicked()
            {
                state.panels.timeline_expanded = !state.panels.timeline_expanded;
            }
            ui.separator();
            if ui.button("Exécuter").clicked() {
                *trigger_run = true;
            }
            if computing {
                ui.spinner();
            }
        });
    });

    if state.panels.ribbon_expanded {
        egui::TopBottomPanel::top("collapsible_ribbon").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Sommaire").clicked() {
                    state.active_tab = WorkspaceTab::Summary;
                }
                if ui.button("Mélangeur 3D").clicked() {
                    state.active_tab = WorkspaceTab::Mesh3D;
                }
                if ui.button("Évolution P_inhom(t)").clicked() {
                    state.active_tab = WorkspaceTab::MatrixEvolution;
                }
                ui.separator();
                ui.menu_button("Calques 3D ▾", |ui: &mut Ui| {
                    ui.checkbox(&mut state.show_small_particles, "4 mm");
                    ui.checkbox(&mut state.show_large_particles, "8 mm");
                    ui.checkbox(&mut state.show_wireframe, "Maillage");
                    ui.checkbox(&mut state.show_faces, "Faces");
                    ui.checkbox(&mut state.show_markov_edges, "Flux");
                    ui.checkbox(&mut state.show_velocities, "Vitesses");
                    ui.checkbox(&mut state.show_drum, "Tambour");
                });
                ui.menu_button("Coloration ▾", |ui: &mut Ui| {
                    for m in [
                        CellColorMode::DemConcentration,
                        CellColorMode::HomConcentration,
                        CellColorMode::InhomConcentration,
                        CellColorMode::HomError,
                        CellColorMode::InhomError,
                        CellColorMode::CellRmseDiscrepancy,
                        CellColorMode::Population,
                        CellColorMode::CellId,
                    ] {
                        ui.selectable_value(&mut state.color_mode, m, m.as_str());
                    }
                });
            });
        });
    }
}
