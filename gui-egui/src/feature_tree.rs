//! Collapsible CAD-style Feature / Model Browser Tree (`egui::CollapsingHeader`).

use dem_mcm_coupling::gui::{
    catalog_methods, CellColorMode, ComparisonBundle, SimulationRequest, ViewerAppState,
};
use egui::Ui;

pub fn draw_feature_tree(
    ui: &mut Ui,
    state: &mut ViewerAppState,
    bundle: Option<&ComparisonBundle>,
    trigger_run: &mut bool,
) {
    ui.horizontal(|ui| {
        ui.strong("ARBRE DE MODÈLE");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("◂").clicked() {
                state.panels.feature_tree_expanded = false;
            }
        });
    });
    ui.separator();

    egui::ScrollArea::vertical().show(ui, |ui| {
        egui::CollapsingHeader::new("1. Source DEM & Particules")
            .default_open(true)
            .show(ui, |ui| {
                egui::ComboBox::from_id_salt("tree_dataset")
                    .selected_text(&state.request.dataset)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut state.request.dataset,
                            "synthetic_drum".to_string(),
                            "Tambour synthétique (0..359)",
                        );
                        ui.selectable_value(
                            &mut state.request.dataset,
                            "real_chunks".to_string(),
                            "DEM réelle data/chunks (0..5999)",
                        );
                    });
                ui.checkbox(&mut state.show_small_particles, "Petites (4 mm)");
                ui.checkbox(&mut state.show_large_particles, "Grosses (8 mm)");
            });

        egui::CollapsingHeader::new("2. Maillage 3D (Partitioner)")
            .default_open(true)
            .show(ui, |ui| {
                egui::ComboBox::from_id_salt("tree_method")
                    .selected_text(&state.request.method)
                    .show_ui(ui, |ui| {
                        for m in catalog_methods() {
                            if ui
                                .selectable_label(state.request.method == m.id, m.name)
                                .clicked()
                            {
                                state.request.method = m.id.to_string();
                                state.request.kwargs = SimulationRequest::default_kwargs_for(m.id);
                            }
                        }
                    });
                ui.label("Coloration 3D:");
                egui::ComboBox::from_id_salt("tree_color")
                    .selected_text(state.color_mode.as_str())
                    .show_ui(ui, |ui| {
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

        egui::CollapsingHeader::new("3. Construction Markov")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("NLT:");
                    ui.add(egui::DragValue::new(&mut state.request.nlt).range(1..=12));
                    ui.label("step:");
                    ui.add(egui::DragValue::new(&mut state.request.step).range(1..=500));
                });
                ui.horizontal(|ui| {
                    ui.label("tau:");
                    ui.add(egui::DragValue::new(&mut state.request.tau).range(1..=500));
                    ui.label("start:");
                    ui.add(egui::DragValue::new(&mut state.request.start_index).range(0..=6000));
                });
                ui.checkbox(
                    &mut state.request.predict_full_duration,
                    "Prédiction DEM complète",
                );
                if ui.button("Recalculer").clicked() {
                    *trigger_run = true;
                }
            });

        egui::CollapsingHeader::new("4. Calques & Critère")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut state.show_wireframe, "Maillage 3D");
                ui.checkbox(&mut state.show_faces, "Faces translucides");
                ui.checkbox(&mut state.show_markov_edges, "Flux Markov P_ij");
                ui.checkbox(&mut state.show_velocities, "Vecteurs vitesse");
                ui.checkbox(&mut state.show_drum, "Paroi tambour");
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Cible:");
                    ui.selectable_value(
                        &mut state.request.target_species,
                        "small".to_string(),
                        "4mm",
                    );
                    ui.selectable_value(
                        &mut state.request.target_species,
                        "large".to_string(),
                        "8mm",
                    );
                });
            });

        egui::CollapsingHeader::new("5. Inspecteur")
            .default_open(true)
            .show(ui, |ui| {
                if let (Some(cid), Some(b)) = (state.selected_cell, bundle) {
                    let crm = b
                        .inhom_metrics
                        .cell_rmse_per_cell
                        .get(cid)
                        .copied()
                        .unwrap_or(0.0);
                    ui.label(format!("Cellule #{cid}"));
                    ui.label(format!("Discrepancy RMSE: {crm:.4}"));
                } else {
                    ui.weak("Cliquez une cellule 3D pour l'inspecter.");
                }
            });
    });
}
