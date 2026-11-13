//! Full-scale Inhomogeneous Matrix Component Evolution (`p_{i->j}(t)`) focus view.

use dem_mcm_coupling::gui::{ComparisonBundle, EvolutionScope, EvolutionViewMode, ViewerAppState};
use egui::Ui;

use crate::plots::{draw_matrix_evolution_overlay, filter_evolution_items};

pub fn draw_matrix_evolution_view(
    ui: &mut Ui,
    bundle: &ComparisonBundle,
    state: &mut ViewerAppState,
) {
    let evol = if bundle.target_species == "large" {
        &bundle.matrix_evolution_large
    } else {
        &bundle.matrix_evolution_small
    };

    egui::CollapsingHeader::new("Filtres Évolution P_inhom(t)")
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(
                    &mut state.evol_filter.mode,
                    EvolutionViewMode::Overlay,
                    "Figure superposée",
                );
                ui.selectable_value(
                    &mut state.evol_filter.mode,
                    EvolutionViewMode::GridCommonScale,
                    "Grille échelle commune",
                );
                ui.separator();
                ui.selectable_value(
                    &mut state.evol_filter.scope,
                    EvolutionScope::OffDiagonal,
                    "Hors-diagonale",
                );
                ui.selectable_value(
                    &mut state.evol_filter.scope,
                    EvolutionScope::Diagonal,
                    "Diagonales",
                );
                ui.selectable_value(&mut state.evol_filter.scope, EvolutionScope::All, "Toutes");
                ui.selectable_value(
                    &mut state.evol_filter.scope,
                    EvolutionScope::BySourceCell,
                    "Par cellule source",
                );
                if state.evol_filter.scope == EvolutionScope::BySourceCell {
                    ui.add(
                        egui::DragValue::new(&mut state.evol_filter.source_cell)
                            .range(0..=evol.n_states.saturating_sub(1)),
                    );
                }
                ui.checkbox(&mut state.evol_filter.show_polynomial_fits, "Fits poly.");
                ui.checkbox(&mut state.evol_filter.common_y_scale, "Échelle Y commune");
                if state.evol_filter.zoomed_pair.is_some() && ui.button("← Vue complète").clicked()
                {
                    state.evol_filter.zoomed_pair = None;
                }
            });
        });

    let items = filter_evolution_items(evol, &state.evol_filter);
    if let Some((zs, zd)) = state.evol_filter.zoomed_pair {
        let zoomed: Vec<_> = items
            .iter()
            .copied()
            .filter(|c| c.src == zs && c.dst == zd)
            .collect();
        let h = (ui.available_height() - 24.0).max(240.0);
        draw_matrix_evolution_overlay(ui, "evol_zoom", evol, &zoomed, &state.evol_filter, h);
        return;
    }

    match state.evol_filter.mode {
        EvolutionViewMode::Overlay => {
            let h = ((ui.available_height() - 24.0) * 0.65).max(220.0);
            draw_matrix_evolution_overlay(ui, "evol_overlay", evol, &items, &state.evol_filter, h);
            ui.separator();
            egui::ScrollArea::vertical().show(ui, |ui| {
                egui::Grid::new("evol_laws_grid")
                    .striped(true)
                    .show(ui, |ui| {
                        ui.strong("Transition");
                        ui.strong("Max");
                        ui.strong("R²");
                        ui.strong("RMSE");
                        ui.strong("Loi ajustée");
                        ui.end_row();
                        for item in items.iter().take(40) {
                            if ui
                                .selectable_label(false, format!("{} -> {}", item.src, item.dst))
                                .clicked()
                            {
                                state.evol_filter.zoomed_pair = Some((item.src, item.dst));
                            }
                            ui.label(format!("{:.3}", item.max_prob));
                            ui.label(format!("{:.2}", item.r2));
                            ui.label(format!("{:.4}", item.rmse));
                            ui.label(&item.law_label);
                            ui.end_row();
                        }
                    });
            });
        }
        EvolutionViewMode::GridCommonScale => {
            let page_size = 12usize;
            let n_pages = items.len().div_ceil(page_size).max(1);
            if state.evol_filter.page >= n_pages {
                state.evol_filter.page = n_pages - 1;
            }
            ui.horizontal(|ui| {
                if ui.button("◀").clicked() && state.evol_filter.page > 0 {
                    state.evol_filter.page -= 1;
                }
                ui.label(format!("Page {}/{}", state.evol_filter.page + 1, n_pages));
                if ui.button("▶").clicked() && state.evol_filter.page + 1 < n_pages {
                    state.evol_filter.page += 1;
                }
            });
            let start = state.evol_filter.page * page_size;
            let slice = &items[start..(start + page_size).min(items.len())];
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.columns(3, |cols| {
                    for (idx, item) in slice.iter().enumerate() {
                        let col = &mut cols[idx % 3];
                        col.group(|ui| {
                            ui.horizontal(|ui| {
                                ui.strong(format!("p({}->{})", item.src, item.dst));
                                if ui.small_button("⤢").clicked() {
                                    state.evol_filter.zoomed_pair = Some((item.src, item.dst));
                                }
                            });
                            let one = vec![*item];
                            draw_matrix_evolution_overlay(
                                ui,
                                &format!("evol_grid_{}_{}", item.src, item.dst),
                                evol,
                                &one,
                                &state.evol_filter,
                                150.0,
                            );
                            ui.small(format!("R²={:.2} | {}", item.r2, item.law_label));
                        });
                    }
                });
            });
        }
    }
}
