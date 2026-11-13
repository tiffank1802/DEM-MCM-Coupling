//! Full-scale Transition Matrices (`P_hom`, `P_blocks`, `Delta P_k`) focus view.

use dem_mcm_coupling::gui::{ActiveSpecies, ComparisonBundle, ViewerAppState};
use egui::Ui;

use crate::plots::draw_transition_heatmap;

pub fn draw_matrices_view(ui: &mut Ui, bundle: &ComparisonBundle, state: &mut ViewerAppState) {
    ui.horizontal(|ui| {
        ui.strong("Matrices de transition");
        ui.selectable_value(&mut state.species, ActiveSpecies::Small, "Petites (4 mm)");
        ui.selectable_value(&mut state.species, ActiveSpecies::Large, "Grosses (8 mm)");
    });
    let side = (ui.available_height() - 60.0)
        .min(ui.available_width() * 0.45)
        .max(180.0);
    ui.columns(2, |cols| {
        let p_hom = match state.species {
            ActiveSpecies::Large => &bundle.p_hom_large,
            _ => &bundle.p_hom_small,
        };
        draw_transition_heatmap(&mut cols[0], p_hom, "Matrice P homogène", side);

        if let Some(blk) = bundle.blocks.get(
            state
                .request
                .active_block
                .min(bundle.blocks.len().saturating_sub(1)),
        ) {
            let pb = match state.species {
                ActiveSpecies::Large => &blk.p_large,
                _ => &blk.p_small,
            };
            draw_transition_heatmap(
                &mut cols[1],
                pb,
                &format!("Bloc B{} [{}..{}]", blk.block_index, blk.t_start, blk.t_end),
                side,
            );
        }
    });
}
