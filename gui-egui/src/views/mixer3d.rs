//! Full-scale 3-D Mixer focus view.

use dem_mcm_coupling::gui::{ComparisonBundle, ViewerAppState};
use egui::Ui;

use crate::viewport3d::draw_viewport_3d;

pub fn draw_mixer3d_view(ui: &mut Ui, bundle: &ComparisonBundle, state: &mut ViewerAppState) {
    let rect = ui.available_rect_before_wrap();
    draw_viewport_3d(ui, rect, bundle, state);
}
