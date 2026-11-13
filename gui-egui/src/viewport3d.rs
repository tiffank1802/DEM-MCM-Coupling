//! Native `egui::Painter` 3-D mesh, particle, velocity, and Markov flow renderer
//! consuming `dem_mcm_coupling::gui::{Camera3D, ComparisonBundle}`.

use std::collections::HashMap;
use std::f64::consts::PI;

use dem_mcm_coupling::gui::{
    Camera3D, CameraPreset, CellColorMode, ComparisonBundle, ViewerAppState,
};
use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui};

enum DrawItem {
    Poly {
        depth: f64,
        pts: Vec<Pos2>,
        fill: Color32,
        stroke: Option<Color32>,
    },
    Line {
        depth: f64,
        a: Pos2,
        b: Pos2,
        stroke: Stroke,
    },
    Circle {
        depth: f64,
        center: Pos2,
        radius: f32,
        fill: Color32,
    },
}

fn color_concentration(v: f64, alpha: u8, is_error: bool) -> Color32 {
    let t = v.clamp(0.0, 1.0);
    if is_error {
        let r = (30.0 + 225.0 * t.powf(0.7)) as u8;
        let g = (200.0 * (1.0 - 0.85 * t)) as u8;
        let b = (130.0 * (1.0 - t)) as u8;
        return Color32::from_rgba_unmultiplied(r, g, b, alpha);
    }
    let (r, g, b) = if t < 0.5 {
        let k = t * 2.0;
        (37.0 + 189.0 * k, 99.0 + 133.0 * k, 235.0 + 5.0 * k)
    } else {
        let k = (t - 0.5) * 2.0;
        (226.0 + 13.0 * k, 232.0 - 164.0 * k, 240.0 - 172.0 * k)
    };
    Color32::from_rgba_unmultiplied(r as u8, g as u8, b as u8, alpha)
}

fn cell_metric_value(
    bundle: &ComparisonBundle,
    cid: usize,
    timestep: usize,
    mode: CellColorMode,
) -> (f64, bool) {
    let eval_idx = bundle
        .hom_metrics
        .times_eval
        .iter()
        .enumerate()
        .min_by_key(|(_, &t)| (t - timestep as i64).abs())
        .map_or(0, |(i, _)| i);

    let c_dem = bundle
        .dem_conc_traj
        .get(eval_idx)
        .and_then(|r| r.get(cid))
        .copied()
        .unwrap_or(0.5);
    let c_hom = bundle
        .hom_conc_traj
        .get(eval_idx)
        .and_then(|r| r.get(cid))
        .copied()
        .unwrap_or(0.5);
    let c_inh = bundle
        .inhom_conc_traj
        .get(eval_idx)
        .and_then(|r| r.get(cid))
        .copied()
        .unwrap_or(0.5);

    match mode {
        CellColorMode::DemConcentration => (c_dem, false),
        CellColorMode::HomConcentration => (c_hom, false),
        CellColorMode::InhomConcentration => (c_inh, false),
        CellColorMode::HomError => (((c_hom - c_dem).abs() * 2.5).min(1.0), true),
        CellColorMode::InhomError => (((c_inh - c_dem).abs() * 2.5).min(1.0), true),
        CellColorMode::CellRmseDiscrepancy => (
            (bundle
                .inhom_metrics
                .cell_rmse_per_cell
                .get(cid)
                .copied()
                .unwrap_or(0.0)
                * 8.0)
                .min(1.0),
            true,
        ),
        _ => (0.5, false),
    }
}

pub fn draw_viewport_3d(
    ui: &mut Ui,
    rect: Rect,
    bundle: &ComparisonBundle,
    state: &mut ViewerAppState,
) {
    let response = ui.allocate_rect(rect, Sense::click_and_drag());

    // Interactive orbit (primary drag), pan (secondary or shift+drag), wheel zoom
    if response.dragged_by(egui::PointerButton::Primary) {
        let delta = response.drag_delta();
        if ui.input(|i| i.modifiers.shift) {
            state
                .camera
                .pan(delta.x as f64, delta.y as f64, rect.height() as f64);
        } else {
            state
                .camera
                .orbit(delta.x as f64 * 0.008, delta.y as f64 * 0.008);
        }
    } else if response.dragged_by(egui::PointerButton::Secondary) {
        let delta = response.drag_delta();
        state
            .camera
            .pan(delta.x as f64, delta.y as f64, rect.height() as f64);
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll.abs() > 0.0 {
            let factor = if scroll > 0.0 { 0.90 } else { 1.11 };
            state.camera.zoom(factor);
        }
    }

    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 4.0, Color32::from_rgb(18, 19, 24));
    painter.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_rgb(54, 56, 66)));

    let w = rect.width() as f64;
    let h = rect.height() as f64;
    let proj = |cam: &Camera3D, p: [f64; 3]| -> Option<(Pos2, f64, f64)> {
        cam.project(p, w, h).map(|pp| {
            (
                Pos2::new(rect.left() + pp.x as f32, rect.top() + pp.y as f32),
                pp.depth,
                pp.scale,
            )
        })
    };

    // Drum cylinder wireframe
    if state.show_drum {
        let dr = &bundle.mesh.drum;
        let n_seg = 36;
        let col = Color32::from_rgba_unmultiplied(148, 163, 184, 70);
        for &z in &[dr.z_min, dr.z_max] {
            let mut ring = Vec::new();
            for i in 0..=n_seg {
                let a = (i as f64 / n_seg as f64) * 2.0 * PI;
                if let Some((pt, _, _)) =
                    proj(&state.camera, [dr.radius * a.cos(), dr.radius * a.sin(), z])
                {
                    ring.push(pt);
                }
            }
            if ring.len() >= 2 {
                painter.add(egui::Shape::line(ring, Stroke::new(1.0, col)));
            }
        }
    }

    let frame = bundle.frames.get(
        state
            .frame_cursor
            .min(bundle.frames.len().saturating_sub(1)),
    );
    let ts = frame.map_or(0, |f| f.timestep);

    let mut items: Vec<DrawItem> = Vec::new();
    let mut cell_screen_centers: Vec<(usize, Pos2)> = Vec::with_capacity(bundle.mesh.cells.len());

    for cell in &bundle.mesh.cells {
        if let Some((cp, _, _)) = proj(&state.camera, cell.centroid) {
            cell_screen_centers.push((cell.id, cp));
        }
        let is_sel = state.selected_cell == Some(cell.id);
        let (val, is_err) = cell_metric_value(bundle, cell.id, ts, state.color_mode);
        let fill = color_concentration(val, if is_sel { 130 } else { 56 }, is_err);

        if state.show_faces {
            for poly in &cell.faces {
                let mut pts = Vec::with_capacity(poly.len());
                let mut depth_sum = 0.0;
                let mut ok = true;
                for &v in poly {
                    if let Some((p2, d, _)) = proj(&state.camera, v) {
                        pts.push(p2);
                        depth_sum += d;
                    } else {
                        ok = false;
                        break;
                    }
                }
                if ok && pts.len() >= 3 {
                    items.push(DrawItem::Poly {
                        depth: depth_sum / pts.len() as f64,
                        pts,
                        fill,
                        stroke: is_sel.then_some(Color32::WHITE),
                    });
                }
            }
        }

        if state.show_wireframe {
            let stroke_col = if is_sel {
                Color32::WHITE
            } else {
                Color32::from_rgba_unmultiplied(148, 163, 184, 110)
            };
            for &(a, b) in &cell.wire_segments {
                if let (Some((pa, da, _)), Some((pb, db, _))) =
                    (proj(&state.camera, a), proj(&state.camera, b))
                {
                    items.push(DrawItem::Line {
                        depth: 0.5 * (da + db) - 1e-4,
                        a: pa,
                        b: pb,
                        stroke: Stroke::new(if is_sel { 1.8 } else { 0.9 }, stroke_col),
                    });
                }
            }
        }
    }

    // Markov flow arrows between centroids
    if state.show_markov_edges {
        let cen_map: HashMap<usize, [f64; 3]> = bundle
            .mesh
            .cells
            .iter()
            .map(|c| (c.id, c.centroid))
            .collect();
        for &(i, j, p) in &bundle.top_edges_small {
            if let (Some(&ca), Some(&cb)) = (cen_map.get(&i), cen_map.get(&j)) {
                if let (Some((pa, da, _)), Some((pb, db, _))) =
                    (proj(&state.camera, ca), proj(&state.camera, cb))
                {
                    items.push(DrawItem::Line {
                        depth: 0.5 * (da + db) - 2e-4,
                        a: pa,
                        b: pb,
                        stroke: Stroke::new(
                            1.0 + (p as f32) * 4.5,
                            Color32::from_rgba_unmultiplied(56, 189, 248, 180),
                        ),
                    });
                }
            }
        }
    }

    // Both particle sizes rendered faithfully at exact diameter scale (0.004m vs 0.008m)
    if state.show_particles && (state.show_small_particles || state.show_large_particles) {
        if let Some(f) = frame {
            for i in 0..f.coords.len() {
                let is_s = *f.is_small.get(i).unwrap_or(&true);
                if (is_s && !state.show_small_particles) || (!is_s && !state.show_large_particles) {
                    continue;
                }
                if let Some((pt, depth, scale)) = proj(&state.camera, f.coords[i]) {
                    let d = f
                        .diameters
                        .get(i)
                        .copied()
                        .unwrap_or(if is_s { 0.004 } else { 0.008 });
                    let min_r = if is_s { 2.0 } else { 4.0 };
                    let max_r = if is_s { 5.0 } else { 10.0 };
                    let rad = ((scale * d * 0.56) as f32).clamp(min_r, max_r);
                    items.push(DrawItem::Circle {
                        depth: depth - 3e-4,
                        center: pt,
                        radius: rad,
                        fill: if is_s {
                            Color32::from_rgb(125, 211, 200)
                        } else {
                            Color32::from_rgb(238, 192, 120)
                        },
                    });
                    if state.show_velocities {
                        if let Some(v) = f.velocities.get(i) {
                            let tip = [
                                f.coords[i][0] + v[0] * 0.025,
                                f.coords[i][1] + v[1] * 0.025,
                                f.coords[i][2] + v[2] * 0.025,
                            ];
                            if let Some((pt_tip, _, _)) = proj(&state.camera, tip) {
                                items.push(DrawItem::Line {
                                    depth: depth - 3.5e-4,
                                    a: pt,
                                    b: pt_tip,
                                    stroke: Stroke::new(1.1, Color32::from_rgb(167, 139, 250)),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // Painter's sort back-to-front
    items.sort_by(|a, b| {
        let da = match a {
            DrawItem::Poly { depth, .. }
            | DrawItem::Line { depth, .. }
            | DrawItem::Circle { depth, .. } => *depth,
        };
        let db = match b {
            DrawItem::Poly { depth, .. }
            | DrawItem::Line { depth, .. }
            | DrawItem::Circle { depth, .. } => *depth,
        };
        db.total_cmp(&da)
    });

    for item in items {
        match item {
            DrawItem::Poly {
                pts, fill, stroke, ..
            } => {
                let s = stroke.map_or(Stroke::NONE, |c| Stroke::new(1.3, c));
                painter.add(egui::Shape::convex_polygon(pts, fill, s));
            }
            DrawItem::Line { a, b, stroke, .. } => {
                painter.line_segment([a, b], stroke);
            }
            DrawItem::Circle {
                center,
                radius,
                fill,
                ..
            } => {
                painter.circle_filled(center, radius, fill);
            }
        }
    }

    // Pick nearest cell centroid on click
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            state.selected_cell = cell_screen_centers
                .iter()
                .filter_map(|&(id, c)| {
                    let d2 = (c.x - pos.x).powi(2) + (c.y - pos.y).powi(2);
                    (d2 <= 26.0 * 26.0).then_some((id, d2))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(id, _)| id);
        }
    }

    // Preset camera buttons overlay top-right
    let btn_rect = Rect::from_min_size(
        rect.right_top() + egui::vec2(-220.0, 8.0),
        egui::vec2(212.0, 26.0),
    );
    ui.allocate_new_ui(egui::UiBuilder::new().max_rect(btn_rect), |ui| {
        ui.horizontal(|ui| {
            for (lbl, preset) in [
                ("Iso", CameraPreset::Iso),
                ("XY", CameraPreset::FrontXY),
                ("XZ", CameraPreset::TopXZ),
                ("YZ", CameraPreset::SideYZ),
            ] {
                if ui.small_button(lbl).clicked() {
                    state.camera.set_preset(preset);
                }
            }
        });
    });
}
