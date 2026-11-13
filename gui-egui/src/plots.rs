//! `egui_plot` views comparing DEM vs Homogeneous vs Inhomogeneous Markov chains
//! and block-by-block matrix inspection.

use dem_mcm_coupling::gui::ComparisonBundle;
use dem_mcm_coupling::Matrix;
use egui::{Color32, Stroke, Ui};
use egui_plot::{Legend, Line, Plot, PlotPoints};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotMetric {
    Rsd,
    Entropy,
    Danckwerts,
    L1Error,
    RmsDiscrepancy,
    ParticleCount,
}

pub fn draw_chain_curves(
    ui: &mut Ui,
    id_source: &str,
    bundle: &ComparisonBundle,
    metric: PlotMetric,
    show_inhom: bool,
) {
    let times = &bundle.hom_metrics.times_eval;
    let (y_dem, y_hom, y_inh) = match metric {
        PlotMetric::Rsd => (
            &bundle.hom_metrics.rsd_dem,
            &bundle.hom_metrics.rsd_markov,
            &bundle.inhom_metrics.rsd_markov,
        ),
        PlotMetric::Entropy => (
            &bundle.hom_metrics.entropy_dem,
            &bundle.hom_metrics.entropy_markov,
            &bundle.inhom_metrics.entropy_markov,
        ),
        PlotMetric::Danckwerts => (
            &bundle.hom_metrics.danckwerts_dem,
            &bundle.hom_metrics.danckwerts_markov,
            &bundle.inhom_metrics.danckwerts_markov,
        ),
        PlotMetric::L1Error => (
            &bundle.hom_metrics.l1_error_total,
            &bundle.hom_metrics.l1_error_small,
            &bundle.inhom_metrics.l1_error_small,
        ),
        PlotMetric::RmsDiscrepancy => (
            &bundle.hom_metrics.rms_discrepancy_total,
            &bundle.hom_metrics.rms_discrepancy_target,
            &bundle.inhom_metrics.rms_discrepancy_target,
        ),
        PlotMetric::ParticleCount => (
            &bundle.hom_metrics.n_particles_total_dem,
            &bundle.hom_metrics.n_particles_total_mk,
            &bundle.inhom_metrics.n_particles_total_mk,
        ),
    };

    let pts = |ys: &[f64]| -> PlotPoints {
        times
            .iter()
            .zip(ys.iter())
            .map(|(&t, &y)| [t as f64, y])
            .collect()
    };

    Plot::new(id_source)
        .legend(Legend::default())
        .height(240.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(pts(y_dem))
                    .name("DEM (référence)")
                    .color(Color32::from_rgb(226, 232, 240))
                    .width(2.2),
            );
            plot_ui.line(
                Line::new(pts(y_hom))
                    .name("Markov Homogène")
                    .color(Color32::from_rgb(59, 130, 246))
                    .width(2.0),
            );
            if show_inhom {
                plot_ui.line(
                    Line::new(pts(y_inh))
                        .name("Markov Inhomogène")
                        .color(Color32::from_rgb(34, 211, 238))
                        .width(2.2),
                );
            }
        });
}

pub fn draw_transition_heatmap(ui: &mut Ui, p: &Matrix, title: &str) {
    ui.label(title);
    let n = p.rows.max(1);
    let side = ui.available_width().min(280.0).max(120.0);
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(side, side), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    let cell = side / n as f32;

    for r in 0..n {
        for c in 0..n {
            let v = p.get(r, c);
            let col = if v.is_finite() {
                let k = v.clamp(0.0, 1.0).powf(0.55);
                Color32::from_rgb(
                    (18.0 + 235.0 * k) as u8,
                    (28.0 + 190.0 * (k * std::f64::consts::PI * 0.85).sin()) as u8,
                    (80.0 + 170.0 * (1.0 - k)) as u8,
                )
            } else {
                Color32::from_rgb(17, 18, 22)
            };
            let cr = egui::Rect::from_min_size(
                rect.min + egui::vec2(c as f32 * cell, r as f32 * cell),
                egui::vec2((cell - 0.5).max(1.0), (cell - 0.5).max(1.0)),
            );
            painter.rect_filled(cr, 0.0, col);
        }
    }
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::GRAY));

    if let Some(pos) = resp.hover_pos() {
        let c = ((pos.x - rect.left()) / cell) as usize;
        let r = ((pos.y - rect.top()) / cell) as usize;
        if r < n && c < n {
            let val = p.get(r, c);
            resp.on_hover_text(format!("P({r} -> {c}) = {:.4}", val));
        }
    }
}
