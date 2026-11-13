//! Native `eframe` / `egui` Liquid-Glass Ribbon application for DEM-MCM 3-D
//! Mesh & Markov Chain visualization (`Teneur`, `Particules`, `Erreurs postprocess.py`).

mod plots;
mod viewport3d;

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::thread;

use dem_mcm_coupling::gui::{
    catalog_methods, run_gui_comparison, run_method_benchmark, ActiveSpecies, Camera3D,
    CellColorMode, ComparisonBundle, GuiDatasetCache, MethodBenchmarkRow, SimulationRequest,
    ViewerAppState, WorkspaceTab,
};
use eframe::egui;

use plots::{draw_chain_curves, draw_transition_heatmap, PlotMetric};
use viewport3d::draw_viewport_3d;

struct DemMcmEguiApp {
    state: ViewerAppState,
    bundle_a: Option<ComparisonBundle>,
    bench_rows: Vec<MethodBenchmarkRow>,
    plot_metric: PlotMetric,
    last_error: Option<String>,
    pending_rx: Option<Receiver<Result<ComparisonBundle, String>>>,
    data_dir: PathBuf,
    computing: bool,
}

impl DemMcmEguiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::apply_liquid_glass_style(&cc.egui_ctx);
        let mut app = Self {
            state: ViewerAppState::default(),
            bundle_a: None,
            bench_rows: Vec::new(),
            plot_metric: PlotMetric::Rsd,
            last_error: None,
            pending_rx: None,
            data_dir: PathBuf::from("../data/chunks"),
            computing: false,
        };
        app.trigger_compute(cc.egui_ctx.clone());
        app
    }

    fn apply_liquid_glass_style(ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(24, 28, 36, 225);
        visuals.window_fill = egui::Color32::from_rgba_unmultiplied(28, 33, 42, 220);
        visuals.widgets.noninteractive.bg_fill =
            egui::Color32::from_rgba_unmultiplied(36, 42, 52, 165);
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgba_unmultiplied(45, 52, 64, 175);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgba_unmultiplied(62, 72, 88, 195);
        visuals.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(125, 211, 189, 75);
        ctx.set_visuals(visuals);
    }

    fn trigger_compute(&mut self, ctx: egui::Context) {
        let req = self.state.request.clone();
        let data_dir = self.data_dir.clone();
        let (tx, rx) = mpsc::channel();
        self.pending_rx = Some(rx);
        self.computing = true;

        thread::spawn(move || {
            let mut cache = GuiDatasetCache::new();
            let res = if req.dataset == "real_chunks" {
                cache
                    .get_or_load_real(&data_dir)
                    .and_then(|frames| run_gui_comparison(&req, &frames))
            } else {
                let frames = cache.get_synthetic();
                run_gui_comparison(&req, &frames)
            };
            let _ = tx.send(res.map_err(|e| e.to_string()));
            ctx.request_repaint();
        });
    }

    fn run_benchmark_sync(&mut self) {
        let mut cache = GuiDatasetCache::new();
        let frames = if self.state.request.dataset == "real_chunks" {
            match cache.get_or_load_real(&self.data_dir) {
                Ok(f) => f,
                Err(e) => {
                    self.last_error = Some(e.to_string());
                    return;
                }
            }
        } else {
            cache.get_synthetic()
        };
        let methods: Vec<String> = [
            "cartesian",
            "cylindrical",
            "quantile",
            "octree",
            "voronoi",
            "physics",
            "adaptive",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        self.bench_rows = run_method_benchmark(&self.state.request, &methods, &frames);
    }
}

impl eframe::App for DemMcmEguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Some(rx) = &self.pending_rx {
            if let Ok(result) = rx.try_recv() {
                self.computing = false;
                self.pending_rx = None;
                match result {
                    Ok(bundle) => {
                        self.state.camera = Camera3D::fit_bounds(bundle.mesh.domain_bounds);
                        self.bundle_a = Some(bundle);
                        self.last_error = None;
                    }
                    Err(e) => self.last_error = Some(e),
                }
            }
        }

        if self.state.playing {
            if let Some(b) = &self.bundle_a {
                if !b.frames.is_empty() {
                    self.state.frame_cursor = (self.state.frame_cursor + 1) % b.frames.len();
                    ctx.request_repaint();
                }
            }
        }

        // Unified Top Ribbon Bar (no sidebar, no heavy titled cards)
        egui::TopBottomPanel::top("ribbon_shell").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.strong("DEM-MCM Studio · Liquid Glass 3D");
                ui.separator();
                for (tab, lbl) in [
                    (WorkspaceTab::Mesh3D, "Mélangeur 3D"),
                    (WorkspaceTab::TeneurParticles, "Teneur & Particules"),
                    (WorkspaceTab::ErrorsChains, "Erreurs & Chaînes"),
                    (WorkspaceTab::Benchmark, "Comparateur & Benchmark"),
                ] {
                    ui.selectable_value(&mut self.state.active_tab, tab, lbl);
                }
                ui.separator();
                egui::ComboBox::from_id_salt("dataset_combo")
                    .selected_text(&self.state.request.dataset)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.state.request.dataset,
                            "synthetic_drum".to_string(),
                            "Tambour synthétique (0..359)",
                        );
                        ui.selectable_value(
                            &mut self.state.request.dataset,
                            "real_chunks".to_string(),
                            "DEM réelle data/chunks (0..5999)",
                        );
                    });
                if ui.button("Exécuter").clicked() {
                    self.trigger_compute(ctx.clone());
                }
            });
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                egui::ComboBox::from_id_salt("method_combo")
                    .selected_text(&self.state.request.method)
                    .show_ui(ui, |ui| {
                        for m in catalog_methods() {
                            if ui
                                .selectable_label(self.state.request.method == m.id, m.name)
                                .clicked()
                            {
                                self.state.request.method = m.id.to_string();
                                self.state.request.kwargs =
                                    SimulationRequest::default_kwargs_for(m.id);
                            }
                        }
                    });
                ui.separator();
                ui.label("NLT:");
                ui.add(egui::DragValue::new(&mut self.state.request.nlt).range(1..=12));
                ui.label("step:");
                ui.add(egui::DragValue::new(&mut self.state.request.step).range(1..=500));
                ui.label("tau:");
                ui.add(egui::DragValue::new(&mut self.state.request.tau).range(1..=500));
                ui.label("start:");
                ui.add(egui::DragValue::new(&mut self.state.request.start_index).range(0..=6000));
                ui.checkbox(
                    &mut self.state.request.predict_full_duration,
                    "Durée DEM complète",
                );
                ui.separator();
                ui.checkbox(&mut self.state.show_small_particles, "4mm");
                ui.checkbox(&mut self.state.show_large_particles, "8mm");
                ui.checkbox(&mut self.state.show_wireframe, "Maillage");
                ui.checkbox(&mut self.state.show_faces, "Faces");
                ui.checkbox(&mut self.state.show_markov_edges, "Flux");
                ui.checkbox(&mut self.state.show_drum, "Tambour");
                egui::ComboBox::from_id_salt("color_combo")
                    .selected_text(self.state.color_mode.as_str())
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
                            ui.selectable_value(&mut self.state.color_mode, m, m.as_str());
                        }
                    });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.last_error {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("Erreur: {err}"));
            }
            if self.computing {
                ui.spinner();
            }
            let Some(bundle) = self.bundle_a.clone() else {
                ui.label("Chargement de la simulation DEM...");
                return;
            };

            ui.horizontal_wrapped(|ui| {
                ui.label(format!(
                    "Méthode: {} ({}/{}) | Teneur {} RMSE: Hom {:.4} / Inhom {:.4} | L1 moy: {:.4} | Horizon DEM [{}, {}]",
                    bundle.method,
                    bundle.diagnostics.n_visited,
                    bundle.mesh.n_cells,
                    bundle.target_species,
                    bundle.hom_metrics.rmse_teneur,
                    bundle.inhom_metrics.rmse_teneur,
                    bundle.inhom_metrics.l1_mean_target,
                    bundle.dem_t_min,
                    bundle.dem_t_max,
                ));
            });

            match self.state.active_tab {
                WorkspaceTab::Mesh3D | WorkspaceTab::DualCompare | WorkspaceTab::EguiPlan => {
                    let avail = ui.available_rect_before_wrap();
                    let vp_rect = egui::Rect::from_min_max(
                        avail.min,
                        egui::pos2(avail.max.x, (avail.min.y + avail.height() * 0.62).max(avail.min.y + 200.0)),
                    );
                    draw_viewport_3d(ui, vp_rect, &bundle, &mut self.state);
                    ui.horizontal(|ui| {
                        if ui
                            .button(if self.state.playing { "Pause" } else { "Lecture" })
                            .clicked()
                        {
                            self.state.playing = !self.state.playing;
                        }
                        let max_f = bundle.frames.len().saturating_sub(1);
                        ui.add(egui::Slider::new(&mut self.state.frame_cursor, 0..=max_f).text("Pas"));
                    });
                    draw_chain_curves(ui, "overview_plot", &bundle, self.plot_metric, true);
                }
                WorkspaceTab::TeneurParticles => {
                    draw_chain_curves(ui, "teneur_rsd", &bundle, PlotMetric::Rsd, true);
                    ui.separator();
                    draw_chain_curves(ui, "part_count", &bundle, PlotMetric::ParticleCount, true);
                }
                WorkspaceTab::ErrorsChains
                | WorkspaceTab::Homogeneous
                | WorkspaceTab::Inhomogeneous => {
                    draw_chain_curves(ui, "l1_err", &bundle, PlotMetric::L1Error, true);
                    draw_chain_curves(ui, "rms_disc", &bundle, PlotMetric::RmsDiscrepancy, true);
                    let p = match self.state.species {
                        ActiveSpecies::Large => &bundle.p_hom_large,
                        _ => &bundle.p_hom_small,
                    };
                    draw_transition_heatmap(ui, p, "Matrice P homogène");
                }
                WorkspaceTab::Benchmark => {
                    if ui.button("Lancer Benchmark (7 méthodes)").clicked() {
                        self.run_benchmark_sync();
                    }
                    egui::Grid::new("bench_grid").striped(true).show(ui, |ui| {
                        ui.strong("Méthode");
                        ui.strong("Vis/Cell");
                        ui.strong("RMSE Teneur (H/I)");
                        ui.strong("MAE Teneur (H/I)");
                        ui.strong("L1 moy (H/I)");
                        ui.strong("RMS disc (H/I)");
                        ui.end_row();
                        for r in &self.bench_rows {
                            ui.label(&r.method);
                            ui.label(format!("{}/{}", r.n_visited, r.n_cells));
                            ui.label(format!(
                                "{:.4} / {:.4}",
                                r.hom_rmse_teneur, r.inhom_rmse_teneur
                            ));
                            ui.label(format!(
                                "{:.4} / {:.4}",
                                r.hom_mae_teneur, r.inhom_mae_teneur
                            ));
                            ui.label(format!("{:.4} / {:.4}", r.hom_l1_mean, r.inhom_l1_mean));
                            ui.label(format!("{:.4} / {:.4}", r.hom_rms_disc, r.inhom_rms_disc));
                            ui.end_row();
                        }
                    });
                }
            }
        });
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "DEM-MCM 3D Studio (egui Liquid Glass)",
        options,
        Box::new(|cc| Ok(Box::new(DemMcmEguiApp::new(cc)))),
    )
}
