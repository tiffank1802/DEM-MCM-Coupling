//! CAD-style application shell coordinating the collapsible Feature Tree,
//! collapsible Ribbon, parametric Timeline, and Focus View tabs.

use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;

use dem_mcm_coupling::gui::{
    run_gui_comparison_cached, run_method_benchmark_cached, Camera3D, ComparisonBundle,
    GuiDatasetCache, MethodBenchmarkRow, ViewerAppState, WorkbenchMode, WorkspaceTab,
};
use dem_mcm_coupling::research::{run_study, ResearchSessionCache, StudyResultBundle};
use eframe::egui;

use crate::feature_tree::draw_feature_tree;
use crate::ribbon::draw_top_strip_and_ribbon;
use crate::views::{
    benchmark::draw_benchmark_view,
    errors::draw_errors_view,
    matrices::draw_matrices_view,
    matrix_evolution::draw_matrix_evolution_view,
    mixer3d::draw_mixer3d_view,
    particles::draw_particles_view,
    study::{draw_study_view, EguiStudyState},
    summary::draw_summary_view,
    teneur::draw_teneur_view,
};

pub struct DemMcmEguiApp {
    state: ViewerAppState,
    study_state: EguiStudyState,
    sessions: Arc<ResearchSessionCache>,
    bundle_a: Option<ComparisonBundle>,
    bench_rows: Vec<MethodBenchmarkRow>,
    last_error: Option<String>,
    pending_rx: Option<Receiver<Result<ComparisonBundle, String>>>,
    study_rx: Option<Receiver<Result<StudyResultBundle, String>>>,
    data_dir: PathBuf,
    computing: bool,
}

impl DemMcmEguiApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::apply_studio_style(&cc.egui_ctx);
        let mut app = Self {
            state: ViewerAppState::default(),
            study_state: EguiStudyState::default(),
            sessions: Arc::new(ResearchSessionCache::new()),
            bundle_a: None,
            bench_rows: Vec::new(),
            last_error: None,
            pending_rx: None,
            study_rx: None,
            data_dir: PathBuf::from("../data/chunks"),
            computing: false,
        };
        app.trigger_compute(cc.egui_ctx.clone());
        app
    }

    fn apply_studio_style(ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgba_unmultiplied(18, 23, 32, 238);
        visuals.window_fill = egui::Color32::from_rgba_unmultiplied(24, 30, 41, 235);
        visuals.widgets.noninteractive.bg_fill =
            egui::Color32::from_rgba_unmultiplied(32, 39, 51, 185);
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgba_unmultiplied(40, 48, 62, 185);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgba_unmultiplied(56, 68, 86, 210);
        visuals.selection.bg_fill = egui::Color32::from_rgba_unmultiplied(125, 211, 189, 75);
        ctx.set_visuals(visuals);
    }

    fn trigger_compute(&mut self, ctx: egui::Context) {
        let req = self.state.request.clone();
        let data_dir = self.data_dir.clone();
        let sessions = Arc::clone(&self.sessions);
        let (tx, rx) = mpsc::channel();
        self.pending_rx = Some(rx);
        self.computing = true;

        thread::spawn(move || {
            let mut cache = GuiDatasetCache::new();
            let res = if req.dataset == "real_chunks" {
                cache
                    .get_or_load_real(&data_dir)
                    .and_then(|frames| run_gui_comparison_cached(&req, &frames, &sessions))
            } else {
                let frames = cache.get_synthetic();
                run_gui_comparison_cached(&req, &frames, &sessions)
            };
            let _ = tx.send(res.map_err(|e| e.to_string()));
            ctx.request_repaint();
        });
    }

    fn trigger_study_run(&mut self, ctx: egui::Context) {
        let spec = self.study_state.build_spec(&self.state.request);
        let data_dir = self.data_dir.clone();
        let sessions = Arc::clone(&self.sessions);
        let (tx, rx) = mpsc::channel();
        self.study_rx = Some(rx);
        self.computing = true;

        thread::spawn(move || {
            let mut cache = GuiDatasetCache::new();
            let res = if spec.base_request.dataset == "real_chunks" {
                cache
                    .get_or_load_real(&data_dir)
                    .and_then(|frames| run_study(&spec, &sessions, &frames))
            } else {
                let frames = cache.get_synthetic();
                run_study(&spec, &sessions, &frames)
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
        self.bench_rows =
            run_method_benchmark_cached(&self.state.request, &methods, &frames, &self.sessions);
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
        if let Some(rx) = &self.study_rx {
            if let Ok(result) = rx.try_recv() {
                self.computing = false;
                self.study_rx = None;
                match result {
                    Ok(sb) => {
                        self.study_state.result = Some(sb);
                        self.study_state.selected_cell = (0, 0);
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

        let mut trigger_run = false;
        let mut trigger_study_run = false;
        let mut run_bench = false;

        draw_top_strip_and_ribbon(
            ctx,
            &mut self.state,
            self.computing,
            &mut trigger_run,
            &mut trigger_study_run,
        );

        let is_explore = self.state.workbench_mode == WorkbenchMode::Explore;

        if is_explore && self.state.panels.feature_tree_expanded {
            egui::SidePanel::left("feature_tree")
                .resizable(true)
                .default_width(250.0)
                .show(ctx, |ui| {
                    draw_feature_tree(
                        ui,
                        &mut self.state,
                        self.bundle_a.as_ref(),
                        &mut trigger_run,
                    );
                });
        }

        if is_explore {
            // Bottom Onshape-style view switcher + collapsible timeline
            egui::TopBottomPanel::bottom("view_tabs_bar").show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for (tab, lbl) in [
                        (WorkspaceTab::Summary, "Sommaire"),
                        (WorkspaceTab::Mesh3D, "Mélangeur 3D"),
                        (WorkspaceTab::Teneur, "Teneur (Espèce)"),
                        (WorkspaceTab::Particles, "Particules (N)"),
                        (WorkspaceTab::Errors, "Erreurs L1 & RMS"),
                        (WorkspaceTab::MatrixEvolution, "Évolution P_inhom(t)"),
                        (WorkspaceTab::Matrices, "Matrices P & ΔP"),
                        (WorkspaceTab::Benchmark, "Benchmark"),
                    ] {
                        ui.selectable_value(&mut self.state.active_tab, tab, lbl);
                    }
                });
            });
        }

        if self.state.workbench_mode != WorkbenchMode::ResearchStudy
            && self.state.panels.timeline_expanded
        {
            if let Some(b) = &self.bundle_a {
                egui::TopBottomPanel::bottom("parametric_timeline").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .button(if self.state.playing {
                                "Pause"
                            } else {
                                "Lecture"
                            })
                            .clicked()
                        {
                            self.state.playing = !self.state.playing;
                        }
                        let max_f = b.frames.len().saturating_sub(1);
                        ui.add(
                            egui::Slider::new(&mut self.state.frame_cursor, 0..=max_f).text("Pas"),
                        );
                    });
                });
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.last_error {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("Erreur: {err}"));
            }
            if self.state.active_tab == WorkspaceTab::Study {
                draw_study_view(
                    ui,
                    &self.state.request,
                    &mut self.study_state,
                    self.computing,
                    &mut trigger_study_run,
                );
                return;
            }
            let Some(bundle) = self.bundle_a.clone() else {
                ui.label("Chargement de la simulation DEM...");
                return;
            };

            match self.state.active_tab {
                WorkspaceTab::Summary => draw_summary_view(ui, &bundle, &mut self.state),
                WorkspaceTab::Mesh3D | WorkspaceTab::DualCompare | WorkspaceTab::EguiPlan => {
                    draw_mixer3d_view(ui, &bundle, &mut self.state);
                }
                WorkspaceTab::Teneur | WorkspaceTab::TeneurParticles => {
                    draw_teneur_view(ui, &bundle);
                }
                WorkspaceTab::Particles => draw_particles_view(ui, &bundle),
                WorkspaceTab::Errors | WorkspaceTab::ErrorsChains => {
                    draw_errors_view(ui, &bundle);
                }
                WorkspaceTab::MatrixEvolution | WorkspaceTab::Inhomogeneous => {
                    draw_matrix_evolution_view(ui, &bundle, &mut self.state);
                }
                WorkspaceTab::Matrices | WorkspaceTab::Homogeneous => {
                    draw_matrices_view(ui, &bundle, &mut self.state);
                }
                WorkspaceTab::Benchmark => {
                    draw_benchmark_view(ui, &self.bench_rows, &mut run_bench);
                }
                WorkspaceTab::Study => {}
            }
        });

        if trigger_run && !self.computing {
            self.trigger_compute(ctx.clone());
        }
        if trigger_study_run && !self.computing {
            self.trigger_study_run(ctx.clone());
        }
        if run_bench {
            self.run_benchmark_sync();
        }
    }
}
