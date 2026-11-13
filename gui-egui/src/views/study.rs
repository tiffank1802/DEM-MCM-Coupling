//! Schema-driven Parametric & Multi-Law Research Study Workshop for `egui`.

use std::collections::HashMap;

use dem_mcm_coupling::gui::SimulationRequest;
use dem_mcm_coupling::research::{
    build_preset_spec, catalog_study_presets, AxisValue, LawFamily, LawSelectionCriterion,
    ReferencePolicy, StudyAxisKind, StudyAxisSpec, StudyChainMode, StudyResultBundle, StudySpec,
};
use egui::{Color32, RichText, Ui};
use egui_plot::{Legend, Line, LineStyle, Plot, PlotPoints, Points};

const STUDY_PALETTE: &[Color32] = &[
    Color32::from_rgb(125, 211, 189),
    Color32::from_rgb(158, 192, 220),
    Color32::from_rgb(238, 192, 120),
    Color32::from_rgb(229, 152, 155),
    Color32::from_rgb(184, 184, 230),
    Color32::from_rgb(216, 164, 143),
    Color32::from_rgb(136, 201, 194),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EguiSensMetric {
    RmseRsdHom,
    RmseTeneurHom,
    L1MeanHom,
    FrobeniusRel,
    NanRowsHom,
    LambdaMax,
}

impl EguiSensMetric {
    pub fn label(self) -> &'static str {
        match self {
            Self::RmseRsdHom => "RMSE RSD (Hom)",
            Self::RmseTeneurHom => "RMSE Teneur (Hom)",
            Self::L1MeanHom => "Erreur L1 moy.",
            Self::FrobeniusRel => "Erreur Frobenius rel.",
            Self::NanRowsHom => "Lignes NaN",
            Self::LambdaMax => "Valeur propre |lambda| max",
        }
    }
}

pub struct EguiStudyState {
    pub preset_id: String,
    pub chain_mode: StudyChainMode,
    pub axis1_kind: StudyAxisKind,
    pub axis1_text: String,
    pub couple_tau: bool,
    pub selected_methods: HashMap<String, bool>,
    pub enable_axis2: bool,
    pub axis2_kind: StudyAxisKind,
    pub axis2_text: String,
    pub enable_law_study: bool,
    pub law_criterion: LawSelectionCriterion,
    pub use_bed_frame: bool,
    pub sens_metric: EguiSensMetric,
    pub selected_cell: (usize, usize),
    pub result: Option<StudyResultBundle>,
}

impl Default for EguiStudyState {
    fn default() -> Self {
        let mut selected_methods = HashMap::new();
        for m in [
            "cartesian",
            "cylindrical",
            "quantile",
            "octree",
            "voronoi",
            "physics",
            "adaptive",
        ] {
            selected_methods.insert(m.to_string(), true);
        }
        selected_methods.insert("gmm".to_string(), false);
        Self {
            preset_id: "hom_start".to_string(),
            chain_mode: StudyChainMode::Homogeneous,
            axis1_kind: StudyAxisKind::StartIndex,
            axis1_text: "0, 20, 40, 80, 120, 180".to_string(),
            couple_tau: false,
            selected_methods,
            enable_axis2: false,
            axis2_kind: StudyAxisKind::Nlt,
            axis2_text: "3, 5, 7".to_string(),
            enable_law_study: false,
            law_criterion: LawSelectionCriterion::Aicc,
            use_bed_frame: false,
            sens_metric: EguiSensMetric::RmseRsdHom,
            selected_cell: (0, 0),
            result: None,
        }
    }
}

impl EguiStudyState {
    pub fn apply_preset(&mut self, preset_id: &str, base_req: &SimulationRequest) {
        self.preset_id = preset_id.to_string();
        let spec = build_preset_spec(preset_id, base_req);
        self.chain_mode = spec.chain_mode;
        self.axis1_kind = spec.primary_axis.kind;
        self.couple_tau = spec.primary_axis.couple_step_to_tau;
        self.enable_law_study = spec.enable_law_study;
        self.law_criterion = spec.law_criterion;
        self.use_bed_frame = spec.use_bed_frame;

        if spec.primary_axis.kind == StudyAxisKind::Method {
            for val in self.selected_methods.values_mut() {
                *val = false;
            }
            for av in &spec.primary_axis.values {
                if let AxisValue::Text(name) = av {
                    self.selected_methods.insert(name.clone(), true);
                }
            }
        } else {
            self.axis1_text = spec
                .primary_axis
                .values
                .iter()
                .map(AxisValue::label)
                .collect::<Vec<_>>()
                .join(", ");
        }

        if let Some(ax2) = &spec.secondary_axis {
            self.enable_axis2 = true;
            self.axis2_kind = ax2.kind;
            self.axis2_text = ax2
                .values
                .iter()
                .map(AxisValue::label)
                .collect::<Vec<_>>()
                .join(", ");
        } else {
            self.enable_axis2 = false;
        }
    }

    pub fn build_spec(&self, base_req: &SimulationRequest) -> StudySpec {
        let axis1_vals = if self.axis1_kind == StudyAxisKind::Method {
            let mut list: Vec<AxisValue> = self
                .selected_methods
                .iter()
                .filter_map(|(k, &on)| {
                    if on {
                        Some(AxisValue::Text(k.clone()))
                    } else {
                        None
                    }
                })
                .collect();
            if list.is_empty() {
                list.push(AxisValue::Text("cartesian".to_string()));
                list.push(AxisValue::Text("voronoi".to_string()));
            }
            list
        } else if self.axis1_kind == StudyAxisKind::SpeciesMode {
            vec![AxisValue::Bool(false), AxisValue::Bool(true)]
        } else {
            parse_axis_values(&self.axis1_text, self.axis1_kind)
        };

        let secondary_axis = if self.enable_axis2 {
            let vals = parse_axis_values(&self.axis2_text, self.axis2_kind);
            if vals.is_empty() {
                None
            } else {
                Some(StudyAxisSpec {
                    kind: self.axis2_kind,
                    values: vals,
                    couple_step_to_tau: false,
                })
            }
        } else {
            None
        };

        let title = if let Some(ax2) = &secondary_axis {
            format!(
                "Étude 2D · {} x {}",
                self.axis1_kind.display_name(),
                ax2.kind.display_name()
            )
        } else {
            format!("Étude · {}", self.axis1_kind.display_name())
        };

        StudySpec {
            study_id: self.preset_id.clone(),
            title,
            subtitle: format!(
                "{} · {}",
                base_req.dataset,
                if self.enable_law_study {
                    "7 familles de lois p(t)"
                } else {
                    "Balayage paramétrique"
                }
            ),
            base_request: base_req.clone(),
            chain_mode: self.chain_mode,
            primary_axis: StudyAxisSpec {
                kind: self.axis1_kind,
                values: axis1_vals,
                couple_step_to_tau: self.couple_tau,
            },
            secondary_axis,
            use_bed_frame: self.use_bed_frame,
            enable_law_study: self.enable_law_study,
            law_families: LawFamily::all(),
            law_criterion: self.law_criterion,
            focus_transition: None,
            reference_policy: if self.axis1_kind == StudyAxisKind::Dt {
                ReferencePolicy::FirstPoint
            } else {
                ReferencePolicy::LastPoint
            },
            include_curves: true,
        }
    }
}

fn parse_axis_values(raw: &str, kind: StudyAxisKind) -> Vec<AxisValue> {
    let mut out = Vec::new();
    for tok in raw.split([',', ';', ' ']) {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        if kind == StudyAxisKind::VelocityWeight {
            if let Ok(v) = t.parse::<f64>() {
                if v.is_finite() && v >= 0.0 {
                    out.push(AxisValue::Float(v));
                }
            }
        } else if let Ok(v) = t.parse::<usize>() {
            out.push(AxisValue::Int(v));
        }
    }
    if out.is_empty() {
        out.push(AxisValue::Int(0));
        out.push(AxisValue::Int(40));
        out.push(AxisValue::Int(80));
    }
    out
}

fn law_color(family: LawFamily) -> Color32 {
    match family {
        LawFamily::Constant => Color32::from_rgb(148, 163, 184),
        LawFamily::Linear => Color32::from_rgb(158, 192, 220),
        LawFamily::Quadratic => Color32::from_rgb(238, 192, 120),
        LawFamily::Cubic => Color32::from_rgb(229, 152, 155),
        LawFamily::ExponentialRelax => Color32::from_rgb(125, 211, 189),
        LawFamily::Logarithmic => Color32::from_rgb(196, 181, 253),
        LawFamily::PowerRelax => Color32::from_rgb(245, 158, 11),
    }
}

pub fn draw_study_view(
    ui: &mut Ui,
    base_req: &SimulationRequest,
    study: &mut EguiStudyState,
    computing: bool,
    trigger_study_run: &mut bool,
) {
    ui.columns(2, |cols| {
        cols[0].heading("Atelier d'Études Paramétriques & Lois");
        cols[1].with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let btn = ui.add_enabled(
                !computing,
                egui::Button::new(if computing {
                    "Étude en cours..."
                } else {
                    "Lancer l'étude"
                }),
            );
            if btn.clicked() {
                *trigger_study_run = true;
            }
        });
    });
    ui.separator();

    egui::SidePanel::left("egui_study_builder")
        .resizable(true)
        .default_width(270.0)
        .show_inside(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.strong("1. Preset scientifique");
                let presets = catalog_study_presets();
                let current_name = presets
                    .iter()
                    .find(|p| p.id == study.preset_id)
                    .map(|p| p.name)
                    .unwrap_or("Personnalisé");
                egui::ComboBox::from_id_salt("egui_study_preset")
                    .selected_text(current_name)
                    .width(240.0)
                    .show_ui(ui, |ui| {
                        for p in presets {
                            if ui
                                .selectable_label(study.preset_id == p.id, p.name)
                                .clicked()
                            {
                                study.apply_preset(p.id, base_req);
                            }
                        }
                    });

                ui.horizontal(|ui| {
                    ui.label("Chaîne:");
                    ui.selectable_value(
                        &mut study.chain_mode,
                        StudyChainMode::Homogeneous,
                        "Hom",
                    );
                    ui.selectable_value(
                        &mut study.chain_mode,
                        StudyChainMode::Inhomogeneous,
                        "Inhom",
                    );
                    ui.selectable_value(&mut study.chain_mode, StudyChainMode::Both, "Les 2");
                });
                ui.checkbox(&mut study.enable_law_study, "Étude multi-lois p_{i->j}(t)");
                ui.checkbox(&mut study.use_bed_frame, "Repère principal du lit (PCA)");

                ui.separator();
                ui.strong("2. Axe primaire (X)");
                egui::ComboBox::from_id_salt("egui_study_axis1")
                    .selected_text(study.axis1_kind.display_name())
                    .width(240.0)
                    .show_ui(ui, |ui| {
                        for kind in [
                            StudyAxisKind::StartIndex,
                            StudyAxisKind::Nlt,
                            StudyAxisKind::Step,
                            StudyAxisKind::Dt,
                            StudyAxisKind::Tau,
                            StudyAxisKind::Method,
                            StudyAxisKind::SpatialResolution,
                            StudyAxisKind::VelocityWeight,
                            StudyAxisKind::PredStart,
                            StudyAxisKind::SpeciesMode,
                        ] {
                            ui.selectable_value(&mut study.axis1_kind, kind, kind.display_name());
                        }
                    });

                // Schema-driven gating: show ONLY fields valid for axis1_kind
                match study.axis1_kind {
                    StudyAxisKind::Method => {
                        ui.label("Méthodes comparées:");
                        let mut keys: Vec<String> =
                            study.selected_methods.keys().cloned().collect();
                        keys.sort();
                        for m in keys {
                            if let Some(on) = study.selected_methods.get_mut(&m) {
                                ui.checkbox(on, m);
                            }
                        }
                    }
                    StudyAxisKind::SpeciesMode => {
                        ui.label("Compare automatiquement Par espèce vs Sans espèces.");
                    }
                    _ => {
                        ui.label("Valeurs (virgules):");
                        ui.text_edit_singleline(&mut study.axis1_text);
                        if study.axis1_kind == StudyAxisKind::Tau {
                            ui.checkbox(
                                &mut study.couple_tau,
                                "Coupler step = tau et dt = tau/10",
                            );
                        }
                    }
                }

                ui.separator();
                ui.checkbox(&mut study.enable_axis2, "3. Grille 2D (Axe secondaire Y)");
                if study.enable_axis2 {
                    egui::ComboBox::from_id_salt("egui_study_axis2")
                        .selected_text(study.axis2_kind.display_name())
                        .width(240.0)
                        .show_ui(ui, |ui| {
                            for kind in [
                                StudyAxisKind::Nlt,
                                StudyAxisKind::StartIndex,
                                StudyAxisKind::Tau,
                                StudyAxisKind::Dt,
                            ] {
                                ui.selectable_value(
                                    &mut study.axis2_kind,
                                    kind,
                                    kind.display_name(),
                                );
                            }
                        });
                    ui.label("Valeurs Axe 2:");
                    ui.text_edit_singleline(&mut study.axis2_text);
                }

                if study.enable_law_study {
                    ui.separator();
                    ui.strong("Critère meilleure loi:");
                    egui::ComboBox::from_id_salt("egui_law_crit")
                        .selected_text(study.law_criterion.as_str())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut study.law_criterion,
                                LawSelectionCriterion::Aicc,
                                "AICc (parcimonie)",
                            );
                            ui.selectable_value(
                                &mut study.law_criterion,
                                LawSelectionCriterion::OutOfBlockRmse,
                                "Out-of-Block RMSE",
                            );
                            ui.selectable_value(
                                &mut study.law_criterion,
                                LawSelectionCriterion::Rmse,
                                "RMSE interne",
                            );
                            ui.selectable_value(
                                &mut study.law_criterion,
                                LawSelectionCriterion::R2,
                                "R^2 maximal",
                            );
                        });
                }
            });
        });

    egui::CentralPanel::default().show_inside(ui, |ui| {
        let Some(res) = &study.result else {
            ui.label(
                "Configurez l'étude à gauche (ou gardez le preset) puis cliquez sur « Lancer l'étude ».",
            );
            return;
        };

        ui.label(
            RichText::new(format!("{} — {}", res.title, res.subtitle)).strong(),
        );

        let avail_h = ui.available_height();
        let plot_h = (avail_h * 0.56).max(210.0);

        ui.columns(2, |cols| {
            // Left large plot: Law comparison or temporal RSD overlay
            if res.enable_law_study && !res.law_points.is_empty() {
                let sel = study.selected_cell;
                let lp = res
                    .law_points
                    .iter()
                    .find(|p| p.idx1 == sel.0 && p.idx2 == sel.1)
                    .unwrap_or(&res.law_points[0]);
                let cmp = &lp.comparison;
                cols[0].label(format!(
                    "Lois p({} -> {}) · start={}, NLT={} · Gagnante: {}",
                    cmp.src,
                    cmp.dst,
                    lp.start_index,
                    lp.nlt,
                    cmp.best_family.display_name()
                ));
                Plot::new("egui_study_laws_plot")
                    .legend(Legend::default())
                    .height(plot_h)
                    .show(&mut cols[0], |plot_ui| {
                        let raw: PlotPoints = cmp
                            .times
                            .iter()
                            .zip(cmp.values.iter())
                            .map(|(&t, &v)| [t, v])
                            .collect();
                        plot_ui.points(
                            Points::new(raw)
                                .name("Blocs P^(b)")
                                .color(Color32::WHITE)
                                .radius(4.8),
                        );
                        for c in &cmp.candidates {
                            let is_best = c.family == cmp.best_family;
                            let pts: PlotPoints = c
                                .dense_times
                                .iter()
                                .zip(c.dense_predicted.iter())
                                .map(|(&t, &v)| [t, v])
                                .collect();
                            let mut line = Line::new(pts)
                                .name(format!("{} (R2={:.2})", c.family.display_name(), c.r2))
                                .color(law_color(c.family))
                                .width(if is_best { 2.8 } else { 1.4 });
                            if !is_best {
                                line = line.style(LineStyle::Dashed { length: 5.0 });
                            }
                            plot_ui.line(line);
                        }
                    });
            } else {
                cols[0].label("Superposition temporelle RSD(t)");
                Plot::new("egui_study_rsd_plot")
                    .legend(Legend::default())
                    .height(plot_h)
                    .show(&mut cols[0], |plot_ui| {
                        if let Some(first) = res.curves.first() {
                            let dem_pts: PlotPoints = first
                                .times
                                .iter()
                                .zip(first.rsd_dem.iter())
                                .map(|(&t, &v)| [t, v])
                                .collect();
                            plot_ui.line(
                                Line::new(dem_pts)
                                    .name("DEM (réf.)")
                                    .color(Color32::WHITE)
                                    .width(2.5),
                            );
                        }
                        for (i, cv) in res.curves.iter().enumerate() {
                            let y = if res.chain_mode == StudyChainMode::Inhomogeneous
                                && !cv.rsd_inhom.is_empty()
                            {
                                &cv.rsd_inhom
                            } else {
                                &cv.rsd_hom
                            };
                            let pts: PlotPoints =
                                cv.times.iter().zip(y.iter()).map(|(&t, &v)| [t, v]).collect();
                            plot_ui.line(
                                Line::new(pts)
                                    .name(&cv.label)
                                    .color(STUDY_PALETTE[i % STUDY_PALETTE.len()])
                                    .width(1.8),
                            );
                        }
                    });
            }

            // Right large plot: 2D clickable grid OR 1D sensitivity curve
            if let Some(ax2) = &res.secondary_axis {
                cols[1].label(format!(
                    "Carte 2D ({} × {}) — cliquez une case:",
                    res.primary_axis.kind.id(),
                    ax2.kind.id()
                ));
                let n_rows = res.primary_axis.values.len().max(1);
                let n_cols = ax2.values.len().max(1);
                egui::Grid::new("egui_study_grid_2d")
                    .spacing([4.0, 4.0])
                    .show(&mut cols[1], |ui| {
                        ui.label("");
                        for v2 in &ax2.values {
                            ui.strong(format!("NLT={}", v2.label()));
                        }
                        ui.end_row();
                        for (r, v1) in res.primary_axis.values.iter().enumerate() {
                            ui.strong(format!("s={}", v1.label()));
                            for c in 0..n_cols {
                                let pt = res.points.iter().find(|p| p.idx1 == r && p.idx2 == c);
                                let lbl = match pt {
                                    Some(p) if p.error.is_none() => p
                                        .best_law_name
                                        .as_deref()
                                        .map(|s| format!("{s}\n{:.3}", p.rmse_rsd_hom))
                                        .unwrap_or_else(|| format!("{:.4}", p.rmse_rsd_hom)),
                                    _ => "—".to_string(),
                                };
                                let selected = study.selected_cell == (r, c);
                                if ui.selectable_label(selected, lbl).clicked() {
                                    study.selected_cell = (r, c);
                                }
                            }
                            ui.end_row();
                        }
                        let _ = n_rows;
                    });
            } else {
                cols[1].horizontal(|ui| {
                    ui.label("Métrique:");
                    egui::ComboBox::from_id_salt("egui_sens_metric")
                        .selected_text(study.sens_metric.label())
                        .show_ui(ui, |ui| {
                            for m in [
                                EguiSensMetric::RmseRsdHom,
                                EguiSensMetric::RmseTeneurHom,
                                EguiSensMetric::L1MeanHom,
                                EguiSensMetric::FrobeniusRel,
                                EguiSensMetric::NanRowsHom,
                                EguiSensMetric::LambdaMax,
                            ] {
                                ui.selectable_value(&mut study.sens_metric, m, m.label());
                            }
                        });
                });
                Plot::new("egui_study_sens_plot")
                    .legend(Legend::default())
                    .height(plot_h)
                    .show(&mut cols[1], |plot_ui| {
                        let pts: PlotPoints = res
                            .points
                            .iter()
                            .enumerate()
                            .filter(|(_, p)| p.error.is_none())
                            .map(|(i, p)| {
                                let x = if res.primary_axis.kind.is_numeric() {
                                    p.param1_num
                                } else {
                                    i as f64
                                };
                                let y = match study.sens_metric {
                                    EguiSensMetric::RmseRsdHom => p.rmse_rsd_hom,
                                    EguiSensMetric::RmseTeneurHom => p.rmse_teneur_hom,
                                    EguiSensMetric::L1MeanHom => p.l1_mean_hom,
                                    EguiSensMetric::FrobeniusRel => {
                                        p.frobenius_rel_err.unwrap_or(0.0)
                                    }
                                    EguiSensMetric::NanRowsHom => p.nan_rows_hom as f64,
                                    EguiSensMetric::LambdaMax => p.lambda_max,
                                };
                                [x, y]
                            })
                            .collect();
                        plot_ui.line(
                            Line::new(pts)
                                .name(study.sens_metric.label())
                                .color(Color32::from_rgb(125, 211, 189))
                                .width(2.2),
                        );
                    });
            }
        });

        ui.separator();
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::Grid::new("egui_study_table")
                .striped(true)
                .show(ui, |ui| {
                    ui.strong("Paramètre");
                    ui.strong("Vis./Cell");
                    ui.strong("NaN");
                    ui.strong("RMSE RSD");
                    ui.strong("RMSE Teneur");
                    ui.strong("Frobenius");
                    ui.strong("Meilleure loi");
                    ui.end_row();
                    for p in &res.points {
                        ui.label(if p.param2_label.is_empty() {
                            p.param1_label.clone()
                        } else {
                            format!("{} × {}", p.param1_label, p.param2_label)
                        });
                        ui.label(format!("{}/{}", p.n_visited, p.n_cells));
                        ui.label(format!("{}", p.nan_rows_hom));
                        ui.label(format!("{:.4}", p.rmse_rsd_hom));
                        ui.label(format!("{:.4}", p.rmse_teneur_hom));
                        ui.label(
                            p.frobenius_rel_err
                                .map(|v| format!("{v:.4}"))
                                .unwrap_or_else(|| "—".to_string()),
                        );
                        ui.label(p.best_law_name.as_deref().unwrap_or("—"));
                        ui.end_row();
                    }
                });
        });
    });
}
