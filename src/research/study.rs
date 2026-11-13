//! Dynamic parametric & multi-law study engine (`StudyRunner`).
//!
//! Accepts declarative [`StudySpec`] definitions (1-D parameter sweeps,
//! 2-D crossed grids such as `start_index x NLT`, method-vs-RSD sweeps, and
//! multi-family `p_{i->j}(t)` law-approximation studies) and executes them over
//! [`ResearchSessionCache`] without redundant spatial partitioning.

use std::collections::{BTreeMap, HashMap};

use crate::data::DemFrame;
use crate::error::{DemMcmError, Result};
use crate::gui::metrics_eval::vec_f64_json;
use crate::gui::state::SimulationRequest;
use crate::io::json::JsonValue;
use crate::research::laws::{
    compare_transition_laws, LawFamily, LawSelectionCriterion, TransitionLawComparison,
};
use crate::research::session::{
    frobenius_relative_error, ResearchSessionCache, TemporalEvalOutput, TemporalEvalParams,
};

/// Parametric axis that can be swept dynamically by [`StudyRunner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyAxisKind {
    StartIndex,
    Nlt,
    Step,
    Dt,
    Tau,
    PredStart,
    Method,
    SpatialResolution,
    VelocityWeight,
    SpeciesMode,
}

impl StudyAxisKind {
    pub fn id(self) -> &'static str {
        match self {
            Self::StartIndex => "start_index",
            Self::Nlt => "nlt",
            Self::Step => "step",
            Self::Dt => "dt",
            Self::Tau => "tau",
            Self::PredStart => "pred_start",
            Self::Method => "method",
            Self::SpatialResolution => "spatial_resolution",
            Self::VelocityWeight => "velocity_weight",
            Self::SpeciesMode => "species_mode",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::StartIndex => "Départ apprentissage (start_index)",
            Self::Nlt => "Nombre de blocs / instants (NLT)",
            Self::Step => "Écart inter-blocs (step)",
            Self::Dt => "Pas intra-bloc (dt)",
            Self::Tau => "Horizon Markov (τ)",
            Self::PredStart => "Début de prédiction",
            Self::Method => "Méthode de découpage spatial",
            Self::SpatialResolution => "Résolution du maillage",
            Self::VelocityWeight => "Poids cinématique (w_V)",
            Self::SpeciesMode => "Distinction d'espèces",
        }
    }

    pub fn from_id(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "start" | "start_index" => Some(Self::StartIndex),
            "nlt" => Some(Self::Nlt),
            "step" => Some(Self::Step),
            "dt" => Some(Self::Dt),
            "tau" => Some(Self::Tau),
            "pred_start" => Some(Self::PredStart),
            "method" => Some(Self::Method),
            "spatial_resolution" | "resolution" | "n_cells" => Some(Self::SpatialResolution),
            "velocity_weight" | "w_v" => Some(Self::VelocityWeight),
            "species_mode" | "no_species" => Some(Self::SpeciesMode),
            _ => None,
        }
    }

    pub fn is_numeric(self) -> bool {
        !matches!(self, Self::Method | Self::SpeciesMode)
    }
}

/// One typed point along a [`StudyAxisSpec`].
#[derive(Debug, Clone)]
pub enum AxisValue {
    Int(usize),
    Float(f64),
    Text(String),
    Bool(bool),
}

impl AxisValue {
    pub fn label(&self) -> String {
        match self {
            Self::Int(v) => v.to_string(),
            Self::Float(v) => format!("{v:.3}")
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string(),
            Self::Text(s) => s.clone(),
            Self::Bool(b) => if *b { "Sans espèces" } else { "Par espèce" }.to_string(),
        }
    }

    pub fn as_f64(&self, fallback_idx: usize) -> f64 {
        match self {
            Self::Int(v) => *v as f64,
            Self::Float(v) => *v,
            Self::Bool(b) => if *b { 1.0 } else { 0.0 },
            Self::Text(_) => fallback_idx as f64,
        }
    }

    pub fn to_json(&self) -> JsonValue {
        match self {
            Self::Int(v) => JsonValue::Number(*v as f64),
            Self::Float(v) => JsonValue::Number(*v),
            Self::Text(s) => JsonValue::String(s.clone()),
            Self::Bool(b) => JsonValue::Bool(*b),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StudyAxisSpec {
    pub kind: StudyAxisKind,
    pub values: Vec<AxisValue>,
    /// When `kind == Tau`, optionally couple `step = tau` and `dt = max(1, tau / 10)`.
    pub couple_step_to_tau: bool,
}

impl StudyAxisSpec {
    pub fn from_json(v: &JsonValue) -> Option<Self> {
        let kind_str = v.get("kind").and_then(|x| x.as_str())?;
        let kind = StudyAxisKind::from_id(kind_str)?;
        let couple_step_to_tau = v
            .get("couple_step_to_tau")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);

        let mut values = Vec::new();
        if let Some(arr) = v.get("values").and_then(|x| x.as_array()) {
            for item in arr {
                match kind {
                    StudyAxisKind::Method => {
                        if let Some(s) = item.as_str() {
                            values.push(AxisValue::Text(s.to_string()));
                        }
                    }
                    StudyAxisKind::SpeciesMode => {
                        if let Some(b) = item.as_bool() {
                            values.push(AxisValue::Bool(b));
                        } else if let Some(s) = item.as_str() {
                            let no_sp = s == "no_species" || s == "true" || s == "all";
                            values.push(AxisValue::Bool(no_sp));
                        } else if let Some(n) = item.as_f64() {
                            values.push(AxisValue::Bool(n > 0.5));
                        }
                    }
                    StudyAxisKind::VelocityWeight => {
                        if let Some(n) = item.as_f64() {
                            values.push(AxisValue::Float(n.max(0.0)));
                        }
                    }
                    _ => {
                        if let Some(n) = item.as_f64() {
                            values.push(AxisValue::Int(n.round().max(0.0) as usize));
                        }
                    }
                }
            }
        }
        if values.is_empty() {
            return None;
        }
        Some(Self {
            kind,
            values,
            couple_step_to_tau,
        })
    }

    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("kind", JsonValue::String(self.kind.id().to_string()));
        o.insert(
            "display_name",
            JsonValue::String(self.kind.display_name().to_string()),
        );
        o.insert("is_numeric", JsonValue::Bool(self.kind.is_numeric()));
        o.insert("couple_step_to_tau", JsonValue::Bool(self.couple_step_to_tau));
        o.insert(
            "values",
            JsonValue::Array(self.values.iter().map(AxisValue::to_json).collect()),
        );
        o.insert(
            "labels",
            JsonValue::Array(
                self.values
                    .iter()
                    .map(|val| JsonValue::String(val.label()))
                    .collect(),
            ),
        );
        o
    }
}

/// Which Markov chain family is evaluated in the study.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudyChainMode {
    Homogeneous,
    Inhomogeneous,
    Both,
}

impl StudyChainMode {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "inhom" | "inhomogeneous" => Self::Inhomogeneous,
            "both" => Self::Both,
            _ => Self::Homogeneous,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Homogeneous => "homogeneous",
            Self::Inhomogeneous => "inhomogeneous",
            Self::Both => "both",
        }
    }
}

/// Reference matrix policy for Frobenius relative convergence error (`||P - P_ref||_F / ||P_ref||_F`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferencePolicy {
    LastPoint,
    FirstPoint,
    None,
}

impl ReferencePolicy {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "first" | "first_point" => Self::FirstPoint,
            "none" => Self::None,
            _ => Self::LastPoint,
        }
    }
}

/// Full declarative definition of a research study.
#[derive(Debug, Clone)]
pub struct StudySpec {
    pub study_id: String,
    pub title: String,
    pub subtitle: String,
    pub base_request: SimulationRequest,
    pub chain_mode: StudyChainMode,
    pub primary_axis: StudyAxisSpec,
    pub secondary_axis: Option<StudyAxisSpec>,
    pub use_bed_frame: bool,
    pub enable_law_study: bool,
    pub law_families: Vec<LawFamily>,
    pub law_criterion: LawSelectionCriterion,
    pub focus_transition: Option<(usize, usize)>,
    pub reference_policy: ReferencePolicy,
    pub include_curves: bool,
}

impl StudySpec {
    pub fn from_json(v: &JsonValue) -> Result<Self> {
        let base_request = v
            .get("base_request")
            .map(SimulationRequest::from_json)
            .unwrap_or_else(|| SimulationRequest::from_json(v));

        let primary_axis = v
            .get("primary_axis")
            .and_then(StudyAxisSpec::from_json)
            .ok_or_else(|| {
                DemMcmError::InvalidArgument(
                    "StudySpec missing valid primary_axis with non-empty values".to_string(),
                )
            })?;

        let secondary_axis = v.get("secondary_axis").and_then(StudyAxisSpec::from_json);

        let study_id = v
            .get("study_id")
            .and_then(|x| x.as_str())
            .unwrap_or("custom")
            .to_string();
        let title = v
            .get("title")
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| {
                if let Some(ref ax2) = secondary_axis {
                    format!(
                        "Étude 2D · {} × {}",
                        primary_axis.kind.display_name(),
                        ax2.kind.display_name()
                    )
                } else {
                    format!("Étude · {}", primary_axis.kind.display_name())
                }
            });
        let subtitle = v
            .get("subtitle")
            .and_then(|x| x.as_str())
            .unwrap_or("Sensibilité paramétrique et dynamique")
            .to_string();

        let chain_mode = v
            .get("chain_mode")
            .and_then(|x| x.as_str())
            .map(StudyChainMode::from_str_lossy)
            .unwrap_or(StudyChainMode::Homogeneous);
        let use_bed_frame = v
            .get("use_bed_frame")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let enable_law_study = v
            .get("enable_law_study")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        let law_families: Vec<LawFamily> = v
            .get("law_families")
            .and_then(|x| x.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().and_then(LawFamily::from_id))
                    .collect()
            })
            .filter(|list: &Vec<LawFamily>| !list.is_empty())
            .unwrap_or_else(LawFamily::all);
        let law_criterion = v
            .get("law_criterion")
            .and_then(|x| x.as_str())
            .map(LawSelectionCriterion::from_str_lossy)
            .unwrap_or(LawSelectionCriterion::Aicc);
        let focus_transition = v.get("focus_transition").and_then(|ft| {
            let src = ft.get("src").and_then(|x| x.as_f64()).map(|n| n as usize)?;
            let dst = ft.get("dst").and_then(|x| x.as_f64()).map(|n| n as usize)?;
            Some((src, dst))
        });
        let reference_policy = v
            .get("reference_policy")
            .and_then(|x| x.as_str())
            .map(ReferencePolicy::from_str_lossy)
            .unwrap_or(ReferencePolicy::LastPoint);
        let include_curves = v
            .get("include_curves")
            .and_then(|x| x.as_bool())
            .unwrap_or(true);

        Ok(Self {
            study_id,
            title,
            subtitle,
            base_request,
            chain_mode,
            primary_axis,
            secondary_axis,
            use_bed_frame,
            enable_law_study,
            law_families,
            law_criterion,
            focus_transition,
            reference_policy,
            include_curves,
        })
    }

    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("study_id", JsonValue::String(self.study_id.clone()));
        o.insert("title", JsonValue::String(self.title.clone()));
        o.insert("subtitle", JsonValue::String(self.subtitle.clone()));
        o.insert("chain_mode", JsonValue::String(self.chain_mode.as_str().to_string()));
        o.insert("use_bed_frame", JsonValue::Bool(self.use_bed_frame));
        o.insert("enable_law_study", JsonValue::Bool(self.enable_law_study));
        o.insert(
            "law_families",
            JsonValue::Array(
                self.law_families
                    .iter()
                    .map(|f| JsonValue::String(f.id().to_string()))
                    .collect(),
            ),
        );
        o.insert(
            "law_criterion",
            JsonValue::String(self.law_criterion.as_str().to_string()),
        );
        o.insert("primary_axis", self.primary_axis.to_json());
        o.insert(
            "secondary_axis",
            self.secondary_axis
                .as_ref()
                .map(StudyAxisSpec::to_json)
                .unwrap_or(JsonValue::Null),
        );
        o.insert(
            "focus_transition",
            match self.focus_transition {
                Some((s, d)) => {
                    let mut ft = JsonValue::object();
                    ft.insert("src", JsonValue::Number(s as f64));
                    ft.insert("dst", JsonValue::Number(d as f64));
                    ft
                }
                None => JsonValue::Null,
            },
        );
        o
    }
}

/// Scalar metrics recorded at one `(idx1[, idx2])` point of the study.
#[derive(Debug, Clone)]
pub struct StudyPointMetrics {
    pub idx1: usize,
    pub idx2: usize,
    pub param1_label: String,
    pub param2_label: String,
    pub param1_num: f64,
    pub param2_num: f64,
    pub method: String,
    pub n_cells: usize,
    pub n_visited: usize,
    pub start_index: usize,
    pub nlt: usize,
    pub step: usize,
    pub dt: usize,
    pub tau: usize,
    pub n_pairs_used: usize,
    pub nan_rows_hom: usize,
    pub nan_rows_inhom_mean: f64,
    pub rmse_rsd_hom: f64,
    pub mae_rsd_hom: f64,
    pub max_rsd_err_hom: f64,
    pub rmse_teneur_hom: f64,
    pub l1_mean_hom: f64,
    pub rms_disc_hom: f64,
    pub t50_dem: Option<f64>,
    pub t50_hom: Option<f64>,
    pub diag_mean: f64,
    pub lambda_max: f64,
    pub frobenius_rel_err: Option<f64>,
    pub rmse_rsd_inhom: Option<f64>,
    pub rmse_teneur_inhom: Option<f64>,
    pub l1_mean_inhom: Option<f64>,
    pub t50_inhom: Option<f64>,
    pub best_law_family: Option<String>,
    pub best_law_name: Option<String>,
    pub best_law_r2: Option<f64>,
    pub best_law_rmse: Option<f64>,
    pub best_law_aicc: Option<f64>,
    pub best_law_oob_rmse: Option<f64>,
    pub error: Option<String>,
}

impl StudyPointMetrics {
    pub fn to_json(&self) -> JsonValue {
        let opt_num = |o: Option<f64>| o.map(JsonValue::Number).unwrap_or(JsonValue::Null);
        let opt_str = |o: &Option<String>| {
            o.as_ref()
                .map(|s| JsonValue::String(s.clone()))
                .unwrap_or(JsonValue::Null)
        };
        let mut o = JsonValue::object();
        o.insert("idx1", JsonValue::Number(self.idx1 as f64));
        o.insert("idx2", JsonValue::Number(self.idx2 as f64));
        o.insert("param1_label", JsonValue::String(self.param1_label.clone()));
        o.insert("param2_label", JsonValue::String(self.param2_label.clone()));
        o.insert("param1_num", JsonValue::Number(self.param1_num));
        o.insert("param2_num", JsonValue::Number(self.param2_num));
        o.insert("method", JsonValue::String(self.method.clone()));
        o.insert("n_cells", JsonValue::Number(self.n_cells as f64));
        o.insert("n_visited", JsonValue::Number(self.n_visited as f64));
        o.insert("start_index", JsonValue::Number(self.start_index as f64));
        o.insert("nlt", JsonValue::Number(self.nlt as f64));
        o.insert("step", JsonValue::Number(self.step as f64));
        o.insert("dt", JsonValue::Number(self.dt as f64));
        o.insert("tau", JsonValue::Number(self.tau as f64));
        o.insert("n_pairs_used", JsonValue::Number(self.n_pairs_used as f64));
        o.insert("nan_rows_hom", JsonValue::Number(self.nan_rows_hom as f64));
        o.insert(
            "nan_rows_inhom_mean",
            JsonValue::Number(self.nan_rows_inhom_mean),
        );
        o.insert("rmse_rsd_hom", JsonValue::Number(self.rmse_rsd_hom));
        o.insert("mae_rsd_hom", JsonValue::Number(self.mae_rsd_hom));
        o.insert("max_rsd_err_hom", JsonValue::Number(self.max_rsd_err_hom));
        o.insert("rmse_teneur_hom", JsonValue::Number(self.rmse_teneur_hom));
        o.insert("l1_mean_hom", JsonValue::Number(self.l1_mean_hom));
        o.insert("rms_disc_hom", JsonValue::Number(self.rms_disc_hom));
        o.insert("t50_dem", opt_num(self.t50_dem));
        o.insert("t50_hom", opt_num(self.t50_hom));
        o.insert("diag_mean", JsonValue::Number(self.diag_mean));
        o.insert("lambda_max", JsonValue::Number(self.lambda_max));
        o.insert("frobenius_rel_err", opt_num(self.frobenius_rel_err));
        o.insert("rmse_rsd_inhom", opt_num(self.rmse_rsd_inhom));
        o.insert("rmse_teneur_inhom", opt_num(self.rmse_teneur_inhom));
        o.insert("l1_mean_inhom", opt_num(self.l1_mean_inhom));
        o.insert("t50_inhom", opt_num(self.t50_inhom));
        o.insert("best_law_family", opt_str(&self.best_law_family));
        o.insert("best_law_name", opt_str(&self.best_law_name));
        o.insert("best_law_r2", opt_num(self.best_law_r2));
        o.insert("best_law_rmse", opt_num(self.best_law_rmse));
        o.insert("best_law_aicc", opt_num(self.best_law_aicc));
        o.insert("best_law_oob_rmse", opt_num(self.best_law_oob_rmse));
        o.insert("error", opt_str(&self.error));
        o
    }
}

/// Time-series curve overlay for one point of the study (`RSD(t)` DEM vs Markov).
#[derive(Debug, Clone)]
pub struct StudyCurveSeries {
    pub idx1: usize,
    pub idx2: usize,
    pub label: String,
    pub times: Vec<f64>,
    pub rsd_dem: Vec<f64>,
    pub rsd_hom: Vec<f64>,
    pub rsd_inhom: Vec<f64>,
}

impl StudyCurveSeries {
    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("idx1", JsonValue::Number(self.idx1 as f64));
        o.insert("idx2", JsonValue::Number(self.idx2 as f64));
        o.insert("label", JsonValue::String(self.label.clone()));
        o.insert("times", vec_f64_json(&self.times));
        o.insert("rsd_dem", vec_f64_json(&self.rsd_dem));
        o.insert("rsd_hom", vec_f64_json(&self.rsd_hom));
        o.insert("rsd_inhom", vec_f64_json(&self.rsd_inhom));
        o
    }
}

/// Multi-law evaluation at one `(idx1, idx2)` point (`(start_index, NLT)`).
#[derive(Debug, Clone)]
pub struct StudyLawPointResult {
    pub idx1: usize,
    pub idx2: usize,
    pub param1_label: String,
    pub param2_label: String,
    pub start_index: usize,
    pub nlt: usize,
    pub comparison: TransitionLawComparison,
}

impl StudyLawPointResult {
    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("idx1", JsonValue::Number(self.idx1 as f64));
        o.insert("idx2", JsonValue::Number(self.idx2 as f64));
        o.insert("param1_label", JsonValue::String(self.param1_label.clone()));
        o.insert("param2_label", JsonValue::String(self.param2_label.clone()));
        o.insert("start_index", JsonValue::Number(self.start_index as f64));
        o.insert("nlt", JsonValue::Number(self.nlt as f64));
        o.insert("comparison", self.comparison.to_json());
        o
    }
}

/// Complete output bundle returned by [`run_study`].
#[derive(Debug, Clone)]
pub struct StudyResultBundle {
    pub study_id: String,
    pub title: String,
    pub subtitle: String,
    pub dataset: String,
    pub method: String,
    pub chain_mode: StudyChainMode,
    pub enable_law_study: bool,
    pub primary_axis: StudyAxisSpec,
    pub secondary_axis: Option<StudyAxisSpec>,
    pub points: Vec<StudyPointMetrics>,
    pub curves: Vec<StudyCurveSeries>,
    pub law_points: Vec<StudyLawPointResult>,
    pub available_transitions: Vec<(usize, usize, f64)>,
    pub active_transition: Option<(usize, usize)>,
}

impl StudyResultBundle {
    pub fn to_json(&self) -> JsonValue {
        let mut o = JsonValue::object();
        o.insert("study_id", JsonValue::String(self.study_id.clone()));
        o.insert("title", JsonValue::String(self.title.clone()));
        o.insert("subtitle", JsonValue::String(self.subtitle.clone()));
        o.insert("dataset", JsonValue::String(self.dataset.clone()));
        o.insert("method", JsonValue::String(self.method.clone()));
        o.insert(
            "chain_mode",
            JsonValue::String(self.chain_mode.as_str().to_string()),
        );
        o.insert("enable_law_study", JsonValue::Bool(self.enable_law_study));
        o.insert("primary_axis", self.primary_axis.to_json());
        o.insert(
            "secondary_axis",
            self.secondary_axis
                .as_ref()
                .map(StudyAxisSpec::to_json)
                .unwrap_or(JsonValue::Null),
        );
        o.insert(
            "points",
            JsonValue::Array(self.points.iter().map(StudyPointMetrics::to_json).collect()),
        );
        o.insert(
            "curves",
            JsonValue::Array(self.curves.iter().map(StudyCurveSeries::to_json).collect()),
        );
        o.insert(
            "law_points",
            JsonValue::Array(
                self.law_points
                    .iter()
                    .map(StudyLawPointResult::to_json)
                    .collect(),
            ),
        );
        o.insert(
            "available_transitions",
            JsonValue::Array(
                self.available_transitions
                    .iter()
                    .map(|&(s, d, amp)| {
                        JsonValue::Array(vec![
                            JsonValue::Number(s as f64),
                            JsonValue::Number(d as f64),
                            JsonValue::Number(amp),
                        ])
                    })
                    .collect(),
            ),
        );
        o.insert(
            "active_transition",
            match self.active_transition {
                Some((s, d)) => {
                    let mut ft = JsonValue::object();
                    ft.insert("src", JsonValue::Number(s as f64));
                    ft.insert("dst", JsonValue::Number(d as f64));
                    ft
                }
                None => JsonValue::Null,
            },
        );
        o
    }
}

fn apply_axis_value(
    kind: StudyAxisKind,
    val: &AxisValue,
    couple_step_to_tau: bool,
    req: &mut SimulationRequest,
    eval: &mut TemporalEvalParams,
) {
    match kind {
        StudyAxisKind::StartIndex => {
            if let AxisValue::Int(v) = val {
                eval.start_index = *v;
                req.start_index = *v;
            }
        }
        StudyAxisKind::Nlt => {
            if let AxisValue::Int(v) = val {
                eval.nlt = (*v).max(1);
                req.nlt = eval.nlt;
            }
        }
        StudyAxisKind::Step => {
            if let AxisValue::Int(v) = val {
                eval.step = (*v).max(1);
                req.step = eval.step;
            }
        }
        StudyAxisKind::Dt => {
            if let AxisValue::Int(v) = val {
                eval.dt = (*v).max(1);
                req.dt = eval.dt;
            }
        }
        StudyAxisKind::Tau => {
            if let AxisValue::Int(v) = val {
                let tau = (*v).max(1);
                eval.tau = tau;
                req.tau = tau;
                if couple_step_to_tau {
                    eval.step = tau;
                    eval.dt = (tau / 10).max(1);
                    req.step = eval.step;
                    req.dt = eval.dt;
                }
            }
        }
        StudyAxisKind::PredStart => {
            if let AxisValue::Int(v) = val {
                eval.predict_full_duration = *v == 0;
                req.predict_full_duration = eval.predict_full_duration;
            }
        }
        StudyAxisKind::Method => {
            if let AxisValue::Text(m) = val {
                req.method = m.clone();
                req.kwargs = SimulationRequest::default_kwargs_for(m);
            }
        }
        StudyAxisKind::SpatialResolution => {
            let n = match val {
                AxisValue::Int(v) => *v,
                AxisValue::Float(v) => v.round().max(1.0) as usize,
                _ => 12,
            }
            .max(2);
            apply_resolution_to_kwargs(&req.method, &mut req.kwargs, n);
        }
        StudyAxisKind::VelocityWeight => {
            let w = match val {
                AxisValue::Float(v) => *v,
                AxisValue::Int(v) => *v as f64,
                _ => 0.5,
            };
            req.kwargs
                .insert("velocity_weight".to_string(), JsonValue::Number(w));
        }
        StudyAxisKind::SpeciesMode => {
            if let AxisValue::Bool(no_sp) = val {
                eval.no_species = *no_sp;
                req.no_species = *no_sp;
            }
        }
    }
}

fn apply_resolution_to_kwargs(
    method: &str,
    kwargs: &mut HashMap<String, JsonValue>,
    target_n: usize,
) {
    match method {
        "cartesian" | "quantile" => {
            // Interpret target_n as nx = ny = target_n with nz = 2 (or direct grid if small)
            let side = target_n.clamp(2, 10);
            kwargs.insert("nx".to_string(), JsonValue::Number(side as f64));
            kwargs.insert("ny".to_string(), JsonValue::Number(side as f64));
            kwargs.insert("nz".to_string(), JsonValue::Number(2.0));
        }
        "cylindrical" => {
            let nr = target_n.clamp(2, 8);
            kwargs.insert("nr".to_string(), JsonValue::Number(nr as f64));
            kwargs.insert("ntheta".to_string(), JsonValue::Number((nr * 2).min(16) as f64));
        }
        "octree" => {
            kwargs.insert(
                "max_particles".to_string(),
                JsonValue::Number(target_n.max(15) as f64),
            );
        }
        "spectral_biclustering" | "voronoi" | "physics" | "physics_full_vel" | "spectral"
        | "gmm" => {
            kwargs.insert(
                "n_cells".to_string(),
                JsonValue::Number(target_n.clamp(2, 64) as f64),
            );
        }
        "dbscan" => {
            let eps = (1.5 / (target_n as f64).sqrt()).clamp(0.12, 0.9);
            kwargs.insert("eps".to_string(), JsonValue::Number(eps));
        }
        _ => {}
    }
}

fn choose_significant_transitions(
    blocks: &[crate::linalg::Matrix],
    threshold: f64,
    max_items: usize,
) -> Vec<(usize, usize, f64)> {
    if blocks.is_empty() {
        return Vec::new();
    }
    let n = blocks[0].rows;
    let mut off_diag: Vec<(usize, usize, f64)> = Vec::new();
    let mut diag: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let mut mx = 0.0f64;
            let mut mn = 1.0f64;
            for b in blocks {
                let v = b.get(i, j);
                if v.is_finite() {
                    mx = mx.max(v);
                    mn = mn.min(v);
                }
            }
            if mx > threshold {
                let score = (mx - mn) * 0.7 + mx * 0.3;
                if i != j {
                    off_diag.push((i, j, score));
                } else {
                    diag.push((i, j, score));
                }
            }
        }
    }
    off_diag.sort_by(|a, b| b.2.total_cmp(&a.2));
    diag.sort_by(|a, b| b.2.total_cmp(&a.2));
    off_diag.extend(diag);
    off_diag.truncate(max_items);
    off_diag
}

fn downsample_curve(
    times: &[i64],
    a: &[f64],
    b: &[f64],
    c: &[f64],
    max_pts: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = times.len();
    if n <= max_pts {
        return (
            times.iter().map(|&t| t as f64).collect(),
            a.to_vec(),
            b.to_vec(),
            c.to_vec(),
        );
    }
    let stride = (n / max_pts).max(1);
    let mut ot = Vec::new();
    let mut oa = Vec::new();
    let mut ob = Vec::new();
    let mut oc = Vec::new();
    for i in (0..n).step_by(stride) {
        ot.push(times[i] as f64);
        oa.push(*a.get(i).unwrap_or(&0.0));
        ob.push(*b.get(i).unwrap_or(&0.0));
        oc.push(*c.get(i).unwrap_or(&0.0));
    }
    if ot.last().copied().unwrap_or(-1.0) != (times[n - 1] as f64) {
        ot.push(times[n - 1] as f64);
        oa.push(*a.last().unwrap_or(&0.0));
        ob.push(*b.last().unwrap_or(&0.0));
        oc.push(*c.last().unwrap_or(&0.0));
    }
    (ot, oa, ob, oc)
}

/// Execute any 1-D or 2-D dynamic research study over cached spatial sessions.
pub fn run_study(
    spec: &StudySpec,
    session_cache: &ResearchSessionCache,
    frames: &BTreeMap<usize, DemFrame>,
) -> Result<StudyResultBundle> {
    if frames.is_empty() {
        return Err(DemMcmError::MissingData(
            "Empty DEM frame collection".to_string(),
        ));
    }
    let build_inhom = spec.chain_mode != StudyChainMode::Homogeneous || spec.enable_law_study;
    let target_is_large = spec.base_request.target_species == "large";

    let sec_values: Vec<Option<&AxisValue>> = match &spec.secondary_axis {
        Some(ax2) => ax2.values.iter().map(Some).collect(),
        None => vec![None],
    };

    struct RawPointRecord {
        idx1: usize,
        idx2: usize,
        p1: AxisValue,
        p2_label: String,
        p2_num: f64,
        req: SimulationRequest,
        n_cells: usize,
        eval: Option<TemporalEvalOutput>,
        err: Option<String>,
    }

    let mut records: Vec<RawPointRecord> =
        Vec::with_capacity(spec.primary_axis.values.len() * sec_values.len());

    for (idx1, val1) in spec.primary_axis.values.iter().enumerate() {
        for (idx2, val2_opt) in sec_values.iter().enumerate() {
            let mut req = spec.base_request.clone();
            let mut eval_params = TemporalEvalParams {
                nlt: req.nlt.max(1),
                step: req.step.max(1),
                dt: req.dt.max(1),
                tau: req.tau.max(1),
                start_index: req.start_index,
                no_species: req.no_species,
                target_species: req.target_species.clone(),
                predict_full_duration: req.predict_full_duration,
                build_inhomogeneous: build_inhom,
            };

            apply_axis_value(
                spec.primary_axis.kind,
                val1,
                spec.primary_axis.couple_step_to_tau,
                &mut req,
                &mut eval_params,
            );
            let (p2_label, p2_num) = if let (Some(ax2), Some(val2)) = (&spec.secondary_axis, val2_opt) {
                apply_axis_value(
                    ax2.kind,
                    val2,
                    ax2.couple_step_to_tau,
                    &mut req,
                    &mut eval_params,
                );
                (val2.label(), val2.as_f64(idx2))
            } else {
                ("-".to_string(), 0.0)
            };

            let session_res = session_cache.get_or_build(
                &req.dataset,
                &req.method,
                &req.kwargs,
                req.permanent_start,
                spec.use_bed_frame,
                frames,
            );

            match session_res {
                Ok(session) => {
                    let n_cells = session.n_cells;
                    match session.evaluate_temporal(&eval_params) {
                        Ok(eval_out) => records.push(RawPointRecord {
                            idx1,
                            idx2,
                            p1: val1.clone(),
                            p2_label,
                            p2_num,
                            req,
                            n_cells,
                            eval: Some(eval_out),
                            err: None,
                        }),
                        Err(e) => records.push(RawPointRecord {
                            idx1,
                            idx2,
                            p1: val1.clone(),
                            p2_label,
                            p2_num,
                            req,
                            n_cells,
                            eval: None,
                            err: Some(e.to_string()),
                        }),
                    }
                }
                Err(e) => records.push(RawPointRecord {
                    idx1,
                    idx2,
                    p1: val1.clone(),
                    p2_label,
                    p2_num,
                    req,
                    n_cells: 0,
                    eval: None,
                    err: Some(e.to_string()),
                }),
            }
        }
    }

    // Reference P matrix for Frobenius relative error (Doucet et al., 2008)
    let ref_matrix = match spec.reference_policy {
        ReferencePolicy::None => None,
        ReferencePolicy::FirstPoint => records
            .iter()
            .find_map(|r| r.eval.as_ref())
            .map(|e| e.p_hom_raw_target(target_is_large).clone()),
        ReferencePolicy::LastPoint => records
            .iter()
            .rev()
            .find_map(|r| r.eval.as_ref())
            .map(|e| e.p_hom_raw_target(target_is_large).clone()),
    };

    // Determine focus transition for multi-law comparison when law study is enabled
    let mut available_transitions: Vec<(usize, usize, f64)> = Vec::new();
    let mut active_transition = spec.focus_transition;
    if spec.enable_law_study {
        // Choose transition candidates from the record with the most blocks so every cell is meaningful
        if let Some(rich_eval) = records
            .iter()
            .filter_map(|r| r.eval.as_ref())
            .max_by_key(|e| e.p_blocks_clean_target(target_is_large).len())
        {
            let blocks = rich_eval.p_blocks_clean_target(target_is_large);
            available_transitions = choose_significant_transitions(blocks, 0.01, 24);
            if active_transition.is_none() {
                active_transition = available_transitions.first().map(|&(s, d, _)| (s, d));
            }
        }
    }

    let mut points = Vec::with_capacity(records.len());
    let mut curves = Vec::new();
    let mut law_points = Vec::new();

    for rec in &records {
        let Some(ref ev) = rec.eval else {
            points.push(StudyPointMetrics {
                idx1: rec.idx1,
                idx2: rec.idx2,
                param1_label: rec.p1.label(),
                param2_label: rec.p2_label.clone(),
                param1_num: rec.p1.as_f64(rec.idx1),
                param2_num: rec.p2_num,
                method: rec.req.method.clone(),
                n_cells: rec.n_cells,
                n_visited: 0,
                start_index: rec.req.start_index,
                nlt: rec.req.nlt,
                step: rec.req.step,
                dt: rec.req.dt,
                tau: rec.req.tau,
                n_pairs_used: 0,
                nan_rows_hom: 0,
                nan_rows_inhom_mean: 0.0,
                rmse_rsd_hom: 0.0,
                mae_rsd_hom: 0.0,
                max_rsd_err_hom: 0.0,
                rmse_teneur_hom: 0.0,
                l1_mean_hom: 0.0,
                rms_disc_hom: 0.0,
                t50_dem: None,
                t50_hom: None,
                diag_mean: 0.0,
                lambda_max: 0.0,
                frobenius_rel_err: None,
                rmse_rsd_inhom: None,
                rmse_teneur_inhom: None,
                l1_mean_inhom: None,
                t50_inhom: None,
                best_law_family: None,
                best_law_name: None,
                best_law_r2: None,
                best_law_rmse: None,
                best_law_aicc: None,
                best_law_oob_rmse: None,
                error: rec.err.clone(),
            });
            continue;
        };

        let frob = ref_matrix.as_ref().and_then(|p_ref| {
            let p_cur = ev.p_hom_raw_target(target_is_large);
            if p_cur.rows == p_ref.rows && p_cur.cols == p_ref.cols {
                Some(frobenius_relative_error(p_cur, p_ref))
            } else {
                None
            }
        });

        let mut best_law_family = None;
        let mut best_law_name = None;
        let mut best_law_r2 = None;
        let mut best_law_rmse = None;
        let mut best_law_aicc = None;
        let mut best_law_oob_rmse = None;

        if spec.enable_law_study {
            if let Some((src, dst)) = active_transition {
                let blocks = ev.p_blocks_clean_target(target_is_large);
                if !blocks.is_empty() && src < blocks[0].rows && dst < blocks[0].cols {
                    let values: Vec<f64> = blocks
                        .iter()
                        .map(|b| {
                            let v = b.get(src, dst);
                            if v.is_finite() { v } else { 0.0 }
                        })
                        .collect();
                    let n_b = values.len().min(ev.block_times.len());
                    if n_b >= 1 {
                        if let Ok(comp) = compare_transition_laws(
                            src,
                            dst,
                            &ev.block_times[..n_b],
                            &values[..n_b],
                            &spec.law_families,
                            spec.law_criterion,
                        ) {
                            if let Some(best) = comp.best_candidate() {
                                best_law_family = Some(best.family.id().to_string());
                                best_law_name = Some(best.family.display_name().to_string());
                                best_law_r2 = Some(best.r2);
                                best_law_rmse = Some(best.rmse);
                                best_law_aicc = Some(best.aicc);
                                best_law_oob_rmse = best.out_of_block_rmse;
                            }
                            law_points.push(StudyLawPointResult {
                                idx1: rec.idx1,
                                idx2: rec.idx2,
                                param1_label: rec.p1.label(),
                                param2_label: rec.p2_label.clone(),
                                start_index: ev.start_index,
                                nlt: ev.nlt,
                                comparison: comp,
                            });
                        }
                    }
                }
            }
        }

        let hm = &ev.hom_metrics;
        let im = ev.inhom_metrics.as_ref();

        points.push(StudyPointMetrics {
            idx1: rec.idx1,
            idx2: rec.idx2,
            param1_label: rec.p1.label(),
            param2_label: rec.p2_label.clone(),
            param1_num: rec.p1.as_f64(rec.idx1),
            param2_num: rec.p2_num,
            method: rec.req.method.clone(),
            n_cells: rec.n_cells,
            n_visited: ev.n_visited_hom,
            start_index: ev.start_index,
            nlt: ev.nlt,
            step: ev.step,
            dt: ev.dt,
            tau: ev.tau,
            n_pairs_used: ev.n_pairs_used,
            nan_rows_hom: ev.nan_rows_hom,
            nan_rows_inhom_mean: ev.nan_rows_inhom_mean,
            rmse_rsd_hom: hm.rmse_rsd,
            mae_rsd_hom: hm.mae_rsd,
            max_rsd_err_hom: hm.max_abs_err_rsd,
            rmse_teneur_hom: hm.rmse_teneur,
            l1_mean_hom: hm.l1_mean_target,
            rms_disc_hom: hm.rms_discrepancy_mean,
            t50_dem: hm.t50_dem,
            t50_hom: hm.t50_markov,
            diag_mean: ev.diag_mean_raw,
            lambda_max: ev.lambda_max_hom,
            frobenius_rel_err: frob,
            rmse_rsd_inhom: im.map(|m| m.rmse_rsd),
            rmse_teneur_inhom: im.map(|m| m.rmse_teneur),
            l1_mean_inhom: im.map(|m| m.l1_mean_target),
            t50_inhom: im.and_then(|m| m.t50_markov),
            best_law_family,
            best_law_name,
            best_law_r2,
            best_law_rmse,
            best_law_aicc,
            best_law_oob_rmse,
            error: None,
        });

        // For 1-D studies (or first slice of 2-D), emit curve overlays capped cleanly
        let should_emit_curve = spec.include_curves
            && (spec.secondary_axis.is_none() || rec.idx2 == 0)
            && curves.len() < 16;
        if should_emit_curve {
            let empty_rsd = Vec::new();
            let rsd_inh = im.map(|m| &m.rsd_markov).unwrap_or(&empty_rsd);
            let (t_ds, dem_ds, hom_ds, inh_ds) =
                downsample_curve(&hm.times_eval, &hm.rsd_dem, &hm.rsd_markov, rsd_inh, 90);
            let lbl = if spec.secondary_axis.is_some() {
                format!("{} · {}", rec.p1.label(), rec.p2_label)
            } else {
                rec.p1.label()
            };
            curves.push(StudyCurveSeries {
                idx1: rec.idx1,
                idx2: rec.idx2,
                label: lbl,
                times: t_ds,
                rsd_dem: dem_ds,
                rsd_hom: hom_ds,
                rsd_inhom: inh_ds,
            });
        }
    }

    Ok(StudyResultBundle {
        study_id: spec.study_id.clone(),
        title: spec.title.clone(),
        subtitle: spec.subtitle.clone(),
        dataset: spec.base_request.dataset.clone(),
        method: spec.base_request.method.clone(),
        chain_mode: spec.chain_mode,
        enable_law_study: spec.enable_law_study,
        primary_axis: spec.primary_axis.clone(),
        secondary_axis: spec.secondary_axis.clone(),
        points,
        curves,
        law_points,
        available_transitions,
        active_transition,
    })
}
