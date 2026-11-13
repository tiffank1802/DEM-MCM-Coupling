//! Built-in reproducible research study presets matching the project's
//! scientific investigations (homogeneous start/NLT/step/dt, Markov horizon tau,
//! partitioning method vs RSD, and 2-D `start_index x NLT` multi-law fits).

use crate::gui::state::SimulationRequest;
use crate::io::json::JsonValue;
use crate::research::laws::{LawFamily, LawSelectionCriterion};
use crate::research::study::{
    AxisValue, ReferencePolicy, StudyAxisKind, StudyAxisSpec, StudyChainMode, StudySpec,
};

#[derive(Debug, Clone)]
pub struct StudyPresetDescriptor {
    pub id: &'static str,
    pub category: &'static str,
    pub name: &'static str,
    pub description: &'static str,
}

pub fn catalog_study_presets() -> Vec<StudyPresetDescriptor> {
    vec![
        StudyPresetDescriptor {
            id: "hom_start",
            category: "Homogène",
            name: "1. Homogène · Départ d'apprentissage (start_index)",
            description: "Évalue l'effet du régime transitoire vs permanent sur RSD(t), Teneur et Erreur L¹.",
        },
        StudyPresetDescriptor {
            id: "hom_nlt_frob",
            category: "Homogène",
            name: "2. Homogène · Nombre de blocs (NLT) & Frobenius",
            description: "Convergence de P_hom vers la référence (norme de Frobenius Doucet) et erreur RSD selon NLT.",
        },
        StudyPresetDescriptor {
            id: "hom_step_dt",
            category: "Homogène",
            name: "3. Homogène · Raffinage intra-bloc (dt) & Lignes NaN",
            description: "Montre l'impact du pas dt sur les états non visités (NaN) et la précision du mélange.",
        },
        StudyPresetDescriptor {
            id: "hom_step",
            category: "Homogène",
            name: "4. Homogène · Écart inter-blocs (step)",
            description: "Sensibilité des métriques à l'espacement temporel entre blocs d'apprentissage.",
        },
        StudyPresetDescriptor {
            id: "tau_sensitivity",
            category: "Temporel",
            name: "5. Horizon Markov τ · Cinétique RSD & t50",
            description: "Impact de l'horizon τ sur la vitesse de mélange prédite, t50 et la stabilité spectrale.",
        },
        StudyPresetDescriptor {
            id: "method_rsd",
            category: "Spatial",
            name: "6. Découpage spatial · Méthodes × RSD(t)",
            description: "Compare dynamiquement l'impact des approches géométriques/physiques/clustering sur le RSD.",
        },
        StudyPresetDescriptor {
            id: "spatial_resolution",
            category: "Spatial",
            name: "7. Découpage spatial · Résolution (n_cells / nx)",
            description: "Compromis résolution spatiale vs erreur RSD, Teneur et apparition de lignes NaN.",
        },
        StudyPresetDescriptor {
            id: "inhom_laws_grid",
            category: "Inhomogène · Lois",
            name: "8. Inhomogène · Lois p_{i→j}(t) : carte (start × NLT)",
            description: "Construit et compare 7 familles de lois selon start_index et le nombre de blocs NLT (AICc & out-of-block).",
        },
    ]
}

pub fn study_presets_json() -> JsonValue {
    let arr = catalog_study_presets()
        .into_iter()
        .map(|p| {
            let mut o = JsonValue::object();
            o.insert("id", JsonValue::String(p.id.to_string()));
            o.insert("category", JsonValue::String(p.category.to_string()));
            o.insert("name", JsonValue::String(p.name.to_string()));
            o.insert("description", JsonValue::String(p.description.to_string()));
            o
        })
        .collect();
    let mut root = JsonValue::object();
    root.insert("presets", JsonValue::Array(arr));
    root
}

/// Build a ready-to-run [`StudySpec`] for `preset_id` calibrated to `base_req.dataset`.
pub fn build_preset_spec(preset_id: &str, base_req: &SimulationRequest) -> StudySpec {
    let mut req = base_req.clone();
    let is_real = req.dataset == "real_chunks";
    if is_real {
        if req.step < 20 {
            req.step = 157;
        }
        if req.tau < 20 {
            req.tau = 157;
        }
        if req.start_index == 40 {
            req.start_index = 157;
        }
        req.permanent_start = req.permanent_start.max(157);
    }

    match preset_id {
        "hom_start" => {
            let vals: Vec<AxisValue> = if is_real {
                [0usize, 80, 157, 314, 785, 1570]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            } else {
                [0usize, 20, 40, 80, 120, 180]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            };
            StudySpec {
                study_id: "hom_start".to_string(),
                title: "Homogène · Impact du départ d'apprentissage (start_index)".to_string(),
                subtitle: "Transitoire initial vs régime permanent sur RSD(t) et Teneur"
                    .to_string(),
                base_request: req,
                chain_mode: StudyChainMode::Homogeneous,
                primary_axis: StudyAxisSpec {
                    kind: StudyAxisKind::StartIndex,
                    values: vals,
                    couple_step_to_tau: false,
                },
                secondary_axis: None,
                use_bed_frame: false,
                enable_law_study: false,
                law_families: LawFamily::all(),
                law_criterion: LawSelectionCriterion::Aicc,
                focus_transition: None,
                reference_policy: ReferencePolicy::LastPoint,
                include_curves: true,
            }
        }
        "hom_nlt_frob" => {
            req.start_index = if is_real { 157 } else { 20 };
            req.step = if is_real { 157 } else { 20 };
            req.tau = if is_real { 157 } else { 10 };
            req.dt = if is_real { 2 } else { 1 };
            let vals: Vec<AxisValue> = if is_real {
                [1usize, 2, 3, 5, 8, 12, 16]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            } else {
                [1usize, 2, 3, 5, 7, 9]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            };
            StudySpec {
                study_id: "hom_nlt_frob".to_string(),
                title: "Homogène · Convergence NLT & Norme de Frobenius".to_string(),
                subtitle: "Erreur relative ||P_NLT - P_réf||_F / ||P_réf||_F et erreur RSD"
                    .to_string(),
                base_request: req,
                chain_mode: StudyChainMode::Homogeneous,
                primary_axis: StudyAxisSpec {
                    kind: StudyAxisKind::Nlt,
                    values: vals,
                    couple_step_to_tau: false,
                },
                secondary_axis: None,
                use_bed_frame: false,
                enable_law_study: false,
                law_families: LawFamily::all(),
                law_criterion: LawSelectionCriterion::Aicc,
                focus_transition: None,
                reference_policy: ReferencePolicy::LastPoint,
                include_curves: true,
            }
        }
        "hom_step_dt" => {
            req.nlt = 3;
            req.step = if is_real { 157 } else { 30 };
            req.tau = if is_real { 157 } else { 15 };
            let vals: Vec<AxisValue> = if is_real {
                [1usize, 2, 4, 8, 16, 39, 78]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            } else {
                [1usize, 2, 4, 8, 15, 25]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            };
            StudySpec {
                study_id: "hom_step_dt".to_string(),
                title: "Homogène · Pas d'échantillonnage dt & Lignes NaN".to_string(),
                subtitle: "Influence de dt sur la visite des cellules et la stochasticité"
                    .to_string(),
                base_request: req,
                chain_mode: StudyChainMode::Homogeneous,
                primary_axis: StudyAxisSpec {
                    kind: StudyAxisKind::Dt,
                    values: vals,
                    couple_step_to_tau: false,
                },
                secondary_axis: None,
                use_bed_frame: false,
                enable_law_study: false,
                law_families: LawFamily::all(),
                law_criterion: LawSelectionCriterion::Aicc,
                focus_transition: None,
                reference_policy: ReferencePolicy::FirstPoint,
                include_curves: true,
            }
        }
        "hom_step" => {
            req.nlt = 3;
            req.dt = 1;
            let vals: Vec<AxisValue> = if is_real {
                [15usize, 50, 157, 314, 471]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            } else {
                [5usize, 15, 30, 50, 80]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            };
            StudySpec {
                study_id: "hom_step".to_string(),
                title: "Homogène · Écart inter-blocs (step)".to_string(),
                subtitle: "Effet de l'espacement entre fenêtres d'apprentissage sur le mélange"
                    .to_string(),
                base_request: req,
                chain_mode: StudyChainMode::Homogeneous,
                primary_axis: StudyAxisSpec {
                    kind: StudyAxisKind::Step,
                    values: vals,
                    couple_step_to_tau: false,
                },
                secondary_axis: None,
                use_bed_frame: false,
                enable_law_study: false,
                law_families: LawFamily::all(),
                law_criterion: LawSelectionCriterion::Aicc,
                focus_transition: None,
                reference_policy: ReferencePolicy::LastPoint,
                include_curves: true,
            }
        }
        "tau_sensitivity" => {
            let vals: Vec<AxisValue> = if is_real {
                [15usize, 40, 78, 157, 314, 471]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            } else {
                [5usize, 10, 15, 25, 40, 60]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            };
            StudySpec {
                study_id: "tau_sensitivity".to_string(),
                title: "Horizon Markov τ · Dynamique RSD(t) & Temps de mélange t50".to_string(),
                subtitle: "Impact spatial et temporel du pas markovien τ sur la prédiction"
                    .to_string(),
                base_request: req,
                chain_mode: StudyChainMode::Both,
                primary_axis: StudyAxisSpec {
                    kind: StudyAxisKind::Tau,
                    values: vals,
                    couple_step_to_tau: false,
                },
                secondary_axis: None,
                use_bed_frame: false,
                enable_law_study: false,
                law_families: LawFamily::all(),
                law_criterion: LawSelectionCriterion::Aicc,
                focus_transition: None,
                reference_policy: ReferencePolicy::None,
                include_curves: true,
            }
        }
        "method_rsd" => {
            let methods = [
                "cartesian",
                "cylindrical",
                "quantile",
                "octree",
                "voronoi",
                "physics",
                "adaptive",
            ];
            StudySpec {
                study_id: "method_rsd".to_string(),
                title: "Découpage spatial · Impact des méthodes sur le RSD(t)".to_string(),
                subtitle: "Comparaison dynamique des maillages sur l'homogénéisation et la teneur"
                    .to_string(),
                base_request: req,
                chain_mode: StudyChainMode::Both,
                primary_axis: StudyAxisSpec {
                    kind: StudyAxisKind::Method,
                    values: methods
                        .into_iter()
                        .map(|m| AxisValue::Text(m.to_string()))
                        .collect(),
                    couple_step_to_tau: false,
                },
                secondary_axis: None,
                use_bed_frame: false,
                enable_law_study: false,
                law_families: LawFamily::all(),
                law_criterion: LawSelectionCriterion::Aicc,
                focus_transition: None,
                reference_policy: ReferencePolicy::None,
                include_curves: true,
            }
        }
        "spatial_resolution" => {
            let vals = match req.method.as_str() {
                "cartesian" | "quantile" | "cylindrical" => [2usize, 3, 4, 5, 6]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect(),
                "octree" => [240usize, 150, 90, 50, 30]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect(),
                _ => [6usize, 10, 14, 18, 24, 32]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect(),
            };
            StudySpec {
                study_id: "spatial_resolution".to_string(),
                title: format!("Résolution spatiale ({}) × Erreurs & NaN", req.method),
                subtitle: "Impact du raffinement de maillage sur RSD, Teneur et cellules vides"
                    .to_string(),
                base_request: req,
                chain_mode: StudyChainMode::Homogeneous,
                primary_axis: StudyAxisSpec {
                    kind: StudyAxisKind::SpatialResolution,
                    values: vals,
                    couple_step_to_tau: false,
                },
                secondary_axis: None,
                use_bed_frame: false,
                enable_law_study: false,
                law_families: LawFamily::all(),
                law_criterion: LawSelectionCriterion::Aicc,
                focus_transition: None,
                reference_policy: ReferencePolicy::None,
                include_curves: true,
            }
        }
        _ => {
            // Default & "inhom_laws_grid"
            req.step = if is_real { 157 } else { 18 };
            req.tau = if is_real { 157 } else { 10 };
            req.dt = if is_real { 2 } else { 1 };
            let starts: Vec<AxisValue> = if is_real {
                [0usize, 157, 314, 785]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            } else {
                [0usize, 30, 60, 100]
                    .into_iter()
                    .map(AxisValue::Int)
                    .collect()
            };
            let nlts: Vec<AxisValue> = if is_real {
                [3usize, 5, 8, 12].into_iter().map(AxisValue::Int).collect()
            } else {
                [3usize, 5, 7].into_iter().map(AxisValue::Int).collect()
            };
            StudySpec {
                study_id: "inhom_laws_grid".to_string(),
                title: "Inhomogène · Lois p_{i→j}(t) : Influence de (start_index × NLT)"
                    .to_string(),
                subtitle: "Comparaison multi-familles (AICc & validation hors-bloc) selon le départ et le nombre de blocs"
                    .to_string(),
                base_request: req,
                chain_mode: StudyChainMode::Inhomogeneous,
                primary_axis: StudyAxisSpec {
                    kind: StudyAxisKind::StartIndex,
                    values: starts,
                    couple_step_to_tau: false,
                },
                secondary_axis: Some(StudyAxisSpec {
                    kind: StudyAxisKind::Nlt,
                    values: nlts,
                    couple_step_to_tau: false,
                }),
                use_bed_frame: false,
                enable_law_study: true,
                law_families: LawFamily::all(),
                law_criterion: LawSelectionCriterion::Aicc,
                focus_transition: None,
                reference_policy: ReferencePolicy::None,
                include_curves: true,
            }
        }
    }
}
