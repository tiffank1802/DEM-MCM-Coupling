use std::sync::Arc;

use dem_mcm_coupling::gui::{generate_synthetic_drum_frames, SimulationRequest};
use dem_mcm_coupling::research::{
    build_preset_spec, catalog_study_presets, compare_transition_laws, fit_law_family, run_study,
    AxisValue, BedFrameBasis, LawFamily, LawSelectionCriterion, ReferencePolicy,
    ResearchSessionCache, StudyAxisKind, StudyAxisSpec, StudyChainMode, StudySpec,
};

#[test]
fn test_research_session_cache_reuses_arc_across_temporal_sweeps() {
    let frames = generate_synthetic_drum_frames(120, 200);
    let cache = ResearchSessionCache::new();
    let kwargs = SimulationRequest::default_kwargs_for("cartesian");

    let s1 = cache
        .get_or_build("synthetic_drum", "cartesian", &kwargs, 20, false, &frames)
        .expect("build session 1");
    let s2 = cache
        .get_or_build("synthetic_drum", "cartesian", &kwargs, 20, false, &frames)
        .expect("reuse session 2");
    assert!(Arc::ptr_eq(&s1, &s2));
    assert_eq!(cache.len(), 1);

    // BedFrameBasis flag creates a distinct spatial session key and transforms coordinates
    let s3 = cache
        .get_or_build("synthetic_drum", "cartesian", &kwargs, 20, true, &frames)
        .expect("build bed-frame session");
    assert!(!Arc::ptr_eq(&s1, &s3));
    assert_eq!(cache.len(), 2);

    let basis = BedFrameBasis::fit_from_coords(&[
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.5, 0.2],
    ]);
    let p = [1.0, 0.5, 0.2];
    let proj = basis.project_point(p);
    assert!(proj.iter().all(|v| v.is_finite()));
}

#[test]
fn test_seven_law_families_fitting_aicc_and_out_of_block_rmse() {
    let times: Vec<f64> = (0..9).map(|i| i as f64 * 15.0).collect();
    let span = 120.0;
    let dense: Vec<f64> = (0..25).map(|i| i as f64 * 5.0).collect();

    // 1. Exponential relaxation curve -> ExponentialRelax should achieve near-perfect R^2 and OOB RMSE
    let y_exp: Vec<f64> = times
        .iter()
        .map(|&t| {
            let u = t / span;
            0.12 + 0.55 * (-2.4 * u).exp()
        })
        .collect();
    let exp_fit = fit_law_family(&times, &y_exp, LawFamily::ExponentialRelax, &dense).unwrap();
    assert!(exp_fit.r2 > 0.995, "exp R^2 was {}", exp_fit.r2);
    assert!(
        exp_fit.out_of_block_rmse.unwrap_or(1.0) < 0.015,
        "exp OOB RMSE too high: {:?}",
        exp_fit.out_of_block_rmse
    );

    let cmp_exp = compare_transition_laws(
        0,
        1,
        &times,
        &y_exp,
        &LawFamily::all(),
        LawSelectionCriterion::OutOfBlockRmse,
    )
    .unwrap();
    assert_eq!(cmp_exp.candidates.len(), 7);
    assert!(cmp_exp.best_candidate().unwrap().r2 > 0.99);

    // 2. Linear curve -> AICc prefers Linear over over-parameterized Cubic
    let y_lin: Vec<f64> = times.iter().map(|&t| 0.20 + 0.40 * (t / span)).collect();
    let cmp_lin = compare_transition_laws(
        1,
        2,
        &times,
        &y_lin,
        &LawFamily::all(),
        LawSelectionCriterion::Aicc,
    )
    .unwrap();
    assert!(
        matches!(cmp_lin.best_family, LawFamily::Linear),
        "Expected AICc to prefer Linear over Cubic, got {:?}",
        cmp_lin.best_family
    );

    // 3. Constant signal -> Constant wins by AICc
    let y_const = vec![0.42; times.len()];
    let cmp_const = compare_transition_laws(
        0,
        0,
        &times,
        &y_const,
        &LawFamily::all(),
        LawSelectionCriterion::Aicc,
    )
    .unwrap();
    assert_eq!(cmp_const.best_family, LawFamily::Constant);
}

#[test]
fn test_all_8_study_presets_execute_and_serialize_cleanly() {
    let frames = generate_synthetic_drum_frames(180, 260);
    let cache = ResearchSessionCache::new();
    let base = SimulationRequest {
        dataset: "synthetic_drum".to_string(),
        method: "cartesian".to_string(),
        kwargs: SimulationRequest::default_kwargs_for("cartesian"),
        nlt: 3,
        step: 15,
        dt: 1,
        tau: 6,
        start_index: 20,
        permanent_start: 20,
        no_species: false,
        target_species: "small".to_string(),
        predict_full_duration: true,
        ..SimulationRequest::default()
    };

    let presets = catalog_study_presets();
    assert_eq!(presets.len(), 8);

    for p in presets {
        let mut spec = build_preset_spec(p.id, &base);
        // Trim grid sizes slightly so unit test runs fast in debug & release
        if spec.primary_axis.values.len() > 4 {
            spec.primary_axis.values.truncate(4);
        }
        if let Some(ref mut ax2) = spec.secondary_axis {
            if ax2.values.len() > 2 {
                ax2.values.truncate(2);
            }
        }
        // Keep values compatible with the 180-frame test slice
        if spec.primary_axis.kind == StudyAxisKind::StartIndex {
            spec.primary_axis.values =
                vec![AxisValue::Int(0), AxisValue::Int(20), AxisValue::Int(40)];
        } else if spec.primary_axis.kind == StudyAxisKind::Step {
            spec.primary_axis.values =
                vec![AxisValue::Int(5), AxisValue::Int(10), AxisValue::Int(15)];
        } else if spec.primary_axis.kind == StudyAxisKind::Tau {
            spec.primary_axis.values =
                vec![AxisValue::Int(4), AxisValue::Int(8), AxisValue::Int(12)];
        }

        let res = run_study(&spec, &cache, &frames)
            .unwrap_or_else(|e| panic!("Preset {} failed: {e}", p.id));

        assert!(!res.points.is_empty(), "Preset {} produced no points", p.id);
        let valid_count = res.points.iter().filter(|pt| pt.error.is_none()).count();
        assert!(
            valid_count >= 1,
            "Preset {} had zero valid points: {:?}",
            p.id,
            res.points
        );

        let json_str = res.to_json().to_string_compact();
        assert!(json_str.contains("\"points\""));
        dem_mcm_coupling::io::json::JsonValue::parse(&json_str)
            .unwrap_or_else(|e| panic!("Preset {} produced invalid JSON: {e}", p.id));

        if p.id == "inhom_laws_grid" {
            assert!(
                !res.law_points.is_empty(),
                "2D laws preset had no law_points"
            );
            assert!(res.points.iter().any(|pt| pt.best_law_family.is_some()));
        }
    }
}

#[test]
fn test_species_mode_and_velocity_weight_custom_studies() {
    let frames = generate_synthetic_drum_frames(150, 240);
    let cache = ResearchSessionCache::new();
    let base = SimulationRequest {
        dataset: "synthetic_drum".to_string(),
        method: "physics".to_string(),
        kwargs: SimulationRequest::default_kwargs_for("physics"),
        nlt: 3,
        step: 15,
        dt: 1,
        tau: 6,
        start_index: 20,
        permanent_start: 20,
        no_species: false,
        target_species: "small".to_string(),
        predict_full_duration: true,
        ..SimulationRequest::default()
    };

    let spec_sp = StudySpec {
        study_id: "species_test".to_string(),
        title: "Espèces vs sans espèces".to_string(),
        subtitle: "Comparaison".to_string(),
        base_request: base.clone(),
        chain_mode: StudyChainMode::Both,
        primary_axis: StudyAxisSpec {
            kind: StudyAxisKind::SpeciesMode,
            values: vec![AxisValue::Bool(false), AxisValue::Bool(true)],
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
    };
    let out_sp = run_study(&spec_sp, &cache, &frames).unwrap();
    assert_eq!(out_sp.points.len(), 2);
    assert!(out_sp
        .points
        .iter()
        .all(|p| p.error.is_none() && p.rmse_rsd_inhom.is_some()));

    let spec_wv = StudySpec {
        study_id: "wv_test".to_string(),
        title: "Poids vitesse physics".to_string(),
        subtitle: "w_V".to_string(),
        base_request: base,
        chain_mode: StudyChainMode::Homogeneous,
        primary_axis: StudyAxisSpec {
            kind: StudyAxisKind::VelocityWeight,
            values: vec![AxisValue::Float(0.0), AxisValue::Float(0.5)],
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
    };
    let out_wv = run_study(&spec_wv, &cache, &frames).unwrap();
    assert_eq!(out_wv.points.len(), 2);
    assert!(out_wv.points.iter().all(|p| p.error.is_none()));
}
