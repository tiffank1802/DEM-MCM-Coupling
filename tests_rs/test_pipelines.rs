//! Integration tests for homogeneous, no-species, and inhomogeneous sweep pipelines.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use dem_mcm_coupling::analyze_results::MarkovAnalyzer;
use dem_mcm_coupling::data::{DataSource, DemFrame, InMemoryDataSource, LocalDataSource};
use dem_mcm_coupling::io::json::JsonValue;
use dem_mcm_coupling::metrics::validate_experiment;
use dem_mcm_coupling::partitioners::geometric::CartesianPartitioner;
use dem_mcm_coupling::run_sweep::{
    compute_p_matrix, get_configs, run_experiment, run_inhomogeneous_experiment,
    run_no_species_experiment, run_sweep, save_inhomogeneous_results, save_results,
    ExperimentConfig, SweepMode,
};
use dem_mcm_coupling::Partitioner;

fn make_rotating_dem_frames(n_timesteps: usize, n_particles: usize) -> BTreeMap<usize, DemFrame> {
    let mut frames = BTreeMap::new();
    let half = n_particles / 2;
    for ts in 0..n_timesteps {
        let mix_frac = ((ts as f64) / (n_timesteps as f64)).min(1.0);
        let mut coords = Vec::with_capacity(n_particles);
        let mut vels = Vec::with_capacity(n_particles);
        let mut diams = Vec::with_capacity(n_particles);
        for p in 0..n_particles {
            let is_small = p < half;
            let local_i = if is_small { p } else { p - half };
            // Initially small occupy x < 0, large occupy x > 0 with identical (y, z) quadrants
            // so all 8 cells are equally populated and C_0 = {1, 0} -> RSD_0 = 1.0000 cleanly.
            let y_q = local_i % 2;
            let z_q = (local_i / 2) % 2;
            let y = if y_q == 0 { -0.04 } else { 0.04 } + 0.01 * (((local_i * 7) % 3) as f64 - 1.0);
            let z =
                if z_q == 0 { -0.015 } else { 0.015 } + 0.004 * (((local_i * 11) % 3) as f64 - 1.0);
            let base_x = 0.04 + 0.02 * (((local_i * 5) % 3) as f64);
            let swapped = ((local_i * 31 + 7) % 100) as f64 / 100.0 < 0.5 * mix_frac;
            let x_sign = match (is_small, swapped) {
                (true, false) | (false, true) => -1.0,
                (false, false) | (true, true) => 1.0,
            };
            let jitter = 0.001 * (((ts * 13 + local_i) % 3) as f64 - 1.0);
            let x = x_sign * base_x + jitter;
            coords.push([x, y, z]);
            vels.push([-4.0 * y, 4.0 * x, 0.0]);
            diams.push(if is_small { 0.004 } else { 0.008 });
        }
        frames.insert(ts, DemFrame::with_kinematics(ts, coords, vels, diams));
    }
    frames
}

#[test]
fn compute_p_matrix_handles_unvisited_nan_and_out_of_zone() {
    let prev = vec![0, 0, 1, -1, 0];
    let curr = vec![0, 1, -1, 0, 0];
    // State 0 visited 3 times: transitions 0->0 (2x), 0->1 (1x)
    // State 1 visited 1 time, transitioned to -1 (out-of-zone): numerator 0 / denom 1 = 0.0
    // State 2 never visited: 0.0 / 0.0 = NaN
    let p = compute_p_matrix(&prev, &curr, 3);
    assert!((p.get(0, 0) - 2.0 / 3.0).abs() < 1e-12);
    assert!((p.get(0, 1) - 1.0 / 3.0).abs() < 1e-12);
    assert!((p.get(0, 2) - 0.0).abs() < 1e-12);
    assert!((p.get(1, 0) - 0.0).abs() < 1e-12);
    assert!(p.get(2, 0).is_nan());
    assert!(p.get(2, 1).is_nan());
    assert!(p.get(2, 2).is_nan());
}

#[test]
fn get_configs_deduplicates_and_auto_dt() {
    let cfgs = get_configs("cartesian", None).unwrap();
    assert!(cfgs.len() >= 10);
    for c in &cfgs {
        assert_eq!(c.method, "cartesian");
        assert!(c.dt >= 1);
    }
    let auto = ExperimentConfig {
        step: 350,
        ..ExperimentConfig::new("cartesian")
    }
    .with_dt_auto();
    assert_eq!(auto.dt, 3);
}

#[test]
fn homogeneous_no_species_and_inhomogeneous_pipelines_roundtrip_and_validate() {
    let frames = make_rotating_dem_frames(80, 120);
    let all_coords: Vec<[f64; 3]> = frames
        .values()
        .flat_map(|f| f.coordinates.iter().copied())
        .collect();

    let mut cart = CartesianPartitioner::new(2, 2, 2);
    cart.fit(&all_coords).unwrap();

    // 1. Homogeneous run
    let mut cfg = ExperimentConfig::new("cartesian");
    cfg.method_kwargs
        .insert("nx".into(), JsonValue::Number(2.0));
    cfg.method_kwargs
        .insert("ny".into(), JsonValue::Number(2.0));
    cfg.method_kwargs
        .insert("nz".into(), JsonValue::Number(2.0));
    cfg.nlt = 2;
    cfg.step = 15;
    cfg.dt = 1;
    cfg.tau = 5;
    cfg.start_index = 5;

    let res_hom = run_experiment(&cfg, &mut cart, &frames).unwrap();
    assert!(res_hom.species.contains_key("small"));
    assert!(res_hom.species.contains_key("large"));
    assert_eq!(
        res_hom
            .stats
            .get("species_masks_applied")
            .and_then(|v| v.as_bool()),
        Some(true)
    );

    let mut mem = InMemoryDataSource::from_timesteps(frames.clone());
    let folder_hom = save_results(&cfg, &cart, &res_hom, None, &mut mem).unwrap();
    let exp_hom = mem.read_experiment(&folder_hom, None).unwrap();
    assert!(!exp_hom.inhomogeneous);
    let rep_hom = validate_experiment(&folder_hom, &exp_hom, 1e-3);
    assert!(rep_hom.passed, "Validation failed:\n{rep_hom}");

    let analyzer = MarkovAnalyzer::new();
    let metrics_hom = analyzer.evaluate_experiment(&exp_hom).unwrap();
    assert!(metrics_hom.rmse_rsd.is_finite());

    // 2. No-species run
    let res_ns = run_no_species_experiment(&cfg, &mut cart, &frames).unwrap();
    assert_eq!(res_ns.species.len(), 1);
    assert!(res_ns.species.contains_key("all"));
    assert_eq!(
        res_ns
            .stats
            .get("species_masks_applied")
            .and_then(|v| v.as_bool()),
        Some(false)
    );
    let folder_ns = format!("nospecies_{}", cfg.output_folder(None).unwrap());
    save_results(&cfg, &cart, &res_ns, Some(&folder_ns), &mut mem).unwrap();
    let exp_ns = mem.read_experiment(&folder_ns, None).unwrap();
    assert_eq!(MarkovAnalyzer::detect_method(&folder_ns, None), "cartesian");
    let rep_ns = validate_experiment(&folder_ns, &exp_ns, 1e-3);
    assert!(rep_ns.passed, "No-species validation failed:\n{rep_ns}");

    // 3. Inhomogeneous run -> write to LocalDataSource on disk and verify files!
    let mut cfg_inh = cfg.clone();
    cfg_inh.inhomogeneous = true;
    cfg_inh.nlt = 3;
    let res_inh = run_inhomogeneous_experiment(&cfg_inh, &mut cart, &frames).unwrap();
    assert_eq!(
        res_inh.stats.get("inhomogeneous").and_then(|v| v.as_bool()),
        Some(true)
    );
    let tmp_dir: PathBuf =
        std::env::temp_dir().join(format!("dem_mcm_test_pipe_{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp_dir);
    let mut local = LocalDataSource::new(&tmp_dir);
    let folder_inh =
        save_inhomogeneous_results(&cfg_inh, &cart, &res_inh, None, &mut local).unwrap();
    assert!(folder_inh.starts_with("inhomogeneous_"));
    assert!(tmp_dir
        .join(&folder_inh)
        .join("inhomogeneous_metadata.json")
        .exists());
    assert!(tmp_dir
        .join(&folder_inh)
        .join("P_blocks_small.npy")
        .exists());

    let exp_inh = local.read_experiment(&folder_inh, None).unwrap();
    assert!(exp_inh.inhomogeneous);
    assert!(exp_inh.species["small"].p_blocks.as_ref().unwrap().len() >= 2);
    let metrics_inh = analyzer.evaluate_experiment(&exp_inh).unwrap();
    assert!(!metrics_inh.rsd_markov.is_empty());

    let _ = fs::remove_dir_all(&tmp_dir);
}

#[test]
fn run_sweep_executes_multiple_configs_via_datasource() {
    let frames = make_rotating_dem_frames(60, 80);
    let mut mem = InMemoryDataSource::from_timesteps(frames);
    let mut c1 = ExperimentConfig::new("cartesian");
    c1.method_kwargs.insert("nx".into(), JsonValue::Number(2.0));
    c1.method_kwargs.insert("ny".into(), JsonValue::Number(2.0));
    c1.method_kwargs.insert("nz".into(), JsonValue::Number(1.0));
    c1.nlt = 1;
    c1.step = 10;
    c1.dt = 1;
    c1.tau = 5;
    c1.start_index = 2;

    let mut c2 = c1.clone();
    c2.method = "single".to_string();
    c2.method_kwargs.clear();

    let outcomes = run_sweep(
        "cartesian",
        SweepMode::Homogeneous,
        Some(vec![c1, c2]),
        None,
        &mut mem,
    )
    .unwrap();
    assert_eq!(outcomes.len(), 2);
    assert!(outcomes.iter().all(|o| o.success), "outcomes={outcomes:?}");

    let mut analyzer = MarkovAnalyzer::new();
    let loaded = analyzer.load_method(&mem, None).unwrap();
    assert_eq!(loaded, 2);
    let avail = analyzer.list_available_models(Some("cartesian"), None, 0.5);
    assert_eq!(avail.len(), 1);
}
