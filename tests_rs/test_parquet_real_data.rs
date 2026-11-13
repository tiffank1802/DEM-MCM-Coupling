//! End-to-end validation against real compressed Parquet DEM chunks in `data/chunks/`.

use std::collections::HashMap;
use std::path::Path;

use dem_mcm_coupling::analyze_results::MarkovAnalyzer;
use dem_mcm_coupling::data::{DataSource, InMemoryDataSource, LocalDataSource};
use dem_mcm_coupling::io::json::JsonValue;
use dem_mcm_coupling::io::parquet::read_dem_parquet;
use dem_mcm_coupling::metrics::validate_experiment;
use dem_mcm_coupling::partitioners::create_partitioner;
use dem_mcm_coupling::run_sweep::{
    run_experiment, run_inhomogeneous_experiment, save_inhomogeneous_results, save_results,
    ExperimentConfig,
};

#[test]
fn pure_rust_parquet_reader_parses_real_simulation_chunk_00() {
    let chunk0 = Path::new("data/chunks/simulation_part_00.parquet.gz");
    if !chunk0.exists() {
        eprintln!("Skipping real parquet test: {chunk0:?} not present");
        return;
    }

    // 1. Validate direct chunk decoding (400 scattered timesteps x 1,030 particles = 412,000 rows)
    let chunk_frames = read_dem_parquet(chunk0, None)
        .expect("Failed to parse data/chunks/simulation_part_00.parquet.gz");
    assert_eq!(chunk_frames.len(), 400);
    for frame in chunk_frames.values() {
        assert_eq!(frame.n_particles(), 1030);
        assert_eq!(frame.velocities.len(), 1030);
        assert_eq!(frame.diameters.len(), 1030);
        assert!(frame
            .coordinates
            .iter()
            .all(|c| c.iter().all(|v| v.is_finite())));
        assert!(frame
            .velocities
            .iter()
            .all(|v| v.iter().all(|x| x.is_finite())));
    }
    let first_frame = chunk_frames.values().next().unwrap();
    let n_small = first_frame
        .diameters
        .iter()
        .filter(|&&d| (d - 0.004).abs() < 1e-6)
        .count();
    let n_large = first_frame
        .diameters
        .iter()
        .filter(|&&d| (d - 0.008).abs() < 1e-6)
        .count();
    assert!(n_small > 0 && n_large > 0 && n_small + n_large == 1030);

    // 2. Load a real contiguous rotating-drum window [250..410) across `data/chunks/`
    let window: Vec<usize> = (250..410).collect();
    let local_chunks = LocalDataSource::new("data/chunks");
    let frames = local_chunks.read_timesteps(Some(&window)).unwrap();
    assert_eq!(frames.len(), 160);
    assert!(frames.contains_key(&250) && frames.contains_key(&409));

    let perm_coords: Vec<[f64; 3]> = frames
        .values()
        .flat_map(|f| f.coordinates.iter().copied())
        .collect();
    let perm_vels: Vec<[f64; 3]> = frames
        .values()
        .flat_map(|f| f.velocities.iter().copied())
        .collect();

    // 3. Run real homogeneous Markov coupling across Cartesian, Cylindrical, Voronoi, Physics
    for &(method, ref kw_pairs) in &[
        ("cartesian", vec![("nx", 4.0), ("ny", 4.0), ("nz", 2.0)]),
        (
            "cylindrical",
            vec![("nr", 3.0), ("ntheta", 6.0), ("nz", 1.0)],
        ),
        ("voronoi", vec![("n_cells", 16.0)]),
        ("physics", vec![("n_cells", 16.0), ("velocity_weight", 0.5)]),
    ] {
        let mut kw: HashMap<String, JsonValue> = HashMap::new();
        for &(k, v) in kw_pairs {
            kw.insert(k.to_string(), JsonValue::Number(v));
        }
        let mut p = create_partitioner(method, &kw).unwrap();
        if method == "physics" {
            p.set_use_velocity(true);
            p.set_dem_velocities(Some(&perm_vels));
        }
        p.fit(&perm_coords).unwrap();

        let mut cfg = ExperimentConfig::new(method);
        cfg.method_kwargs = kw;
        cfg.nlt = 2;
        cfg.step = 50;
        cfg.dt = 1;
        cfg.tau = 20;
        cfg.start_index = 250;

        let res = run_experiment(&cfg, p.as_mut(), &frames).unwrap();
        let mut mem = InMemoryDataSource::new();
        let fname = save_results(&cfg, p.as_ref(), &res, None, &mut mem).unwrap();
        let exp = mem.read_experiment(&fname, None).unwrap();

        let rep = validate_experiment(&fname, &exp, 1e-3);
        assert!(
            rep.passed,
            "Real-data validation failed for {method}:\n{rep}"
        );

        let analyzer = MarkovAnalyzer::new();
        let cmp = analyzer.evaluate_experiment(&exp).unwrap();
        assert!(cmp.rmse_rsd.is_finite() && cmp.rmse_rsd < 0.6);
        if matches!(method, "voronoi" | "physics") {
            assert!(cmp.rmse_rsd < 0.25, "{method} rmse={}", cmp.rmse_rsd);
        }
    }

    // 4. Run real inhomogeneous Markov coupling across 3 blocks on real DEM trajectories
    let mut kw = HashMap::new();
    kw.insert("nx".into(), JsonValue::Number(3.0));
    kw.insert("ny".into(), JsonValue::Number(3.0));
    kw.insert("nz".into(), JsonValue::Number(1.0));
    let mut p = create_partitioner("cartesian", &kw).unwrap();
    p.fit(&perm_coords).unwrap();
    let mut cfg_inh = ExperimentConfig::new("cartesian");
    cfg_inh.method_kwargs = kw;
    cfg_inh.nlt = 3;
    cfg_inh.step = 30;
    cfg_inh.dt = 1;
    cfg_inh.tau = 15;
    cfg_inh.start_index = 250;
    cfg_inh.inhomogeneous = true;

    let res_inh = run_inhomogeneous_experiment(&cfg_inh, p.as_mut(), &frames).unwrap();
    let mut mem = InMemoryDataSource::new();
    let fname_inh =
        save_inhomogeneous_results(&cfg_inh, p.as_ref(), &res_inh, None, &mut mem).unwrap();
    let exp_inh = mem.read_experiment(&fname_inh, None).unwrap();
    let analyzer = MarkovAnalyzer::new();
    let cmp_inh = analyzer.evaluate_experiment(&exp_inh).unwrap();
    assert!(cmp_inh.rmse_rsd.is_finite());
}
