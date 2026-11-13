use dem_mcm_coupling::data::{DataSource, LocalDataSource};
use dem_mcm_coupling::gui::{
    catalog_methods, generate_synthetic_drum_frames, run_gui_comparison, run_method_benchmark,
    SimulationRequest,
};

#[test]
fn test_all_14_methods_produce_valid_3d_meshes_and_full_dem_chains() {
    let frames = generate_synthetic_drum_frames(180, 320);
    let methods = catalog_methods();
    assert_eq!(methods.len(), 14);

    for m in methods {
        let req = SimulationRequest {
            dataset: "synthetic_drum".to_string(),
            method: m.id.to_string(),
            kwargs: SimulationRequest::default_kwargs_for(m.id),
            nlt: 3,
            step: 15,
            dt: 1,
            tau: 8,
            start_index: 30,
            permanent_start: 20,
            no_species: false,
            target_species: "small".to_string(),
            predict_full_duration: true,
            ..SimulationRequest::default()
        };
        let bundle = run_gui_comparison(&req, &frames)
            .unwrap_or_else(|e| panic!("Method '{}' failed GUI comparison: {e}", m.id));

        assert!(bundle.mesh.n_cells >= 1, "{} has no cells", m.id);
        assert!(
            !bundle.mesh.cells.is_empty() && bundle.mesh.cells.len() <= bundle.mesh.n_cells,
            "{} cell geometry count {} invalid for n_cells {}",
            m.id,
            bundle.mesh.cells.len(),
            bundle.mesh.n_cells
        );
        assert!(!bundle.frames.is_empty(), "{} has no playback frames", m.id);
        // Full DEM prediction starts at ts_min = 0 while model builds at [30..99]
        assert_eq!(bundle.pred_t_start, 0);
        assert_eq!(bundle.model_t_start, 30);
        assert!(bundle.pred_t_end >= 170);
        assert!(bundle.hom_metrics.rmse_teneur.is_finite());
        assert!(bundle.inhom_metrics.rmse_teneur.is_finite());
        assert!(bundle.hom_metrics.l1_mean_target.is_finite());
        assert!(bundle.inhom_metrics.rms_discrepancy_mean.is_finite());
        assert!(bundle.hom_metrics.mass_drift_pct < 1e-4);
        assert!(bundle.inhom_metrics.mass_drift_pct < 1e-4);
        // Check both particle sizes are represented in 3D frames
        let fr0 = &bundle.frames[0];
        assert!(fr0.n_small_total > 0 && fr0.n_large_total > 0);
        assert_eq!(fr0.n_small_total + fr0.n_large_total, fr0.n_total);
        assert!(fr0.diameters.iter().any(|&d| (d - 0.004).abs() < 1e-6));
        assert!(fr0.diameters.iter().any(|&d| (d - 0.008).abs() < 1e-6));

        let json_str = bundle.to_json().to_string_compact();
        assert!(json_str.contains("\"rmse_teneur\""));
        assert!(json_str.contains("\"l1_mean_target\""));
        assert!(json_str.contains("\"rms_discrepancy_mean\""));
        assert!(json_str.contains("\"diameters\""));
        assert!(json_str.contains("\"matrix_evolution_small\""));
        assert!(json_str.contains("\"matrix_evolution_large\""));

        // Verify inhomogeneous matrix component evolution covers diagonals & off-diagonals (> 0.01)
        assert!(!bundle.matrix_evolution_small.components.is_empty());
        assert!(bundle.matrix_evolution_small.global_vmax > 0.0);
        assert!(bundle
            .matrix_evolution_small
            .components
            .iter()
            .all(|c| !c.values.is_empty() && !c.dense_predicted.is_empty()));
    }
}

#[test]
fn test_web_ui_cad_shell_and_matrix_evolution_view() {
    let html = dem_mcm_coupling::gui::web::index_html();
    assert!(html.contains("DEM-MCM Studio"));
    assert!(!html.contains("Liquid Glass"));
    assert!(html.contains("pane-summary"));
    assert!(html.contains("pane-matrix_evol"));
    assert!(html.contains("pane-study"));
    assert!(html.contains("workbench-mode-switch"));
    assert!(html.contains("deriveUiVisibility"));
    assert!(html.contains("feature-tree"));
    assert!(html.contains("ribbon-bar"));
}

#[test]
fn test_target_species_large_and_multi_method_benchmark() {
    let frames = generate_synthetic_drum_frames(150, 260);
    let base = SimulationRequest {
        dataset: "synthetic_drum".to_string(),
        method: "cartesian".to_string(),
        kwargs: SimulationRequest::default_kwargs_for("cartesian"),
        nlt: 3,
        step: 15,
        dt: 1,
        tau: 6,
        start_index: 20,
        permanent_start: 15,
        no_species: false,
        target_species: "large".to_string(),
        predict_full_duration: true,
        ..SimulationRequest::default()
    };
    let b = run_gui_comparison(&base, &frames).unwrap();
    assert_eq!(b.target_species, "large");
    assert!(!b.discrepancy_heatmap_inhom.is_empty());

    let bench = run_method_benchmark(
        &base,
        &[
            "cartesian".to_string(),
            "cylindrical".to_string(),
            "voronoi".to_string(),
            "octree".to_string(),
        ],
        &frames,
    );
    assert_eq!(bench.len(), 4);
    for row in &bench {
        assert!(row.n_cells >= 1);
        assert!(row.hom_rmse_teneur.is_finite());
        assert!(row.inhom_l1_mean.is_finite());
        assert!(row.n_particles_dem_mean > 200.0);
    }
}

#[test]
fn test_gui_octree_split_planes_and_adaptive_zones_on_real_chunk() {
    let data_dir = std::path::Path::new("data/chunks");
    if !data_dir.exists() {
        return;
    }
    let src = LocalDataSource::new(data_dir);
    let ts_slice: Vec<usize> = (250..320).collect();
    let frames = src
        .read_timesteps(Some(&ts_slice))
        .expect("read real DEM chunks");

    let req_oct = SimulationRequest {
        dataset: "real_chunks".to_string(),
        method: "octree".to_string(),
        kwargs: SimulationRequest::default_kwargs_for("octree"),
        nlt: 2,
        step: 15,
        dt: 1,
        tau: 5,
        start_index: 250,
        permanent_start: 250,
        no_species: false,
        target_species: "small".to_string(),
        predict_full_duration: true,
        ..SimulationRequest::default()
    };
    let b_oct = run_gui_comparison(&req_oct, &frames).unwrap();
    assert!(b_oct.mesh.n_cells >= 2);
    assert!(b_oct.mesh.cells.iter().all(|c| !c.faces.is_empty()));

    let req_adap = SimulationRequest {
        dataset: "real_chunks".to_string(),
        method: "adaptive".to_string(),
        kwargs: SimulationRequest::default_kwargs_for("adaptive"),
        nlt: 2,
        step: 15,
        dt: 1,
        tau: 5,
        start_index: 250,
        permanent_start: 250,
        no_species: false,
        target_species: "small".to_string(),
        predict_full_duration: true,
        ..SimulationRequest::default()
    };
    let b_adap = run_gui_comparison(&req_adap, &frames).unwrap();
    assert!(!b_adap.mesh.split_planes.is_empty());
}
