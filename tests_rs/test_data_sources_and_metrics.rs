//! Integration tests for DataSource implementations, I/O codecs, and physics metrics.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use dem_mcm_coupling::config::{ExperimentData, SpeciesExperimentData};
use dem_mcm_coupling::data::{DataSource, InMemoryDataSource, LocalDataSource};
use dem_mcm_coupling::io::json::JsonValue;
use dem_mcm_coupling::metrics::{
    clean_transition_matrix, concentration_from_s, detect_convention, entropy_concentration,
    entropy_from_s, fit_probability_evolution, intensity_of_segregation, mixing_times,
    propagate_markov, propagate_markov_inhomogeneous, rsd_concentration, rsd_from_s,
    standardize_transition_matrix, stationary_distribution, validate_experiment, Convention,
};
use dem_mcm_coupling::utils::data_source_from_uri;
use dem_mcm_coupling::Matrix;

#[test]
fn convention_detection_standardization_and_nan_propagation() {
    let p_row = Matrix::from_rows(&[vec![0.7, 0.3], vec![0.4, 0.6]]).unwrap();
    assert_eq!(detect_convention(&p_row, 1e-6), Convention::Row);
    let (std_r, trans_r) = standardize_transition_matrix(&p_row);
    assert!(!trans_r);
    assert_eq!(std_r, p_row);

    let p_col = p_row.transpose();
    assert_eq!(detect_convention(&p_col, 1e-6), Convention::Column);
    let (std_c, trans_c) = standardize_transition_matrix(&p_col);
    assert!(trans_c);
    assert_eq!(std_c, p_row);

    // Unvisited NaN row propagates NaN to column sums -> clean zeros out NaN rows cleanly
    let p_nan = Matrix::from_rows(&[
        vec![0.8, 0.2, 0.0],
        vec![0.3, 0.7, 0.0],
        vec![f64::NAN, f64::NAN, f64::NAN],
    ])
    .unwrap();
    let (cleaned, active) = clean_transition_matrix(&p_nan, 0.5);
    assert_eq!(active, vec![true, true, false]);
    assert_eq!(cleaned.get(2, 0), 0.0);

    // Propagate conserves active mass and stays zero on unvisited states
    let s0 = vec![10.0, 20.0, 99.0];
    let times: Vec<i64> = (0..10).collect();
    let (traj, tm) = propagate_markov(&s0, &cleaned, &times, 0, 2, &active).unwrap();
    assert_eq!(tm, vec![0, 2, 4, 6, 8]);
    assert_eq!(traj.shape(), (5, 3));
    for r in 0..5 {
        assert!((traj.row(r).iter().sum::<f64>() - 30.0).abs() < 1e-10);
        assert_eq!(traj.get(r, 2), 0.0);
    }
}

#[test]
fn inhomogeneous_propagation_uses_block_duration_when_step_and_nlt_known() {
    let b0 = Matrix::from_rows(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
    let b1 = Matrix::from_rows(&[vec![0.0, 1.0], vec![1.0, 0.0]]).unwrap();
    let times: Vec<i64> = (0..=15).collect();
    let act = vec![true, true];
    // tau=3, step=3 -> block_duration = 6:
    // t=1 (curr=0 -> block 0: id) -> [10, 0]
    // t=2 (curr=3 -> block 0: id) -> [10, 0]
    // t=3 (curr=6 -> block 1: swap) -> [0, 10]
    let (traj, tm) = propagate_markov_inhomogeneous(
        &[10.0, 0.0],
        &[b0, b1],
        &times,
        0,
        3,
        &act,
        Some(3),
        Some(2),
    )
    .unwrap();
    assert_eq!(tm, vec![0, 3, 6, 9, 12, 15]);
    assert_eq!(traj.row(1), &[10.0, 0.0]);
    assert_eq!(traj.row(2), &[10.0, 0.0]);
    assert_eq!(traj.row(3), &[0.0, 10.0]);
}

#[test]
fn mixing_metrics_stationary_and_poly_fit_behavior() {
    let s_small = Matrix::from_rows(&[vec![10.0, 0.0], vec![7.0, 3.0], vec![5.0, 5.0]]).unwrap();
    let s_large = Matrix::from_rows(&[vec![0.0, 10.0], vec![3.0, 7.0], vec![5.0, 5.0]]).unwrap();
    let act = vec![true, true];

    let rsd_s = rsd_from_s(&s_small, &act);
    assert!((rsd_s[0] - 1.0).abs() < 1e-12);
    assert!(rsd_s[2].abs() < 1e-12);

    let conc = concentration_from_s(&s_small, &s_large);
    assert!((conc.get(0, 0) - 1.0).abs() < 1e-12);
    assert!((conc.get(2, 0) - 0.5).abs() < 1e-12);

    let rsd_c = rsd_concentration(&s_small, &s_large, &act, &act);
    assert!((rsd_c[0] - 1.0).abs() < 1e-12);
    assert!(rsd_c[2].abs() < 1e-12);

    let ent_c = entropy_concentration(&s_small, &s_large, &act, &act, true);
    assert!(ent_c[0].abs() < 1e-12);
    assert!((ent_c[2] - 1.0).abs() < 1e-12);

    let ent_s = entropy_from_s(&s_small, &act);
    assert!((ent_s[2] - 2.0f64.ln()).abs() < 1e-12);

    let iseg = intensity_of_segregation(&s_small, &s_large, &act, &act);
    assert!((iseg[0] - 1.0).abs() < 1e-12);
    assert!(iseg[2].abs() < 1e-12);

    let mt = mixing_times(&rsd_c, &[0.0, 1.0, 2.0], &[0.5, 0.1]);
    assert_eq!(mt[0].1, Some(1.0));
    assert_eq!(mt[1].1, Some(2.0));

    // Stationary distribution
    let p = Matrix::from_rows(&[vec![0.8, 0.2], vec![0.4, 0.6]]).unwrap();
    let pi = stationary_distribution(&p).unwrap();
    assert!((pi[0] - 2.0 / 3.0).abs() < 1e-9);
    assert!((pi[1] - 1.0 / 3.0).abs() < 1e-9);

    // Parsimonious polynomial fitting: linear data selects degree 1
    let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let ys: Vec<f64> = xs.iter().map(|&x| 3.0 * x + 1.0).collect();
    let fit = fit_probability_evolution(&xs, &ys, 3).unwrap();
    assert_eq!(fit.degree, 1);
    assert!(fit.r2 > 0.999999);
}

#[test]
fn validate_experiment_detects_negative_and_bad_stochastic_matrices() {
    let mut sp = HashMap::new();
    sp.insert(
        "small".to_string(),
        SpeciesExperimentData {
            p: Some(Matrix::from_rows(&[vec![-0.1, 1.1], vec![0.5, 0.5]]).unwrap()),
            p_blocks: None,
            s_matrix: Matrix::from_rows(&[vec![10.0, 5.0]]).unwrap(),
            times: vec![0],
        },
    );
    let bad_exp = ExperimentData {
        config: JsonValue::object(),
        stats: JsonValue::object(),
        inhomogeneous: false,
        inhomogeneous_metadata: None,
        states_matrix: None,
        species: sp,
    };
    let rep = validate_experiment("bad_exp", &bad_exp, 1e-3);
    assert!(!rep.passed);
}

#[test]
fn local_datasource_csv_and_uri_factory() {
    let dir: PathBuf =
        std::env::temp_dir().join(format!("dem_mcm_test_csv_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    fs::write(
        dir.join("data_0.csv"),
        "coordinates:0,coordinates:1,coordinates:2,Velocity:0,Velocity:1,Velocity:2,Diameter\n\
         0.01,0.02,0.03,0.1,0.2,0.3,0.004\n\
         -0.01,-0.02,-0.03,-0.1,-0.2,-0.3,0.008\n",
    )
    .unwrap();
    fs::write(
        dir.join("data_5.csv"),
        "coordinates:0,coordinates:1,coordinates:2,Velocity:0,Velocity:1,Velocity:2,Diameter\n\
         0.02,0.03,0.04,0.2,0.3,0.4,0.004\n\
         -0.02,-0.03,-0.04,-0.2,-0.3,-0.4,0.008\n",
    )
    .unwrap();

    let src = data_source_from_uri(dir.to_str().unwrap()).unwrap();
    let frames = src.read_timesteps(None).unwrap();
    assert_eq!(frames.len(), 2);
    assert!(frames.contains_key(&0));
    assert!(frames.contains_key(&5));
    assert_eq!(frames[&0].n_particles(), 2);
    assert!((frames[&5].coordinates[0][0] - 0.02).abs() < 1e-12);

    // Subsetting timesteps works
    let sub = src.read_timesteps(Some(&[5])).unwrap();
    assert_eq!(sub.len(), 1);
    assert!(sub.contains_key(&5));

    // DataSource trait on LocalDataSource vs InMemoryDataSource
    let mem = InMemoryDataSource::from_timesteps(frames);
    assert_eq!(mem.read_timesteps(Some(&[0])).unwrap().len(), 1);
    let _local = LocalDataSource::new(&dir);

    let _ = fs::remove_dir_all(&dir);
}
