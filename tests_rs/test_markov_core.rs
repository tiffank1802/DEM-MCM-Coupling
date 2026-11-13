//! Integration tests for `Markov` core orchestrator (`tests_rs/test_markov_core.rs`).

use std::collections::{BTreeMap, HashMap};

use dem_mcm_coupling::data::{DemFrame, InMemoryDataSource};
use dem_mcm_coupling::io::json::JsonValue;
use dem_mcm_coupling::{Markov, Matrix, PartitioningMethod};

fn synthetic_source(n_timesteps: usize, n_particles: usize) -> InMemoryDataSource {
    let mut frames = BTreeMap::new();
    for ts in 0..n_timesteps {
        let mut coords = Vec::with_capacity(n_particles);
        let mut vels = Vec::with_capacity(n_particles);
        let mut diams = Vec::with_capacity(n_particles);
        for p in 0..n_particles {
            let phase = (ts as f64) * 0.08 + (p as f64) * 0.13;
            coords.push([
                0.08 * phase.cos(),
                0.08 * phase.sin(),
                -0.02 + 0.04 * ((p % 7) as f64) / 6.0,
            ]);
            vels.push([
                -0.3 * phase.sin(),
                0.3 * phase.cos(),
                0.01 * ((p % 5) as f64 - 2.0),
            ]);
            diams.push(if p < n_particles / 2 { 0.004 } else { 0.008 });
        }
        frames.insert(ts, DemFrame::with_kinematics(ts, coords, vels, diams));
    }
    InMemoryDataSource::from_timesteps(frames)
}

#[test]
fn markov_init_accepts_all_canonical_methods_and_aliases() {
    for &m in PartitioningMethod::all() {
        let mk = Markov::new(m.as_str(), None).unwrap();
        assert_eq!(mk.method, m);
    }
    assert_eq!(
        Markov::new("physics_aware", None).unwrap().method,
        PartitioningMethod::Physics
    );
    assert_eq!(
        Markov::new("full_velocity_kmeans", None).unwrap().method,
        PartitioningMethod::PhysicsFullVel
    );
    assert_eq!(
        Markov::new("biclustering", None).unwrap().method,
        PartitioningMethod::SpectralBiclustering
    );
    assert_eq!(
        Markov::new("single_cell", None).unwrap().method,
        PartitioningMethod::Single
    );
    assert!(Markov::new("nonexistent_method", None).is_err());
}

#[test]
fn markov_end_to_end_cartesian_and_species_mask() {
    let src = synthetic_source(10, 60);
    let mut kw = HashMap::new();
    kw.insert("nx".into(), JsonValue::Number(2.0));
    kw.insert("ny".into(), JsonValue::Number(2.0));
    kw.insert("nz".into(), JsonValue::Number(2.0));

    let mut mk = Markov::new("cartesian", Some(&kw)).unwrap();
    mk.load_dem_data(&src).unwrap();
    let coords = mk.get_coords(&[0, 1, 2, 3, 4]).unwrap().to_vec();
    assert_eq!(coords.len(), 5 * 60);
    let vels = mk.get_velocities(&[0, 1]).unwrap().to_vec();
    assert_eq!(vels.len(), 2 * 60);

    mk.fit_partitioner(&coords).unwrap();
    assert_eq!(mk.partitioner.n_cells(), 8);

    // Unmasked initial state vector at t=0
    let sv_all = mk.build_initial_state_vector(0, None, true).unwrap();
    assert_eq!(sv_all.total_particles, 60);
    let sum_all: f64 = sv_all.phi.iter().sum();
    assert!((sum_all - 60.0).abs() < 1e-9);

    // Masked initial state vector: keep only small species (first 30 particles)
    let mask: Vec<bool> = (0..60).map(|i| i < 30).collect();
    let sv_small = mk.build_initial_state_vector(0, Some(&mask), true).unwrap();
    assert_eq!(sv_small.total_particles, 30);
    let sum_small: f64 = sv_small.phi.iter().sum();
    assert!((sum_small - 30.0).abs() < 1e-9);

    // Identity transition matrix preserves phi(0)
    let id = Matrix::identity(8);
    let traj = mk.propagate_markov(&sv_small.phi, &id, 10).unwrap();
    assert_eq!(traj.states.shape(), (11, 8));
    for step in 0..=10 {
        for c in 0..8 {
            assert!((traj.states.get(step, c) - sv_small.phi[c]).abs() < 1e-12);
        }
    }

    // Doubly-stochastic uniform matrix converges to uniform distribution
    let unif = Matrix::full(8, 8, 1.0 / 8.0);
    let traj_u = mk.propagate_markov(&sv_small.phi, &unif, 3).unwrap();
    for step in 1..=3 {
        let total: f64 = traj_u.states.row(step).iter().sum();
        assert!((total - 30.0).abs() < 1e-9);
        for c in 0..8 {
            assert!((traj_u.states.get(step, c) - 30.0 / 8.0).abs() < 1e-9);
        }
    }
}
