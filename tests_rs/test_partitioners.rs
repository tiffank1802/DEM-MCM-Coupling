//! Comprehensive integration tests for all 14 partitioners + save/load roundtrips.

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use dem_mcm_coupling::io::json::JsonValue;
use dem_mcm_coupling::partitioners::geometric::{
    CartesianPartitioner, CylindricalPartitioner, OctreePartitioner,
};
use dem_mcm_coupling::partitioners::physics::PhysicsAwarePartitioner;
use dem_mcm_coupling::partitioners::{create_partitioner, Partitioner};
use dem_mcm_coupling::rng::Rng;

fn temp_test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("dem_mcm_test_part_{name}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn rotating_drum_sample(n: usize) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let mut rng = Rng::seed_from_u64(123);
    let mut coords = Vec::with_capacity(n);
    let mut vels = Vec::with_capacity(n);
    for i in 0..n {
        let r = 0.09 * (rng.next_f64()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * rng.next_f64();
        let x = r * theta.cos();
        let y = r * theta.sin() - 0.02 * (if i % 3 == 0 { 1.0 } else { -1.0 });
        let z = -0.02 + 0.04 * rng.next_f64();
        coords.push([x, y, z]);
        // Solid body rotation + shear fluctuation
        let vx = -4.0 * y + 0.05 * rng.next_normal();
        let vy = 4.0 * x + 0.05 * rng.next_normal();
        let vz = 0.02 * rng.next_normal();
        vels.push([vx, vy, vz]);
    }
    (coords, vels)
}

#[test]
fn all_14_methods_fit_predict_and_roundtrip_artifacts() {
    let (coords, vels) = rotating_drum_sample(400);
    let root = temp_test_dir("all14");

    let specs: Vec<(&str, Vec<(&str, JsonValue)>)> = vec![
        (
            "cartesian",
            vec![
                ("nx", JsonValue::Number(3.0)),
                ("ny", JsonValue::Number(3.0)),
                ("nz", JsonValue::Number(2.0)),
            ],
        ),
        (
            "cylindrical",
            vec![
                ("nr", JsonValue::Number(2.0)),
                ("ntheta", JsonValue::Number(4.0)),
                ("nz", JsonValue::Number(2.0)),
                ("radial_mode", JsonValue::String("equal_area".into())),
            ],
        ),
        ("voronoi", vec![("n_cells", JsonValue::Number(12.0))]),
        (
            "quantile",
            vec![
                ("nx", JsonValue::Number(3.0)),
                ("ny", JsonValue::Number(3.0)),
                ("nz", JsonValue::Number(1.0)),
            ],
        ),
        (
            "octree",
            vec![
                ("max_particles", JsonValue::Number(80.0)),
                ("max_depth", JsonValue::Number(2.0)),
            ],
        ),
        (
            "physics",
            vec![
                ("n_cells", JsonValue::Number(10.0)),
                ("velocity_weight", JsonValue::Number(0.5)),
            ],
        ),
        (
            "physics_full_vel",
            vec![
                ("n_cells", JsonValue::Number(10.0)),
                ("velocity_weight", JsonValue::Number(0.5)),
            ],
        ),
        (
            "spectral",
            vec![
                ("n_cells", JsonValue::Number(8.0)),
                ("velocity_weight", JsonValue::Number(0.5)),
                ("n_neighbors", JsonValue::Number(8.0)),
                ("max_samples", JsonValue::Number(150.0)),
            ],
        ),
        (
            "dbscan",
            vec![
                ("eps", JsonValue::Number(0.8)),
                ("min_samples", JsonValue::Number(4.0)),
                ("velocity_weight", JsonValue::Number(0.5)),
                ("max_samples", JsonValue::Number(200.0)),
            ],
        ),
        (
            "gmm",
            vec![
                ("n_cells", JsonValue::Number(8.0)),
                ("velocity_weight", JsonValue::Number(0.5)),
                ("max_fit_samples", JsonValue::Number(200.0)),
            ],
        ),
        (
            "spectral_biclustering",
            vec![
                ("n_cells", JsonValue::Number(8.0)),
                ("n_col_clusters", JsonValue::Number(3.0)),
                ("method", JsonValue::String("log".into())),
                ("max_samples", JsonValue::Number(200.0)),
            ],
        ),
        (
            "adaptive",
            vec![
                ("y_split", JsonValue::Number(0.85)),
                ("y_split_mode", JsonValue::String("quantile".into())),
                ("top_method", JsonValue::String("single".into())),
                ("bottom_method", JsonValue::String("voronoi".into())),
            ],
        ),
        (
            "multizone",
            vec![
                ("y_mode", JsonValue::String("quantile".into())),
                ("zones", {
                    let mut z1 = JsonValue::object();
                    z1.insert("y_min", JsonValue::Number(0.0));
                    z1.insert("y_max", JsonValue::Number(0.8));
                    z1.insert("method", JsonValue::String("voronoi".into()));
                    let mut k1 = JsonValue::object();
                    k1.insert("n_cells", JsonValue::Number(6.0));
                    z1.insert("kwargs", k1);
                    let mut z2 = JsonValue::object();
                    z2.insert("y_min", JsonValue::Number(0.8));
                    z2.insert("y_max", JsonValue::Number(1.0));
                    z2.insert("method", JsonValue::String("single".into()));
                    JsonValue::Array(vec![z1, z2])
                }),
            ],
        ),
        ("single", vec![]),
    ];

    assert_eq!(specs.len(), 14);

    for (method_name, kv) in specs {
        let kw: HashMap<String, JsonValue> =
            kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        let mut p = create_partitioner(method_name, &kw).unwrap();
        if matches!(
            method_name,
            "physics"
                | "physics_full_vel"
                | "spectral"
                | "dbscan"
                | "gmm"
                | "spectral_biclustering"
        ) {
            p.set_use_velocity(true);
            p.set_dem_velocities(Some(&vels));
        }
        p.fit(&coords).unwrap();
        assert!(p.n_cells() >= 1, "{method_name} returned 0 cells");

        let vel_arg = if matches!(
            method_name,
            "physics"
                | "physics_full_vel"
                | "spectral"
                | "dbscan"
                | "gmm"
                | "spectral_biclustering"
        ) {
            Some(vels.as_slice())
        } else {
            None
        };
        let states = p.compute_states(&coords, vel_arg).unwrap();
        assert_eq!(states.len(), coords.len());
        if method_name != "dbscan" {
            for &s in &states {
                assert!(
                    s >= 0 && (s as usize) < p.n_cells(),
                    "{method_name}: out of range {s}"
                );
            }
        }

        // Save artifacts & reload into a fresh instance
        let save_dir = root.join(method_name);
        p.save(&save_dir).unwrap();
        let mut p_loaded = create_partitioner(method_name, &kw).unwrap();
        if matches!(
            method_name,
            "physics"
                | "physics_full_vel"
                | "spectral"
                | "dbscan"
                | "gmm"
                | "spectral_biclustering"
        ) {
            p_loaded.set_use_velocity(true);
        }
        p_loaded.load(&save_dir).unwrap();
        let states_reloaded = p_loaded.compute_states(&coords, vel_arg).unwrap();
        assert_eq!(
            states, states_reloaded,
            "Save/load mismatch for {method_name}"
        );
    }

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn cartesian_matches_exact_eps_and_round_ties_to_even() {
    let mut cart = CartesianPartitioner::new(2, 2, 2);
    let coords = vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    cart.fit(&coords).unwrap();
    let b = cart.bounds.unwrap();
    assert!((b[0] - (-0.001)).abs() < 1e-12);
    assert!((b[1] - 1.001).abs() < 1e-12);
    assert_eq!(cart.n_cells(), 8);
    let s = cart.compute_states(&coords, None).unwrap();
    assert_eq!(s[0], 0);
    assert_eq!(s[1], 7); // 1 + 1*2 + 1*4
}

#[test]
fn cylindrical_out_of_zone_returns_minus_one_and_equal_dr_works() {
    let mut cyl = CylindricalPartitioner::new(3, 4, 2, "equal_dr");
    cyl.r_max_limit_input = Some(0.1);
    let coords = vec![[0.02, 0.0, 0.0], [0.08, 0.0, 0.5], [0.20, 0.0, 0.25]];
    cyl.fit(&coords).unwrap();
    let states = cyl.compute_states(&coords, None).unwrap();
    assert!(states[0] >= 0);
    assert!(states[1] >= 0);
    assert_eq!(states[2], -1);
}

#[test]
fn octree_all_oblique_methods_and_normalize_work() {
    let (coords, _) = rotating_drum_sample(250);
    for &om in OctreePartitioner::OBLIQUE_METHODS {
        let mut oct =
            OctreePartitioner::new(40, 2, Some("normalize".to_string()), Some(om.to_string()));
        oct.fit(&coords).unwrap();
        assert!(oct.n_cells() >= 2, "oblique={om} produced <2 leaves");
        let st = oct.compute_states(&coords, None).unwrap();
        assert_eq!(st.len(), coords.len());
        assert!(st.iter().all(|&v| v >= 0 && (v as usize) < oct.n_cells()));
    }
}

#[test]
fn physics_aware_components_mode_uses_6_features() {
    let (coords, vels) = rotating_drum_sample(150);
    let mut pa = PhysicsAwarePartitioner::new(6, 0.5, 42, "components");
    pa.set_use_velocity(true);
    pa.set_dem_velocities(Some(&vels));
    pa.fit(&coords).unwrap();
    assert_eq!(pa.n_features, 6);
    assert!(pa.label().contains("compvel"));
    let st = pa.compute_states(&coords, Some(&vels)).unwrap();
    assert_eq!(st.len(), 150);
}
