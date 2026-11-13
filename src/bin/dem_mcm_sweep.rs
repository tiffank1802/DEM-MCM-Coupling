//! Pure-Rust CLI entrypoint for `dem-mcm-sweep`.

use std::env;
use std::process;

use dem_mcm_coupling::analyze_results::MarkovAnalyzer;
use dem_mcm_coupling::config::PartitioningMethod;
use dem_mcm_coupling::data::{DataSource, LocalDataSource};
use dem_mcm_coupling::metrics::validate_experiment;
use dem_mcm_coupling::partitioners::create_partitioner;
use dem_mcm_coupling::run_sweep::{get_configs, run_sweep, SweepMode, BASE_OUTPUT_DIR};

fn print_usage() {
    eprintln!(
        "dem-mcm-sweep — Pure-Rust DEM-MCM Markov Chain Coupling Engine\n\
         \n\
         USAGE:\n\
           dem-mcm-sweep [OPTIONS]\n\
         \n\
         OPTIONS:\n\
           --method <METHOD>      Partitioning method or 'all' (default: cartesian)\n\
           --data <PATH>          DEM input directory or parquet/csv file (default: data/chunks)\n\
           --output <DIR>         Output directory for experiments (default: {BASE_OUTPUT_DIR})\n\
           --diameter <FLOAT>     Optional particle diameter filter: 0.004 or 0.008\n\
           --inhomogeneous        Enable inhomogeneous mode (one P matrix per NLT block)\n\
           --no-species           Disable species masks (single transition matrix for all particles)\n\
           --list                 List generated sweep configurations without running\n\
           --limit <N>            Run at most N configurations from the sweep\n\
           --validate <DIR>       Validate saved experiments in DIR\n\
           -h, --help             Show this help message"
    );
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut method = "cartesian".to_string();
    let mut data_path = "data/chunks".to_string();
    let mut output_dir = BASE_OUTPUT_DIR.to_string();
    let mut diameter: Option<f64> = None;
    let mut inhomogeneous = false;
    let mut no_species = false;
    let mut list_only = false;
    let mut limit: Option<usize> = None;
    let mut validate_dir: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                return;
            }
            "--method" => {
                i += 1;
                method = args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("Missing value for --method");
                    process::exit(2);
                });
            }
            "--data" => {
                i += 1;
                data_path = args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("Missing value for --data");
                    process::exit(2);
                });
            }
            "--output" => {
                i += 1;
                output_dir = args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("Missing value for --output");
                    process::exit(2);
                });
            }
            "--diameter" => {
                i += 1;
                let s = args.get(i).unwrap_or_else(|| {
                    eprintln!("Missing value for --diameter");
                    process::exit(2);
                });
                diameter = Some(s.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --diameter value: {s}");
                    process::exit(2);
                }));
            }
            "--limit" => {
                i += 1;
                let s = args.get(i).unwrap_or_else(|| {
                    eprintln!("Missing value for --limit");
                    process::exit(2);
                });
                limit = Some(s.parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --limit value: {s}");
                    process::exit(2);
                }));
            }
            "--validate" => {
                i += 1;
                validate_dir = Some(args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("Missing value for --validate");
                    process::exit(2);
                }));
            }
            "--inhomogeneous" => inhomogeneous = true,
            "--no-species" => no_species = true,
            "--list" => list_only = true,
            other => {
                eprintln!("Unknown argument: {other}");
                print_usage();
                process::exit(2);
            }
        }
        i += 1;
    }

    if let Some(dir) = validate_dir {
        let src = LocalDataSource::new(&dir);
        let folders = src.list_experiments(None).unwrap_or_else(|e| {
            eprintln!("Error listing {dir}: {e}");
            process::exit(1);
        });
        let analyzer = MarkovAnalyzer::new();
        let mut all_ok = true;
        for f in folders {
            if f.starts_with('_') {
                continue;
            }
            match src.read_experiment(&f, None) {
                Ok(exp) => {
                    let report = validate_experiment(&f, &exp, 1e-3);
                    println!("{report}");
                    if let Ok(metrics) = analyzer.evaluate_experiment(&exp) {
                        println!(
                            "  RMSE(RSD) = {:.4}, MAE(RSD) = {:.4}",
                            metrics.rmse_rsd, metrics.mae_rsd
                        );
                    }
                    all_ok &= report.passed;
                }
                Err(e) => eprintln!("Skipping {f}: {e}"),
            }
        }
        if !all_ok {
            process::exit(1);
        }
        return;
    }

    if list_only {
        let diam = if no_species { None } else { diameter };
        let prefix = if no_species {
            "nospecies_"
        } else if inhomogeneous {
            "inhomogeneous_"
        } else {
            ""
        };
        let methods: Vec<&str> = if method == "all" {
            PartitioningMethod::all()
                .iter()
                .map(|m| m.as_str())
                .collect()
        } else {
            vec![method.as_str()]
        };
        for m in methods {
            match get_configs(m, diam) {
                Ok(cfgs) => {
                    println!("\n{} ({} configs):", m.to_uppercase(), cfgs.len());
                    for c in cfgs {
                        if let Ok(p) = create_partitioner(&c.method, &c.method_kwargs) {
                            println!(
                                "  {prefix}{} NLT={} step={} dt={} tau={} diameter={:?}",
                                p.label(),
                                c.nlt,
                                c.step,
                                c.dt,
                                c.tau,
                                c.particle_diameter
                            );
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    process::exit(1);
                }
            }
        }
        return;
    }

    let mode = if no_species {
        SweepMode::NoSpecies
    } else if inhomogeneous {
        SweepMode::Inhomogeneous
    } else {
        SweepMode::Homogeneous
    };

    let reader_src = LocalDataSource::new(&data_path);
    let mut writer_src = LocalDataSource::new(&output_dir);

    let diam = if no_species { None } else { diameter };
    let mut explicit_configs = if method == "all" {
        let mut list = Vec::new();
        for m in PartitioningMethod::all() {
            list.extend(get_configs(m.as_str(), diam).unwrap());
        }
        list
    } else {
        get_configs(&method, diam).unwrap_or_else(|e| {
            eprintln!("Error: {e}");
            process::exit(1);
        })
    };
    if inhomogeneous {
        for c in &mut explicit_configs {
            c.inhomogeneous = true;
        }
    }
    if let Some(lim) = limit {
        explicit_configs.truncate(lim);
    }

    struct SplitDataSource<'a> {
        reader: &'a LocalDataSource,
        writer: &'a mut LocalDataSource,
    }
    impl DataSource for SplitDataSource<'_> {
        fn read_timesteps(
            &self,
            timestep_indices: Option<&[usize]>,
        ) -> dem_mcm_coupling::Result<std::collections::BTreeMap<usize, dem_mcm_coupling::DemFrame>>
        {
            self.reader.read_timesteps(timestep_indices)
        }
        fn list_experiments(&self, prefix: Option<&str>) -> dem_mcm_coupling::Result<Vec<String>> {
            self.writer.list_experiments(prefix)
        }
        fn read_experiment(
            &self,
            folder_name: &str,
            prefix: Option<&str>,
        ) -> dem_mcm_coupling::Result<dem_mcm_coupling::ExperimentData> {
            self.writer.read_experiment(folder_name, prefix)
        }
        fn write_experiment(
            &mut self,
            payload: dem_mcm_coupling::data::ExperimentWritePayload<'_>,
        ) -> dem_mcm_coupling::Result<()> {
            self.writer.write_experiment(payload)
        }
    }

    let mut split = SplitDataSource {
        reader: &reader_src,
        writer: &mut writer_src,
    };
    let outcomes = run_sweep(&method, mode, Some(explicit_configs), diam, &mut split)
        .unwrap_or_else(|e| {
            eprintln!("Sweep failed: {e}");
            process::exit(1);
        });
    let ok = outcomes.iter().filter(|o| o.success).count();
    println!(
        "Completed {ok}/{} experiments into '{output_dir}'.",
        outcomes.len()
    );
    for o in &outcomes {
        if !o.success {
            eprintln!(
                "  [FAILED] {}: {}",
                o.folder_name,
                o.error.as_deref().unwrap_or("unknown")
            );
        }
    }
}
