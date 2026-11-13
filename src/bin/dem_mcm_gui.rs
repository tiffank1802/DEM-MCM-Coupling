//! Pure-Rust HTTP interactive 3-D visualization server for DEM-MCM Coupling.
//!
//! Binds on `0.0.0.0:<PORT>` (default 8080) with zero external dependencies,
//! exposes JSON endpoints powered by `dem_mcm_coupling::gui`, and serves the
//! interactive 3-D Web UI styled after `egui`.

use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use dem_mcm_coupling::gui::simulation::{
    run_gui_comparison, run_method_benchmark, GuiDatasetCache,
};
use dem_mcm_coupling::gui::state::{catalog_methods, SimulationRequest};
use dem_mcm_coupling::gui::web_ui::INDEX_HTML;
use dem_mcm_coupling::io::json::JsonValue;

const EGUI_ARCHITECTURE_NOTE: &str = r#"# Planification d'architecture Rust pour egui & Liquid Glass (gui-egui/)

1. Cœur scientifique indépendant du backend UI (src/gui/) — zéro dépendance crates.io :
   - camera3d.rs   : Camera3D (orbite sphérique yaw/pitch/distance/pan + vues Iso/XY/XZ/YZ).
   - mesh3d.rs     : Extraction géométrique 3-D exacte des 14 approches de maillage :
                     * Boîtes hexaédriques (Cartésien, Quantile, Octree axial)
                     * Secteurs annulaires courbes 3-D (Cylindrique)
                     * Clipper convexe de demi-espaces 3-D (Voronoï polyédrique & Octree oblique)
                     * Plans de coupe y_split / strates récursives (Adaptatif & MultiZone)
   - state.rs      : Ribbon UI (WorkspaceTab), critère Teneur en espèce (small/large),
                     visibilité indépendante des deux tailles de particules (4mm / 8mm).
   - simulation.rs : Cache Arc<BTreeMap<usize, DemFrame>>, construction des modèles sur
                     [start_index .. start_index + NLT*(step+tau)) vs prédiction sur toute la
                     durée de la DEM, métriques postprocess.py / postprocess_inhomogeneous.py
                     (Teneur RMSE/MAE, L1 normalisée, Discrepancy RMS, N_particles, lois p_ij(t)),
                     et benchmark multi-méthodes (run_method_benchmark).

2. Crate client natif/WASM prêt pour egui (gui-egui/) :
   - Cargo.toml          : dépend de dem-mcm-coupling (path = ".."), eframe 0.29, egui 0.29, egui_plot 0.29.
   - src/main.rs         : impl eframe::App for DemMcmEguiApp (Ribbon TopBottomPanel, thème Liquid Glass
                           sombre léger, chargement asynchrone mpsc + ctx.request_repaint()).
   - src/viewport3d.rs   : Rendu 3-D interactif via egui::Painter avec les deux tailles de particules
                           à l'échelle 1:2 (0.004 m vs 0.008 m), vitesses, maillage et flux Markov.
   - src/plots.rs        : Courbes DEM vs Homogène vs Inhomogène (RSD, Teneur, L1, RMS discrepancy,
                           Nombre de particules) via egui_plot::Plot et heatmaps de matrices P / P^(b).

3. Déploiement & Lancement :
   - Serveur HTTP pure-Rust : dem-mcm-gui --port 8080 --data-dir ./data/chunks
   - Export statique GitHub Pages : dem-mcm-gui --export-static _site --data-dir ./data/chunks
   - Client natif Desktop (OpenGL + crates.io) : cd gui-egui && cargo run --release
"#;

struct ServerState {
    data_dir: PathBuf,
    cache: Mutex<GuiDatasetCache>,
}

fn read_http_request(stream: &mut TcpStream) -> std::io::Result<(String, String, Vec<u8>)> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut req_line = String::new();
    if reader.read_line(&mut req_line)? == 0 {
        return Ok((String::new(), String::new(), Vec::new()));
    }
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let mut content_len = 0usize;
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_len = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_len.min(16 * 1024 * 1024)];
    if !body.is_empty() {
        reader.read_exact(&mut body)?;
    }
    Ok((method, path, body))
}

fn write_http_response(
    stream: &mut TcpStream,
    status: &str,
    content_type: &str,
    body: &[u8],
    send_body: bool,
) -> std::io::Result<()> {
    let hdr = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Access-Control-Allow-Methods: GET, POST, HEAD, OPTIONS\r\n\
         Access-Control-Allow-Headers: Content-Type\r\n\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(hdr.as_bytes())?;
    if send_body && !body.is_empty() {
        stream.write_all(body)?;
    }
    stream.flush()
}

fn json_error(status: &str, msg: &str) -> (String, String) {
    let mut o = JsonValue::object();
    o.insert("error", JsonValue::String(msg.to_string()));
    (status.to_string(), o.to_string_compact())
}

fn methods_catalog_json() -> String {
    let arr = catalog_methods()
        .into_iter()
        .map(|m| {
            let mut o = JsonValue::object();
            o.insert("id", JsonValue::String(m.id.to_string()));
            o.insert("name", JsonValue::String(m.name.to_string()));
            o.insert("family", JsonValue::String(m.family.to_string()));
            o.insert("uses_velocity", JsonValue::Bool(m.uses_velocity));
            o.insert("summary", JsonValue::String(m.summary.to_string()));
            o
        })
        .collect();
    let mut root = JsonValue::object();
    root.insert("methods", JsonValue::Array(arr));
    root.to_string_compact()
}

fn export_static_bundle(
    out_dir: &std::path::Path,
    data_dir: &std::path::Path,
) -> std::io::Result<()> {
    let static_dir = out_dir.join("static_data");
    std::fs::create_dir_all(&static_dir)?;
    std::fs::write(out_dir.join(".nojekyll"), b"")?;
    std::fs::write(out_dir.join("index.html"), INDEX_HTML.as_bytes())?;
    std::fs::write(
        static_dir.join("methods.json"),
        methods_catalog_json().as_bytes(),
    )?;
    std::fs::write(
        static_dir.join("egui_plan.txt"),
        EGUI_ARCHITECTURE_NOTE.as_bytes(),
    )?;

    let mut cache = GuiDatasetCache::new();
    let synth_frames = cache.get_synthetic();
    let all_methods: Vec<String> = catalog_methods()
        .into_iter()
        .map(|m| m.id.to_string())
        .collect();
    let bench_methods: Vec<String> = vec![
        "cartesian".into(),
        "cylindrical".into(),
        "quantile".into(),
        "octree".into(),
        "voronoi".into(),
        "physics".into(),
        "adaptive".into(),
    ];

    for sp in ["small", "large"] {
        for m in &all_methods {
            let req = SimulationRequest {
                dataset: "synthetic_drum".to_string(),
                method: m.clone(),
                kwargs: SimulationRequest::default_kwargs_for(m),
                target_species: sp.to_string(),
                ..SimulationRequest::default()
            };
            if let Ok(bundle) = run_gui_comparison(&req, &synth_frames) {
                let p = static_dir.join(format!("sim_synthetic_drum_{m}_{sp}.json"));
                std::fs::write(p, bundle.to_json().to_string_compact().as_bytes())?;
            }
        }
        let breq = SimulationRequest {
            dataset: "synthetic_drum".to_string(),
            target_species: sp.to_string(),
            ..SimulationRequest::default()
        };
        let rows = run_method_benchmark(&breq, &bench_methods, &synth_frames);
        let mut root = JsonValue::object();
        root.insert("target_species", JsonValue::String(sp.to_string()));
        root.insert("dataset", JsonValue::String("synthetic_drum".to_string()));
        root.insert(
            "rows",
            JsonValue::Array(rows.into_iter().map(|r| r.to_json()).collect()),
        );
        std::fs::write(
            static_dir.join(format!("benchmark_synthetic_drum_{sp}.json")),
            root.to_string_compact().as_bytes(),
        )?;
    }

    if data_dir.exists() {
        if let Ok(real_frames) = cache.get_or_load_real(data_dir) {
            let real_methods = ["cartesian", "cylindrical", "voronoi", "adaptive"];
            let sp = "small";
            for m in real_methods {
                let req = SimulationRequest {
                    dataset: "real_chunks".to_string(),
                    method: m.to_string(),
                    kwargs: SimulationRequest::default_kwargs_for(m),
                    nlt: 3,
                    step: 157,
                    dt: 1,
                    tau: 157,
                    start_index: 157,
                    permanent_start: 157,
                    no_species: false,
                    target_species: sp.to_string(),
                    predict_full_duration: true,
                    active_block: 0,
                };
                if let Ok(bundle) = run_gui_comparison(&req, &real_frames) {
                    let p = static_dir.join(format!("sim_real_chunks_{m}_{sp}.json"));
                    std::fs::write(p, bundle.to_json().to_string_compact().as_bytes())?;
                }
            }
            let breq = SimulationRequest {
                dataset: "real_chunks".to_string(),
                nlt: 3,
                step: 157,
                dt: 1,
                tau: 157,
                start_index: 157,
                permanent_start: 157,
                target_species: sp.to_string(),
                ..SimulationRequest::default()
            };
            let rm: Vec<String> = real_methods.iter().map(|s| s.to_string()).collect();
            let rows = run_method_benchmark(&breq, &rm, &real_frames);
            let mut root = JsonValue::object();
            root.insert("target_species", JsonValue::String(sp.to_string()));
            root.insert("dataset", JsonValue::String("real_chunks".to_string()));
            root.insert(
                "rows",
                JsonValue::Array(rows.into_iter().map(|r| r.to_json()).collect()),
            );
            std::fs::write(
                static_dir.join(format!("benchmark_real_chunks_{sp}.json")),
                root.to_string_compact().as_bytes(),
            )?;
        }
    }

    println!(
        "Static GitHub Pages bundle exported to {}",
        out_dir.display()
    );
    Ok(())
}
fn handle_client(mut stream: TcpStream, state: Arc<ServerState>) {
    let Ok((method, path, body)) = read_http_request(&mut stream) else {
        return;
    };
    if method.is_empty() {
        return;
    }
    if method == "OPTIONS" {
        let _ = write_http_response(&mut stream, "204 No Content", "text/plain", b"", false);
        return;
    }

    let send_body = method != "HEAD";
    let clean_path = path.split('?').next().unwrap_or(path.as_str());

    match (method.as_str(), clean_path) {
        ("GET" | "HEAD", "/" | "/index.html") => {
            let _ = write_http_response(
                &mut stream,
                "200 OK",
                "text/html; charset=utf-8",
                INDEX_HTML.as_bytes(),
                send_body,
            );
        }
        ("GET" | "HEAD", "/favicon.ico") => {
            let _ = write_http_response(&mut stream, "204 No Content", "image/x-icon", b"", false);
        }
        ("GET" | "HEAD", "/api/health") => {
            let mut o = JsonValue::object();
            o.insert("status", JsonValue::String("ok".to_string()));
            o.insert(
                "real_chunks_present",
                JsonValue::Bool(state.data_dir.exists()),
            );
            let _ = write_http_response(
                &mut stream,
                "200 OK",
                "application/json; charset=utf-8",
                o.to_string_compact().as_bytes(),
                send_body,
            );
        }
        ("GET" | "HEAD", "/api/methods") => {
            let body_str = methods_catalog_json();
            let _ = write_http_response(
                &mut stream,
                "200 OK",
                "application/json; charset=utf-8",
                body_str.as_bytes(),
                send_body,
            );
        }
        ("GET" | "HEAD", "/api/egui_plan") => {
            let _ = write_http_response(
                &mut stream,
                "200 OK",
                "text/plain; charset=utf-8",
                EGUI_ARCHITECTURE_NOTE.as_bytes(),
                send_body,
            );
        }
        ("POST", "/api/simulate") => {
            let raw = String::from_utf8_lossy(&body);
            let req = match JsonValue::parse(&raw) {
                Ok(v) => SimulationRequest::from_json(&v),
                Err(e) => {
                    let (st, txt) = json_error("400 Bad Request", &format!("Invalid JSON: {e}"));
                    let _ = write_http_response(
                        &mut stream,
                        &st,
                        "application/json; charset=utf-8",
                        txt.as_bytes(),
                        true,
                    );
                    return;
                }
            };

            let frames_res = {
                let mut guard = state.cache.lock().unwrap();
                if req.dataset == "real_chunks" {
                    guard.get_or_load_real(&state.data_dir)
                } else {
                    Ok(guard.get_synthetic())
                }
            };

            let resp = match frames_res.and_then(|frames| run_gui_comparison(&req, &frames)) {
                Ok(bundle) => ("200 OK".to_string(), bundle.to_json().to_string_compact()),
                Err(e) => json_error("422 Unprocessable Entity", &format!("{e}")),
            };
            let _ = write_http_response(
                &mut stream,
                &resp.0,
                "application/json; charset=utf-8",
                resp.1.as_bytes(),
                true,
            );
        }
        ("POST", "/api/benchmark") => {
            let raw = String::from_utf8_lossy(&body);
            let parsed = match JsonValue::parse(&raw) {
                Ok(v) => v,
                Err(e) => {
                    let (st, txt) = json_error("400 Bad Request", &format!("Invalid JSON: {e}"));
                    let _ = write_http_response(
                        &mut stream,
                        &st,
                        "application/json; charset=utf-8",
                        txt.as_bytes(),
                        true,
                    );
                    return;
                }
            };
            let req = SimulationRequest::from_json(&parsed);
            let methods: Vec<String> = parsed
                .get("methods")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .filter(|v: &Vec<String>| !v.is_empty())
                .unwrap_or_else(|| {
                    vec![
                        "cartesian".into(),
                        "cylindrical".into(),
                        "quantile".into(),
                        "octree".into(),
                        "voronoi".into(),
                        "physics".into(),
                        "adaptive".into(),
                    ]
                });

            let frames_res = {
                let mut guard = state.cache.lock().unwrap();
                if req.dataset == "real_chunks" {
                    guard.get_or_load_real(&state.data_dir)
                } else {
                    Ok(guard.get_synthetic())
                }
            };

            let resp = match frames_res {
                Ok(frames) => {
                    let rows = run_method_benchmark(&req, &methods, &frames);
                    let mut root = JsonValue::object();
                    root.insert("target_species", JsonValue::String(req.target_species));
                    root.insert("dataset", JsonValue::String(req.dataset));
                    root.insert(
                        "rows",
                        JsonValue::Array(rows.into_iter().map(|r| r.to_json()).collect()),
                    );
                    ("200 OK".to_string(), root.to_string_compact())
                }
                Err(e) => json_error("422 Unprocessable Entity", &format!("{e}")),
            };
            let _ = write_http_response(
                &mut stream,
                &resp.0,
                "application/json; charset=utf-8",
                resp.1.as_bytes(),
                true,
            );
        }
        _ => {
            let _ = write_http_response(
                &mut stream,
                "404 Not Found",
                "text/plain",
                b"Not Found",
                send_body,
            );
        }
    }
}

fn main() {
    let mut port: u16 = env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    let mut data_dir = PathBuf::from("data/chunks");
    let mut export_dir: Option<PathBuf> = None;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" if i + 1 < args.len() => {
                port = args[i + 1].parse().unwrap_or(port);
                i += 2;
            }
            "--data" | "--data-dir" if i + 1 < args.len() => {
                data_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--export-static" if i + 1 < args.len() => {
                export_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "-h" | "--help" => {
                println!(
                    "dem-mcm-gui — Interactive 3-D DEM-MCM Visualization Server\n\n\
                     USAGE:\n  dem-mcm-gui [--port <PORT>] [--data-dir <DIR>] [--export-static <OUT_DIR>]\n"
                );
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    if let Some(out_dir) = export_dir {
        if let Err(e) = export_static_bundle(&out_dir, &data_dir) {
            eprintln!("Static export failed: {e}");
            std::process::exit(1);
        }
        return;
    }

    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("Failed to bind {addr}: {e}");
        std::process::exit(1);
    });

    let mut cache = GuiDatasetCache::new();
    let _ = cache.get_synthetic(); // warm up synthetic rotating drum immediately

    let state = Arc::new(ServerState {
        data_dir,
        cache: Mutex::new(cache),
    });

    println!("DEM-MCM 3D Viewer server listening on http://{addr}");
    for stream in listener.incoming().flatten() {
        let st = Arc::clone(&state);
        thread::spawn(move || handle_client(stream, st));
    }
}
