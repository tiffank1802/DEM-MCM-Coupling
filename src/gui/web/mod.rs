//! Modular CAD-style Web UI assembler (`src/gui/web/`).
//!
//! Each stylesheet, HTML shell, state store, 3-D viewport, chart primitive, and
//! focus view lives in its own isolated source file and is embedded at compile
//! time via `include_str!` with zero external crate dependencies.

use std::sync::OnceLock;

pub const STYLES_CSS: &str = include_str!("styles.css");
pub const SHELL_HTML: &str = include_str!("shell.html");

pub const STORE_JS: &str = include_str!("store.js");
pub const UI_MACHINE_JS: &str = include_str!("ui_machine.js");
pub const VIEWPORT3D_JS: &str = include_str!("viewport3d.js");
pub const CHARTS_JS: &str = include_str!("charts.js");

pub const VIEW_SUMMARY_JS: &str = include_str!("views/summary.js");
pub const VIEW_MIXER3D_JS: &str = include_str!("views/mixer3d.js");
pub const VIEW_TENEUR_JS: &str = include_str!("views/teneur.js");
pub const VIEW_PARTICLES_JS: &str = include_str!("views/particles.js");
pub const VIEW_ERRORS_JS: &str = include_str!("views/errors.js");
pub const VIEW_MATRIX_EVOL_JS: &str = include_str!("views/matrix_evol.js");
pub const VIEW_MATRICES_JS: &str = include_str!("views/matrices.js");
pub const VIEW_BENCHMARK_JS: &str = include_str!("views/benchmark.js");
pub const VIEW_STUDY_JS: &str = include_str!("views/study.js");
pub const VIEW_PLAN_JS: &str = include_str!("views/plan.js");
pub const APP_JS: &str = include_str!("app.js");

static INDEX_CACHE: OnceLock<String> = OnceLock::new();

/// Build and cache the single-file interactive HTML document for HTTP serving
/// and static GitHub Pages export (`--export-static`).
pub fn index_html() -> &'static str {
    INDEX_CACHE.get_or_init(|| {
        let scripts = [
            UI_MACHINE_JS,
            STORE_JS,
            VIEWPORT3D_JS,
            CHARTS_JS,
            VIEW_SUMMARY_JS,
            VIEW_MIXER3D_JS,
            VIEW_TENEUR_JS,
            VIEW_PARTICLES_JS,
            VIEW_ERRORS_JS,
            VIEW_MATRIX_EVOL_JS,
            VIEW_MATRICES_JS,
            VIEW_BENCHMARK_JS,
            VIEW_STUDY_JS,
            VIEW_PLAN_JS,
            APP_JS,
        ]
        .join("\n\n");

        SHELL_HTML
            .replace("/*__STYLES__*/", STYLES_CSS)
            .replace("/*__SCRIPTS__*/", &scripts)
    })
}
