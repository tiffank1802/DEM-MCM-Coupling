//! Multi-method Benchmark focus view.

use dem_mcm_coupling::gui::MethodBenchmarkRow;
use egui::Ui;

pub fn draw_benchmark_view(ui: &mut Ui, bench_rows: &[MethodBenchmarkRow], run_bench: &mut bool) {
    if ui.button("Lancer Benchmark (7 méthodes)").clicked() {
        *run_bench = true;
    }
    ui.separator();
    egui::ScrollArea::both().show(ui, |ui| {
        egui::Grid::new("bench_grid").striped(true).show(ui, |ui| {
            ui.strong("Méthode");
            ui.strong("Vis/Cell");
            ui.strong("RMSE Teneur (H/I)");
            ui.strong("MAE Teneur (H/I)");
            ui.strong("L1 moy (H/I)");
            ui.strong("RMS disc (H/I)");
            ui.end_row();
            for r in bench_rows {
                ui.label(&r.method);
                ui.label(format!("{}/{}", r.n_visited, r.n_cells));
                ui.label(format!(
                    "{:.4} / {:.4}",
                    r.hom_rmse_teneur, r.inhom_rmse_teneur
                ));
                ui.label(format!(
                    "{:.4} / {:.4}",
                    r.hom_mae_teneur, r.inhom_mae_teneur
                ));
                ui.label(format!("{:.4} / {:.4}", r.hom_l1_mean, r.inhom_l1_mean));
                ui.label(format!("{:.4} / {:.4}", r.hom_rms_disc, r.inhom_rms_disc));
                ui.end_row();
            }
        });
    });
}
