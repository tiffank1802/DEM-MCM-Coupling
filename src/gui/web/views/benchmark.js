function renderDualView() {
  const bA = APP.bundleA;
  const bB = APP.bundleB;
  if (!bA) return;

  const lblA = document.getElementById('dual-a-lbl');
  const lblB = document.getElementById('dual-b-lbl');
  if (lblA) lblA.textContent = `${bA.method} (${bA.mesh.n_cells}c)`;
  if (lblB) lblB.textContent = bB ? `${bB.method} (${bB.mesh.n_cells}c)` : '—';

  renderViewport3D(document.getElementById('canvas3d-a'), bA, false);
  if (bB) {
    renderViewport3D(document.getElementById('canvas3d-b'), bB, false);
    const hm = bA.hom_metrics, im = bA.inhom_metrics;
    const mRange = [bA.model_t_start, bA.model_t_end];
    drawLineChart(document.getElementById('chart-dual'), hm.times_eval, [
      { label: 'DEM réf. RSD', data: hm.rsd_dem, color: '#e6ebf2', lw: 2.1 },
      { label: `${bA.method} Inhom RSD`, data: im.rsd_markov, color: '#7dd3bd', lw: 2.2 },
      { label: `${bB.method} Inhom RSD`, data: bB.inhom_metrics.rsd_markov, color: '#eec078', lw: 2.2 },
      { label: `${bA.method} L1`, data: im.l1_error_small, color: '#9ec0dc', lw: 1.5, dash: [4, 3] },
      { label: `${bB.method} L1`, data: bB.inhom_metrics.l1_error_small, color: '#e5989b', lw: 1.5, dash: [4, 3] },
    ], { modelRange: mRange });
  }
}

function renderBenchmarkView() {
  renderDualView();
  const rows = APP.benchRows;
  if (!rows) return;
  const metricKey = document.getElementById('bench-metric-key')?.value || 'rmse_teneur';
  drawBarBenchmark(document.getElementById('chart-bench'), rows, metricKey);

  document.getElementById('bench-table-wrap').innerHTML = `
    <table>
      <thead>
        <tr>
          <th>Méthode</th><th>Vis./Cell</th><th>N_DEM</th><th>N_Inhom</th>
          <th>RMSE Teneur (H/I)</th><th>MAE Teneur (H/I)</th>
          <th>L¹ moy (H/I)</th><th>RMS disc (H/I)</th><th>RMSE RSD (H/I)</th><th>t50 (H/I)</th>
        </tr>
      </thead>
      <tbody>
        ${rows.map(r => `<tr>
          <td class="soft">${r.method}</td>
          <td>${r.n_visited}/${r.n_cells}</td>
          <td>${r.n_particles_dem_mean.toFixed(0)}</td>
          <td>${r.n_particles_inhom_final.toFixed(0)}</td>
          <td>${r.hom_rmse_teneur.toFixed(4)} / <b class="ok">${r.inhom_rmse_teneur.toFixed(4)}</b></td>
          <td>${r.hom_mae_teneur.toFixed(4)} / <b class="ok">${r.inhom_mae_teneur.toFixed(4)}</b></td>
          <td>${r.hom_l1_mean.toFixed(4)} / <b class="ok">${r.inhom_l1_mean.toFixed(4)}</b></td>
          <td>${r.hom_rms_disc.toFixed(4)} / <b class="ok">${r.inhom_rms_disc.toFixed(4)}</b></td>
          <td>${r.hom_rmse_rsd.toFixed(4)} / <b class="ok">${r.inhom_rmse_rsd.toFixed(4)}</b></td>
          <td>${r.t50_hom == null ? '-' : r.t50_hom.toFixed(0)} / ${r.t50_inhom == null ? '-' : r.t50_inhom.toFixed(0)}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}
