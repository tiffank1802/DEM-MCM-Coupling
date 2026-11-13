function renderSummaryView(b) {
  if (!b) return;
  const hm = b.hom_metrics, im = b.inhom_metrics;
  const mRange = [b.model_t_start, b.model_t_end];
  const frame = b.frames[Math.min(APP.frameIdx, b.frames.length - 1)] || b.frames[0];

  // 1. Mini 3D viewport
  renderViewport3D(document.getElementById('canvas3d-mini'), b, false);
  const lbl = document.getElementById('mini-3d-lbl');
  if (lbl) {
    lbl.innerHTML = `<b style="color:#fff">${b.method}</b> · ${b.diagnostics.n_visited}/${b.mesh.n_cells} cell. · t=${frame.timestep}`;
  }

  // 2. Mini Teneur / RSD
  drawLineChart(document.getElementById('chart-sum-teneur'), hm.times_eval, [
    { label: 'DEM', data: hm.rsd_dem, color: '#e6ebf2', lw: 2.0 },
    { label: 'Hom', data: hm.rsd_markov, color: '#9ec0dc', lw: 1.7, dash: [4, 3] },
    { label: 'Inhom', data: im.rsd_markov, color: '#7dd3bd', lw: 2.0 },
  ], { blocks: b.blocks, modelRange: mRange, compact: true });

  // 3. Mini Particle counts
  drawLineChart(document.getElementById('chart-sum-particles'), hm.times_eval, [
    { label: 'DEM Tot', data: hm.n_particles_total_dem, color: '#e6ebf2', lw: 2.0 },
    { label: 'Hom Tot', data: hm.n_particles_total_mk, color: '#9ec0dc', lw: 1.7, dash: [4, 3] },
    { label: 'Inhom Tot', data: im.n_particles_total_mk, color: '#7dd3bd', lw: 2.0 },
  ], { modelRange: mRange, compact: true });

  // 4. Mini Errors L1 & RMS
  drawLineChart(document.getElementById('chart-sum-errors'), hm.times_eval, [
    { label: 'L1 Hom', data: hm.l1_error_small, color: '#9ec0dc', lw: 1.7, dash: [4, 3] },
    { label: 'L1 Inhom', data: im.l1_error_small, color: '#7dd3bd', lw: 2.0 },
    { label: 'RMS Inhom', data: im.rms_discrepancy_target, color: '#eec078', lw: 1.6 },
  ], { blocks: b.blocks, modelRange: mRange, compact: true });

  // 5. Mini Inhomogeneous matrix components evolution
  const evol = b.target_species === 'large'
    ? (b.matrix_evolution_large || null)
    : (b.matrix_evolution_small || null);
  const comps = (evol && evol.components ? evol.components.filter(c => !c.is_diagonal) : (b.p_laws || []));
  if (comps.length) {
    const xs = comps[0].block_times;
    const series = [];
    comps.slice(0, 5).forEach((l, i) => {
      const col = PALETTE[i % PALETTE.length];
      series.push({ label: `${l.src}→${l.dst}`, data: l.values, x: xs, color: col, lw: 1.9, points: true, pointRadius: 3 });
      if (l.dense_times && l.dense_predicted) {
        series.push({ label: '', data: l.dense_predicted, x: l.dense_times, color: col, lw: 1.1, dash: [3, 3] });
      }
    });
    drawLineChart(document.getElementById('chart-sum-evol'), xs, series, { compact: true });
  }

  // 6. Summary compact table
  const fmtT = v => v == null ? 'n/a' : `${v.toFixed(0)}`;
  document.getElementById('summary-table-wrap').innerHTML = `
    <table>
      <thead>
        <tr><th>Modèle</th><th>RMSE(T)</th><th>MAE(T)</th><th>L¹ moy</th><th>RMS d.</th><th>RMSE(RSD)</th><th>t50</th><th>Dérive N</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Homogène</td>
          <td>${hm.rmse_teneur.toFixed(4)}</td>
          <td>${hm.mae_teneur.toFixed(4)}</td>
          <td>${hm.l1_mean_target.toFixed(4)}</td>
          <td>${hm.rms_discrepancy_mean.toFixed(4)}</td>
          <td>${hm.rmse_rsd.toFixed(4)}</td>
          <td>${fmtT(hm.t50_markov)}</td>
          <td>${hm.mass_drift_pct.toFixed(3)}%</td>
        </tr>
        <tr>
          <td class="ok">Inhomogène</td>
          <td>${im.rmse_teneur.toFixed(4)}</td>
          <td>${im.mae_teneur.toFixed(4)}</td>
          <td>${im.l1_mean_target.toFixed(4)}</td>
          <td>${im.rms_discrepancy_mean.toFixed(4)}</td>
          <td>${im.rmse_rsd.toFixed(4)}</td>
          <td>${fmtT(im.t50_markov)}</td>
          <td>${im.mass_drift_pct.toFixed(3)}%</td>
        </tr>
      </tbody>
    </table>`;
}
