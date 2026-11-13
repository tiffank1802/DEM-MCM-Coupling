function renderTeneurView(b) {
  if (!b) return;
  const hm = b.hom_metrics, im = b.inhom_metrics;
  const mRange = [b.model_t_start, b.model_t_end];
  const fmtT = v => v == null ? 'n/a' : `${v.toFixed(0)}`;

  const kpi = document.getElementById('teneur-kpi-lbl');
  if (kpi) {
    kpi.textContent = `Cible: ${b.target_species} · RMSE Hom ${hm.rmse_teneur.toFixed(4)} / Inhom ${im.rmse_teneur.toFixed(4)} · t50=${fmtT(hm.t50_dem)}`;
  }

  // Detailed metrics dropdown table
  document.getElementById('teneur-table-wrap').innerHTML = `
    <table>
      <thead><tr><th>Métrique (postprocess.py)</th><th>DEM réf.</th><th>Homogène</th><th>Inhomogène</th></tr></thead>
      <tbody>
        <tr><td>RMSE Teneur (${b.target_species})</td><td>0.0000</td><td>${hm.rmse_teneur.toFixed(4)}</td><td class="ok">${im.rmse_teneur.toFixed(4)}</td></tr>
        <tr><td>MAE Teneur (${b.target_species})</td><td>0.0000</td><td>${hm.mae_teneur.toFixed(4)}</td><td class="ok">${im.mae_teneur.toFixed(4)}</td></tr>
        <tr><td>Erreur Max Teneur</td><td>0.0000</td><td>${hm.max_err_teneur.toFixed(4)}</td><td class="ok">${im.max_err_teneur.toFixed(4)}</td></tr>
        <tr><td>RMSE RSD(Teneur)</td><td>0.0000</td><td>${hm.rmse_rsd.toFixed(4)}</td><td class="ok">${im.rmse_rsd.toFixed(4)}</td></tr>
        <tr><td>Temps mélange t50 / t90</td><td>${fmtT(hm.t50_dem)} / ${fmtT(hm.t90_dem)}</td><td>${fmtT(hm.t50_markov)} / ${fmtT(hm.t90_markov)}</td><td>${fmtT(im.t50_markov)} / ${fmtT(im.t90_markov)}</td></tr>
        <tr><td>|λ|_max / Diag(P) moy</td><td>—</td><td>${hm.lambda_max_small.toFixed(5)} / ${hm.diag_mean_small.toFixed(3)}</td><td>${im.lambda_max_small.toFixed(5)} / ${im.diag_mean_small.toFixed(3)}</td></tr>
      </tbody>
    </table>`;

  // Cell concentration chart
  const cSel = document.getElementById('teneur-cell-select').value;
  const cellsToShow = (cSel === 'all' || !cSel) ? b.top_cells.slice(0, 5) : [parseInt(cSel)];
  const tSeries = [];
  cellsToShow.forEach((cid, i) => {
    const col = PALETTE[i % PALETTE.length];
    tSeries.push({ label: `DEM C${cid}`, data: b.dem_conc_traj.map(r => r[cid]), color: col, lw: 2.2 });
    tSeries.push({ label: `Hom C${cid}`, data: b.hom_conc_traj.map(r => r[cid]), color: col, lw: 1.5, dash: [4, 3] });
    if (cellsToShow.length === 1) {
      tSeries.push({ label: `Inhom C${cid}`, data: b.inhom_conc_traj.map(r => r[cid]), color: '#7dd3bd', lw: 2.4 });
    }
  });
  drawLineChart(document.getElementById('chart-teneur-cell'), hm.times_eval, tSeries, {
    blocks: b.blocks,
    modelRange: mRange,
  });

  // Global mixing indicator chart
  const gm = document.getElementById('teneur-global-metric').value;
  const gTitle = document.getElementById('teneur-global-title');
  let gSeries = [];
  if (gm === 'entropy') {
    if (gTitle) gTitle.textContent = 'Entropie binaire normalisée H(t)';
    gSeries = [
      { label: 'H(t) DEM', data: hm.entropy_dem, color: '#e6ebf2', lw: 2.3 },
      { label: 'H(t) Homogène', data: hm.entropy_markov, color: '#9ec0dc', lw: 2.0, dash: [5, 4] },
      { label: 'H(t) Inhomogène', data: im.entropy_markov, color: '#7dd3bd', lw: 2.3 },
    ];
  } else if (gm === 'danckwerts') {
    if (gTitle) gTitle.textContent = 'Indice de ségrégation de Danckwerts I_D(t)';
    gSeries = [
      { label: 'I_D(t) DEM', data: hm.danckwerts_dem, color: '#e6ebf2', lw: 2.3 },
      { label: 'I_D(t) Homogène', data: hm.danckwerts_markov, color: '#9ec0dc', lw: 2.0, dash: [5, 4] },
      { label: 'I_D(t) Inhomogène', data: im.danckwerts_markov, color: '#7dd3bd', lw: 2.3 },
    ];
  } else {
    if (gTitle) gTitle.textContent = `RSD(Teneur) · t50 DEM=${fmtT(hm.t50_dem)} · Hom=${fmtT(hm.t50_markov)} · Inhom=${fmtT(im.t50_markov)}`;
    gSeries = [
      { label: 'RSD DEM', data: hm.rsd_dem, color: '#e6ebf2', lw: 2.4 },
      { label: 'RSD Homogène', data: hm.rsd_markov, color: '#9ec0dc', lw: 2.0, dash: [5, 4] },
      { label: 'RSD Inhomogène', data: im.rsd_markov, color: '#7dd3bd', lw: 2.4 },
    ];
  }
  drawLineChart(document.getElementById('chart-teneur-global'), hm.times_eval, gSeries, {
    blocks: b.blocks,
    modelRange: mRange,
  });
}
