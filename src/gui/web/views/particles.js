function renderParticlesView(b) {
  if (!b) return;
  const hm = b.hom_metrics, im = b.inhom_metrics;
  const mRange = [b.model_t_start, b.model_t_end];

  const kpi = document.getElementById('npart-kpi-lbl');
  if (kpi) {
    const nEndDem = hm.n_particles_total_dem.slice(-1)[0] ?? 0;
    const nEndInh = im.n_particles_total_mk.slice(-1)[0] ?? 0;
    kpi.textContent = `Final DEM=${nEndDem.toFixed(0)} · Inhom=${nEndInh.toFixed(0)} · Dérive Hom=${hm.mass_drift_pct.toFixed(3)}% / Inhom=${im.mass_drift_pct.toFixed(3)}%`;
  }

  const npScope = document.getElementById('npart-species').value;
  let npSeries = [];
  if (npScope === 'all_split') {
    npSeries = [
      { label: 'DEM Total', data: hm.n_particles_total_dem, color: '#e6ebf2', lw: 2.4 },
      { label: 'Inhom Total', data: im.n_particles_total_mk, color: '#9ec0dc', lw: 2.0, dash: [5, 4] },
      { label: 'DEM 4mm', data: hm.n_particles_small_dem, color: '#7dd3bd', lw: 2.0 },
      { label: 'Inhom 4mm', data: im.n_particles_small_mk, color: '#7dd3bd', lw: 1.5, dash: [3, 3] },
      { label: 'DEM 8mm', data: hm.n_particles_large_dem, color: '#eec078', lw: 2.0 },
      { label: 'Inhom 8mm', data: im.n_particles_large_mk, color: '#eec078', lw: 1.5, dash: [3, 3] },
    ];
  } else if (npScope === 'small') {
    npSeries = [
      { label: 'DEM 4mm', data: hm.n_particles_small_dem, color: '#e6ebf2', lw: 2.3 },
      { label: 'Hom 4mm', data: hm.n_particles_small_mk, color: '#9ec0dc', lw: 1.9, dash: [5, 4] },
      { label: 'Inhom 4mm', data: im.n_particles_small_mk, color: '#7dd3bd', lw: 2.3 },
    ];
  } else if (npScope === 'large') {
    npSeries = [
      { label: 'DEM 8mm', data: hm.n_particles_large_dem, color: '#e6ebf2', lw: 2.3 },
      { label: 'Hom 8mm', data: hm.n_particles_large_mk, color: '#9ec0dc', lw: 1.9, dash: [5, 4] },
      { label: 'Inhom 8mm', data: im.n_particles_large_mk, color: '#eec078', lw: 2.3 },
    ];
  } else {
    npSeries = [
      { label: 'DEM Total', data: hm.n_particles_total_dem, color: '#e6ebf2', lw: 2.4 },
      { label: 'Hom Total', data: hm.n_particles_total_mk, color: '#9ec0dc', lw: 2.0, dash: [5, 4] },
      { label: 'Inhom Total', data: im.n_particles_total_mk, color: '#7dd3bd', lw: 2.4 },
    ];
  }
  drawLineChart(document.getElementById('chart-npart'), hm.times_eval, npSeries, {
    blocks: b.blocks,
    modelRange: mRange,
  });

  const kCells = Math.min(parseInt(document.getElementById('npart-cell-count').value) || 4, b.top_cells.length);
  const popSeries = [];
  for (let i = 0; i < kCells; i++) {
    const cid = b.top_cells[i];
    const col = PALETTE[i % PALETTE.length];
    popSeries.push({ label: `DEM C${cid}`, data: b.top_cell_pop_dem.map(r => r[i]), color: col, lw: 2.1 });
    popSeries.push({ label: `Inhom C${cid}`, data: b.top_cell_pop_inhom.map(r => r[i]), color: col, lw: 1.5, dash: [4, 3] });
  }
  drawLineChart(document.getElementById('chart-top-pop'), hm.times_eval, popSeries, {
    modelRange: mRange,
  });
}
