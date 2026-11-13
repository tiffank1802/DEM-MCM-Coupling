function renderErrorsView(b) {
  if (!b) return;
  const hm = b.hom_metrics, im = b.inhom_metrics;
  const mRange = [b.model_t_start, b.model_t_end];

  const kpi = document.getElementById('errors-kpi-lbl');
  if (kpi) {
    kpi.textContent = `L1 moy: Hom ${hm.l1_mean_target.toFixed(4)} / Inhom ${im.l1_mean_target.toFixed(4)} · RMS disc: Hom ${hm.rms_discrepancy_mean.toFixed(4)} / Inhom ${im.rms_discrepancy_mean.toFixed(4)}`;
  }

  const l1s = document.getElementById('l1-scope').value;
  const pickL1 = m => l1s === 'total'
    ? m.l1_error_total
    : (b.target_species === 'large' ? m.l1_error_large : m.l1_error_small);
  drawLineChart(document.getElementById('chart-l1'), hm.times_eval, [
    { label: 'L1 Homogène', data: pickL1(hm), color: '#9ec0dc', lw: 2.0, dash: [5, 4] },
    { label: 'L1 Inhomogène', data: pickL1(im), color: '#7dd3bd', lw: 2.3 },
  ], { blocks: b.blocks, modelRange: mRange });

  drawLineChart(document.getElementById('chart-rms-disc'), hm.times_eval, [
    { label: 'RMS disc. Homogène', data: hm.rms_discrepancy_target, color: '#9ec0dc', lw: 2.0, dash: [5, 4] },
    { label: 'RMS disc. Inhomogène', data: im.rms_discrepancy_target, color: '#7dd3bd', lw: 2.3 },
    { label: 'RMS total Inhom', data: im.rms_discrepancy_total, color: '#eec078', lw: 1.6 },
  ], { blocks: b.blocks, modelRange: mRange });

  const discGrid = document.getElementById('heat-disc-chain').value === 'hom'
    ? b.discrepancy_heatmap_hom
    : b.discrepancy_heatmap_inhom;
  drawRectHeatmap(document.getElementById('heat-discrepancy'), discGrid, b.heatmap_times, b.active_cell_ids);
}
