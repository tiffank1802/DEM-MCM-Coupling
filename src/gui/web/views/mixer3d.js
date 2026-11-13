function updateCellInspector(cellHit) {
  const b = APP.bundleA;
  const box3d = document.getElementById('cell-info');
  const treeBox = document.getElementById('tree-inspector-body');

  if (!cellHit || !b) {
    if (box3d) box3d.classList.remove('show');
    if (treeBox) {
      treeBox.innerHTML = '<span class="hint-txt">Cliquez une cellule dans la vue 3D ou une composante p_{i→j} pour l\'inspecter ici.</span>';
    }
    return;
  }

  const fr = b.frames[APP.frameIdx] || b.frames[0];
  const ei = nearestEvalRow(b, fr.timestep);
  const tD = b.dem_conc_traj[ei]?.[cellHit.id] ?? fr.cell_conc_dem[cellHit.id] ?? 0;
  const tH = b.hom_conc_traj[ei]?.[cellHit.id] ?? 0;
  const tI = b.inhom_conc_traj[ei]?.[cellHit.id] ?? 0;
  const crm = b.inhom_metrics.cell_rmse_per_cell?.[cellHit.id] ?? 0;

  const html = `<b style="color:#fff">Cellule #${cellHit.id}</b> (zone ${cellHit.cell.zone})<br>` +
    `Particules N_${cellHit.id}(t=${fr.timestep}) : <b>${cellHit.pop}</b><br>` +
    `Teneur (${b.target_species}) : DEM <b>${tD.toFixed(3)}</b> · Hom <b>${tH.toFixed(3)}</b> · Inhom <b class="ok">${tI.toFixed(3)}</b><br>` +
    `Discrepancy RMSE : <b>${crm.toFixed(4)}</b>`;

  if (box3d) {
    box3d.innerHTML = html;
    box3d.classList.add('show');
  }
  if (treeBox) {
    treeBox.innerHTML = `<div style="font-size:11px;line-height:1.45;">${html}</div>`;
  }
}

function renderMixer3DView(b) {
  if (!b) return;
  const frame = b.frames[Math.min(APP.frameIdx, b.frames.length - 1)] || b.frames[0];
  const hud = document.getElementById('hud-info');
  if (hud) {
    hud.innerHTML =
      `<b style="color:var(--text);">${b.label}</b> · t = ${frame.timestep} / ${b.dem_t_max}<br>` +
      `Modèle [${b.model_t_start}..${b.model_t_end}] · Critère : teneur <b>${b.target_species}</b><br>` +
      `Teneur RMSE : Hom ${b.hom_metrics.rmse_teneur.toFixed(4)} · Inhom <span class="ok">${b.inhom_metrics.rmse_teneur.toFixed(4)}</span>`;
  }
  renderViewport3D(document.getElementById('canvas3d'), b, true);
}
