//! Embedded Liquid-Glass Ribbon UI for DEM-MCM 3-D Mesh & Markov Chain Studio.
//!
//! Architectural principles:
//! - Pure Ribbon navigation (no heavy titled cards or cluttered sidebars).
//! - Soft dark Liquid Glass palette (`backdrop-filter: blur`, subtle translucent slate, no harsh blues).
//! - Both particle sizes rendered faithfully in 3-D at exact 2:1 diameter ratio (`0.004 m` vs `0.008 m`).
//! - Primary physical criterion: **teneur en l'espèce** (`S_target / (S_small + S_large)`),
//!   particle counts (`N_small`, `N_large`, `N_total`), and full `postprocess.py` /
//!   `postprocess_inhomogeneous.py` error metrics across the full DEM duration.

pub const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DEM-MCM Studio · Liquid Glass · Maillage 3D &amp; Teneur</title>
<style>
  :root {
    --bg-base: #15181e;
    --bg-grad-a: rgba(186, 204, 224, 0.065);
    --bg-grad-b: rgba(154, 202, 192, 0.050);
    --glass-1: rgba(255, 255, 255, 0.042);
    --glass-2: rgba(255, 255, 255, 0.072);
    --glass-3: rgba(255, 255, 255, 0.105);
    --glass-border: rgba(255, 255, 255, 0.115);
    --glass-highlight: rgba(255, 255, 255, 0.16);
    --text: #e6ebf2;
    --text-soft: #b6c0ce;
    --muted: #8490a2;
    --accent-ice: #cfe0f0;
    --accent-sage: #7dd3bd;
    --accent-sky: #9ec0dc;
    --accent-amber: #eec078;
    --accent-rose: #e5989b;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background:
      radial-gradient(ellipse 65% 55% at 14% 8%, var(--bg-grad-a), transparent 70%),
      radial-gradient(ellipse 60% 50% at 88% 92%, var(--bg-grad-b), transparent 70%),
      linear-gradient(160deg, #13161b 0%, #191d24 52%, #151920 100%);
    color: var(--text);
    font-family: var(--sans);
    font-size: 12.5px;
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 6px 8px;
  }

  .glass {
    background: linear-gradient(155deg, rgba(255,255,255,0.068), rgba(255,255,255,0.025));
    backdrop-filter: blur(18px) saturate(125%);
    -webkit-backdrop-filter: blur(18px) saturate(125%);
    border: 1px solid var(--glass-border);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 8px 26px rgba(0,0,0,0.28);
    border-radius: 12px;
  }

  /* ── RIBBON CHROME ─────────────────────────────────────────────────────── */
  .ribbon-shell {
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    overflow: visible;
  }
  .ribbon-tabs {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 5px 10px 4px 10px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 7px;
    font-weight: 600;
    letter-spacing: 0.01em;
    color: var(--text);
    margin-right: 10px;
    white-space: nowrap;
  }
  .brand-pill {
    font-family: var(--mono);
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 999px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    color: var(--accent-ice);
  }
  .rtab {
    background: transparent;
    color: var(--muted);
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.14s ease;
    white-space: nowrap;
  }
  .rtab:hover { background: var(--glass-1); color: var(--text); }
  .rtab.active {
    background: var(--glass-2);
    color: var(--text);
    border-color: rgba(255,255,255,0.16);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.18);
  }
  .spacer { flex: 1; }
  .status-badge {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent-sage);
    display: none;
    margin-right: 6px;
  }
  .status-badge.show { display: inline-block; }

  .ribbon-body {
    display: flex;
    align-items: stretch;
    gap: 2px;
    padding: 6px 10px;
    overflow-x: auto;
    min-height: 62px;
  }
  .rgroup {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 0 10px 0 6px;
    border-right: 1px solid rgba(255,255,255,0.08);
    flex-shrink: 0;
  }
  .rgroup:last-child { border-right: none; }
  .rgroup-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: nowrap;
  }
  .rgroup-label {
    font-size: 9.5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    text-align: center;
    margin-top: 4px;
    user-select: none;
  }

  .rfield { display: flex; flex-direction: column; gap: 2px; }
  .rfield label { font-size: 10px; color: var(--muted); white-space: nowrap; }

  select, input[type="number"], input[type="text"] {
    background: rgba(15, 18, 24, 0.55);
    color: var(--text);
    border: 1px solid rgba(255,255,255,0.11);
    border-radius: 7px;
    padding: 4px 7px;
    font-size: 11.5px;
    font-family: var(--sans);
    transition: border-color 0.12s;
  }
  select:focus, input:focus { outline: none; border-color: rgba(207, 224, 240, 0.45); }
  input[type="number"] { width: 62px; font-family: var(--mono); }

  .btn {
    background: var(--glass-2);
    color: var(--text);
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 11.5px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.13s ease;
  }
  .btn:hover { background: var(--glass-3); border-color: rgba(255,255,255,0.22); }
  .btn.primary {
    background: linear-gradient(135deg, rgba(207,224,240,0.22), rgba(125,211,189,0.18));
    border-color: rgba(207,224,240,0.34);
    color: #fff;
    font-weight: 600;
  }
  .btn.primary:hover {
    background: linear-gradient(135deg, rgba(207,224,240,0.30), rgba(125,211,189,0.25));
  }
  .btn.sm { padding: 3px 8px; font-size: 11px; border-radius: 6px; }
  .pill-chk {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 8px;
    border-radius: 999px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.09);
    font-size: 11px;
    color: var(--text-soft);
    cursor: pointer;
    white-space: nowrap;
    user-select: none;
  }
  .pill-chk input { accent-color: var(--accent-sage); cursor: pointer; }

  /* ── TELEMETRY STRIP (NO CARDS) ─────────────────────────────────────────── */
  .telemetry {
    display: flex;
    gap: 6px;
    align-items: center;
    flex-wrap: wrap;
    flex-shrink: 0;
  }
  .tpill {
    display: flex;
    align-items: baseline;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.038);
    border: 1px solid rgba(255,255,255,0.085);
    backdrop-filter: blur(12px);
  }
  .tpill .k { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .tpill .v { font-size: 12px; font-weight: 600; font-family: var(--mono); color: var(--text); }
  .ok { color: var(--accent-sage) !important; }
  .soft { color: var(--accent-sky) !important; }
  .amber { color: var(--accent-amber) !important; }
  .rose { color: var(--accent-rose) !important; }

  /* ── CLEAN WORKSPACE PANES (NO TITLED CARDS) ─────────────────────────────── */
  .workspace {
    flex: 1;
    min-height: 0;
    display: flex;
    position: relative;
  }
  .pane {
    flex: 1;
    min-height: 0;
    min-width: 0;
    display: none;
    gap: 8px;
  }
  .pane.active { display: flex; }
  .col { display: flex; flex-direction: column; gap: 8px; min-height: 0; min-width: 0; }
  .surface {
    position: relative;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
  }
  .subtle-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 4px;
    font-size: 11px;
    color: var(--text-soft);
    flex-shrink: 0;
  }
  .viewport-wrap {
    flex: 1;
    position: relative;
    border-radius: 10px;
    background: radial-gradient(circle at 50% 42%, rgba(44,50,62,0.55) 0%, rgba(16,19,25,0.85) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    overflow: hidden;
    min-height: 240px;
  }
  canvas { display: block; width: 100%; height: 100%; }

  .hud-tl {
    position: absolute;
    top: 8px;
    left: 8px;
    padding: 6px 10px;
    border-radius: 9px;
    background: rgba(21, 24, 30, 0.62);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.10);
    font-size: 11px;
    line-height: 1.45;
    pointer-events: none;
    color: var(--text-soft);
  }
  .hud-tr {
    position: absolute;
    top: 8px;
    right: 8px;
    padding: 5px 9px;
    border-radius: 9px;
    background: rgba(21, 24, 30, 0.62);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.10);
    font-size: 10.5px;
    color: var(--text-soft);
    display: flex;
    align-items: center;
    gap: 10px;
    pointer-events: none;
  }
  .dot-s { display:inline-block; width:7px; height:7px; border-radius:50%; background:#7dd3c8; border:1px solid rgba(255,255,255,0.5); }
  .dot-l { display:inline-block; width:13px; height:13px; border-radius:50%; background:#eec078; border:1px solid rgba(255,255,255,0.5); }

  .hud-cell {
    position: absolute;
    bottom: 54px;
    right: 8px;
    max-width: 280px;
    padding: 7px 10px;
    border-radius: 9px;
    background: rgba(21, 24, 30, 0.72);
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.12);
    font-size: 11px;
    line-height: 1.45;
    color: var(--text-soft);
    display: none;
  }
  .hud-cell.show { display: block; }

  .hud-bottom {
    position: absolute;
    bottom: 8px;
    left: 8px;
    right: 8px;
    padding: 6px 10px;
    border-radius: 9px;
    background: rgba(21, 24, 30, 0.68);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.10);
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .timeline-wrap { flex: 1; display: flex; flex-direction: column; gap: 2px; }
  .timeline-bar {
    height: 4px;
    border-radius: 3px;
    background: rgba(255,255,255,0.09);
    position: relative;
    overflow: hidden;
  }
  .timeline-model {
    position: absolute;
    top: 0;
    bottom: 0;
    background: rgba(125, 211, 189, 0.45);
    border-radius: 3px;
  }
  input[type="range"] { width: 100%; accent-color: var(--accent-ice); cursor: pointer; }
  .legend-grad {
    width: 90px;
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, #64829c, #a7c4d9, #e6ebf2, #e5b776, #d98282);
  }

  .table-wrap { overflow: auto; flex: 1; min-height: 0; }
  table { width: 100%; border-collapse: collapse; font-size: 11.5px; font-family: var(--mono); }
  th, td { border-bottom: 1px solid rgba(255,255,255,0.07); padding: 4px 7px; text-align: left; white-space: nowrap; }
  th {
    color: var(--accent-ice);
    font-family: var(--sans);
    font-weight: 600;
    font-size: 11px;
    position: sticky;
    top: 0;
    background: rgba(26, 30, 38, 0.92);
    backdrop-filter: blur(8px);
  }
  tr:hover td { background: rgba(255,255,255,0.03); }

  pre {
    background: rgba(14, 17, 22, 0.55);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 9px;
    padding: 12px;
    font-family: var(--mono);
    font-size: 11.5px;
    overflow: auto;
    line-height: 1.5;
    color: var(--text-soft);
    flex: 1;
  }
</style>
</head>
<body>

  <!-- UNIFIED LIQUID GLASS RIBBON HEADER -->
  <header class="ribbon-shell glass">
    <div class="ribbon-tabs">
      <div class="brand">
        <span>DEM-MCM Studio</span>
        <span class="brand-pill">Liquid Glass · 3D · plan egui</span>
      </div>
      <button class="rtab active" data-tab="mesh3d">Mélangeur 3D</button>
      <button class="rtab" data-tab="teneur">Teneur &amp; Particules</button>
      <button class="rtab" data-tab="errors">Erreurs &amp; Chaînes</button>
      <button class="rtab" data-tab="benchmark">Comparateur &amp; Benchmark</button>
      <button class="rtab" data-tab="plan">Plan egui</button>
      <div class="spacer"></div>
      <span class="status-badge" id="loader">Calcul Rust en cours...</span>
      <select id="dataset" title="Source DEM">
        <option value="synthetic_drum">Tambour synthétique (0..359 · 1030 part.)</option>
        <option value="real_chunks">DEM réelle data/chunks (0..5999 · 1030 part.)</option>
      </select>
      <button class="btn primary" id="run-btn">Exécuter</button>
    </div>

    <!-- CONTEXTUAL RIBBON CONTROLS (grouped horizontally without card chrome) -->
    <div class="ribbon-body">
      <div class="rgroup">
        <div class="rgroup-row">
          <div class="rfield">
            <label>Méthode de découpage (14)</label>
            <select id="method"></select>
          </div>
          <div id="method-params" class="rgroup-row"></div>
        </div>
        <div class="rgroup-label" id="method-family-lbl">Approche de découpage</div>
      </div>

      <div class="rgroup">
        <div class="rgroup-row">
          <div class="rfield"><label>NLT</label><input id="nlt" type="number" min="1" max="12" value="3"></div>
          <div class="rfield"><label>step</label><input id="step" type="number" min="1" max="500" value="30"></div>
          <div class="rfield"><label>dt</label><input id="dt" type="number" min="1" max="50" value="1"></div>
          <div class="rfield"><label>tau</label><input id="tau" type="number" min="1" max="500" value="12"></div>
          <div class="rfield"><label>start_modèle</label><input id="start_index" type="number" min="0" max="6000" value="40"></div>
          <div class="rfield">
            <label>Critère teneur</label>
            <select id="target-species">
              <option value="small">Teneur petites (4 mm)</option>
              <option value="large">Teneur grosses (8 mm)</option>
            </select>
          </div>
          <div class="rgroup-row" style="flex-direction:column;align-items:flex-start;gap:3px;">
            <label class="pill-chk"><input type="checkbox" id="full-duration" checked> Prédire toute la durée DEM</label>
            <label class="pill-chk"><input type="checkbox" id="no-species"> Sans distinction d'espèces</label>
          </div>
        </div>
        <div class="rgroup-label">Construction modèles &amp; prédiction DEM complète</div>
      </div>

      <div class="rgroup">
        <div class="rgroup-row">
          <div class="rfield">
            <label>Coloration 3D</label>
            <select id="color-mode">
              <option value="dem_teneur">Teneur DEM T_i(t)</option>
              <option value="hom_teneur">Teneur Homogène</option>
              <option value="inhom_teneur">Teneur Inhomogène</option>
              <option value="hom_err">|Teneur Hom − DEM|</option>
              <option value="inhom_err">|Teneur Inhom − DEM|</option>
              <option value="cell_rmse">Discrepancy RMSE cellule</option>
              <option value="population">Nombre de particules N_i</option>
              <option value="cell_id">Zone / ID cellule</option>
            </select>
          </div>
          <div class="rgroup-row" style="flex-wrap:wrap;max-width:340px;gap:4px;">
            <label class="pill-chk"><input type="checkbox" id="show-small" checked> Petites 4mm</label>
            <label class="pill-chk"><input type="checkbox" id="show-large" checked> Grosses 8mm</label>
            <label class="pill-chk"><input type="checkbox" id="show-wire" checked> Maillage</label>
            <label class="pill-chk"><input type="checkbox" id="show-faces" checked> Faces</label>
            <label class="pill-chk"><input type="checkbox" id="show-vel"> Vitesses</label>
            <label class="pill-chk"><input type="checkbox" id="show-edges" checked> Flux P_ij</label>
            <label class="pill-chk"><input type="checkbox" id="show-drum" checked> Tambour</label>
          </div>
          <div class="rgroup-row" style="flex-direction:column;gap:3px;">
            <div style="display:flex;gap:3px;">
              <button class="btn sm" onclick="setCamPreset('iso')">Iso</button>
              <button class="btn sm" onclick="setCamPreset('xy')">XY</button>
            </div>
            <div style="display:flex;gap:3px;">
              <button class="btn sm" onclick="setCamPreset('xz')">XZ</button>
              <button class="btn sm" onclick="setCamPreset('yz')">YZ</button>
            </div>
          </div>
        </div>
        <div class="rgroup-label">Mélangeur 3D &amp; deux tailles</div>
      </div>
    </div>
  </header>

  <!-- MINIMAL LIQUID GLASS TELEMETRY STRIP -->
  <div class="telemetry" id="telemetry"></div>

  <!-- MAIN WORKSPACE -->
  <main class="workspace">

    <!-- PANE 1: 3D MIXER + FULL-DURATION CURVES -->
    <section class="pane active" id="pane-mesh3d">
      <div class="surface glass" style="flex:1.55;">
        <div class="viewport-wrap">
          <canvas id="canvas3d"></canvas>
          <div class="hud-tl" id="hud-info">Chargement...</div>
          <div class="hud-tr">
            <span><span class="dot-s"></span> 4 mm (<span id="cnt-small">-</span>)</span>
            <span><span class="dot-l"></span> 8 mm (<span id="cnt-large">-</span>)</span>
            <span>Total : <b id="cnt-total" style="color:var(--text);font-family:var(--mono);">-</b></span>
          </div>
          <div class="hud-cell" id="cell-info"></div>
          <div class="hud-bottom">
            <button class="btn sm" id="play-btn">Lecture</button>
            <span style="font-family:var(--mono);font-size:11px;min-width:68px;" id="time-label">t = 0</span>
            <div class="timeline-wrap">
              <div class="timeline-bar"><div class="timeline-model" id="model-window-bar"></div></div>
              <input type="range" id="time-slider" min="0" max="0" value="0">
            </div>
            <div style="display:flex;align-items:center;gap:5px;font-size:10px;color:var(--muted);">
              <span>0</span><div class="legend-grad"></div><span>1</span>
            </div>
          </div>
        </div>
      </div>

      <div class="col" style="flex:1;">
        <div class="surface glass" style="flex:1.15;">
          <div class="subtle-bar">
            <span>Prédiction complète DEM (bande verte = construction des modèles)</span>
            <select id="overview-metric">
              <option value="rsd">RSD de teneur</option>
              <option value="l1">Erreur L1 normalisée</option>
              <option value="rms_disc">Discrepancy RMS</option>
              <option value="n_part">Nombre de particules</option>
              <option value="entropy">Entropie H(t)</option>
              <option value="danckwerts">Danckwerts I_D(t)</option>
            </select>
          </div>
          <div style="flex:1;min-height:170px;"><canvas id="chart-overview"></canvas></div>
        </div>
        <div class="surface glass" style="flex:0.85;">
          <div class="subtle-bar">
            <span>Synthèse Homogène vs Inhomogène (critère teneur &amp; erreurs)</span>
            <span id="horizon-badge" style="font-family:var(--mono);color:var(--accent-sage);"></span>
          </div>
          <div class="table-wrap" id="summary-table-wrap"></div>
        </div>
      </div>
    </section>

    <!-- PANE 2: TENEUR & PARTICLE COUNTS -->
    <section class="pane" id="pane-teneur">
      <div class="col" style="flex:1.2;">
        <div class="surface glass" style="flex:1.25;">
          <div class="subtle-bar">
            <span>Teneur en l'espèce cible par cellule (top cellules peuplées — DEM vs Homogène vs Inhomogène)</span>
            <select id="teneur-cell-select"></select>
          </div>
          <div style="flex:1;min-height:200px;"><canvas id="chart-teneur-cell"></canvas></div>
        </div>
        <div class="surface glass" style="flex:0.9;">
          <div class="subtle-bar">
            <span>Nombre de particules total &amp; par espèce (conservation DEM vs Markov)</span>
            <select id="npart-species">
              <option value="total">Total (petites + grosses)</option>
              <option value="small">Petites (4 mm)</option>
              <option value="large">Grosses (8 mm)</option>
            </select>
          </div>
          <div style="flex:1;min-height:160px;"><canvas id="chart-npart"></canvas></div>
        </div>
      </div>
      <div class="col" style="flex:1;">
        <div class="surface glass" style="flex:1;">
          <div class="subtle-bar"><span>Population par cellule (top-5 occupées) — DEM vs Markov</span></div>
          <div style="flex:1;min-height:170px;"><canvas id="chart-top-pop"></canvas></div>
        </div>
        <div class="surface glass" style="flex:0.9;">
          <div class="subtle-bar"><span>Erreurs sur la teneur, temps de mélange t50/t90 et conservation</span></div>
          <div class="table-wrap" id="teneur-table-wrap"></div>
        </div>
      </div>
    </section>

    <!-- PANE 3: ERRORS & MARKOV CHAINS (postprocess.py & postprocess_inhomogeneous.py) -->
    <section class="pane" id="pane-errors">
      <div class="col" style="flex:1.1;">
        <div class="surface glass" style="flex:1;">
          <div class="subtle-bar">
            <span>Erreur L1 normalisée par le nombre de particules : Σ|p_DEM − p_MC| (Hom vs Inhom)</span>
            <select id="l1-scope">
              <option value="target">Espèce cible</option>
              <option value="total">Toutes espèces</option>
            </select>
          </div>
          <div style="flex:1;min-height:170px;"><canvas id="chart-l1"></canvas></div>
        </div>
        <div class="surface glass" style="flex:1;">
          <div class="subtle-bar">
            <span>Discrepancy heatmap Temps × Cellules (|p_MK − p_DEM|)</span>
            <select id="heat-disc-chain">
              <option value="inhom">Chaîne Inhomogène</option>
              <option value="hom">Chaîne Homogène</option>
            </select>
          </div>
          <div style="flex:1;min-height:170px;"><canvas id="heat-discrepancy"></canvas></div>
        </div>
      </div>

      <div class="col" style="flex:1;">
        <div class="surface glass" style="flex:1;">
          <div class="subtle-bar">
            <span>Matrices de transition : P Homogène vs Bloc P^(b) vs ΔP_k = P_{k+1} − P_k</span>
            <div style="display:flex;gap:5px;">
              <select id="mat-view">
                <option value="hom">P Homogène</option>
                <option value="block">Bloc P^(b)</option>
                <option value="diff">Différence ΔP_k</option>
              </select>
              <select id="block-select"></select>
              <span id="hover-p-lbl" style="font-family:var(--mono);color:var(--accent-ice);"></span>
            </div>
          </div>
          <div style="flex:1;min-height:200px;"><canvas id="heat-matrix"></canvas></div>
        </div>
        <div class="surface glass" style="flex:0.95;">
          <div class="subtle-bar">
            <span>Évolution p_{i→j}(t) sur les blocs NLT &amp; lois polynomiales (postprocess_inhomogeneous)</span>
          </div>
          <div style="flex:1;min-height:150px;"><canvas id="chart-plaws"></canvas></div>
        </div>
      </div>
    </section>

    <!-- PANE 4: DUAL 3D & MULTI-METHOD BENCHMARK -->
    <section class="pane" id="pane-benchmark">
      <div class="col" style="flex:1;">
        <div class="surface glass" style="flex:1;">
          <div class="subtle-bar">
            <span>Comparaison 3D synchronisée — Maillage A (<b id="dual-a-lbl"></b>) vs Maillage B :</span>
            <div style="display:flex;gap:6px;align-items:center;">
              <select id="method-b"></select>
              <button class="btn sm primary" id="run-dual-btn">Comparer 3D A/B</button>
            </div>
          </div>
          <div style="flex:1;display:flex;gap:8px;min-height:0;">
            <div class="viewport-wrap" style="flex:1;"><canvas id="canvas3d-a"></canvas></div>
            <div class="viewport-wrap" style="flex:1;"><canvas id="canvas3d-b"></canvas></div>
          </div>
        </div>
        <div class="surface glass" style="flex:0.85;">
          <div class="subtle-bar"><span>Comparaison temporelle A vs B (RSD teneur &amp; Erreur L1)</span></div>
          <div style="flex:1;min-height:160px;"><canvas id="chart-dual"></canvas></div>
        </div>
      </div>

      <div class="col" style="flex:1.1;">
        <div class="surface glass" style="flex:1;">
          <div class="subtle-bar">
            <span>Benchmark multi-méthodes (métriques, erreurs, particules)</span>
            <button class="btn sm primary" id="run-bench-btn">Lancer Benchmark (7 approches)</button>
          </div>
          <div style="flex:0.8;min-height:160px;margin-bottom:6px;"><canvas id="chart-bench"></canvas></div>
          <div class="table-wrap" id="bench-table-wrap"></div>
        </div>
      </div>
    </section>

    <!-- PANE 5: EGUI ARCHITECTURE PLAN -->
    <section class="pane" id="pane-plan">
      <div class="surface glass" style="flex:1;">
        <div class="subtle-bar"><span>Planification d'architecture Rust pour egui + 3D + Liquid Glass</span></div>
        <pre id="egui-plan-text"></pre>
      </div>
    </section>
  </main>

<script>
const APP = {
  staticMode: false,
  methods: [],
  bundleA: null,
  bundleB: null,
  benchRows: null,
  frameIdx: 0,
  playing: false,
  playTimer: null,
  selectedCell: null,
  cam: { yaw: 0.68, pitch: 0.44, distance: 0.34, target: [0, -0.01, 0], fov: 42 },
  drag: { active: false, mode: 'orbit', x: 0, y: 0 },
  projectedCells: [],
};

const PALETTE = [
  '#8faec9', '#7dd3bd', '#eec078', '#e5989b', '#b8b8e6', '#d8a48f',
  '#88c9c2', '#a5bcd6', '#cfa8c4', '#b4cc8e', '#c9a6e0', '#e8cf7a'
];

function hexToRgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n>>16)&255},${(n>>8)&255},${n&255},${a})`;
}

function teneurColor(val, alpha, isErr = false) {
  const t = Math.max(0, Math.min(1, val));
  if (isErr) {
    const r = Math.round(90 + 145 * Math.pow(t, 0.72));
    const g = Math.round(200 * (1 - t * 0.55));
    const b = Math.round(170 * (1 - t * 0.75));
    return `rgba(${r},${g},${b},${alpha})`;
  }
  // Soft desaturated cool-slate -> pearl -> warm-coral scale
  let r, g, b;
  if (t < 0.5) {
    const k = t * 2;
    r = Math.round(92 + (222 - 92) * k);
    g = Math.round(126 + (228 - 126) * k);
    b = Math.round(156 + (236 - 156) * k);
  } else {
    const k = (t - 0.5) * 2;
    r = Math.round(222 + (218 - 222) * k);
    g = Math.round(228 + (132 - 228) * k);
    b = Math.round(236 + (132 - 236) * k);
  }
  return `rgba(${r},${g},${b},${alpha})`;
}

function renderMethodParams(method) {
  const box = document.getElementById('method-params');
  const m = APP.methods.find(x => x.id === method);
  if (m) {
    document.getElementById('method-family-lbl').textContent =
      m.family + (m.uses_velocity ? ' · cinématique' : '') + ' — ' + m.summary;
  }
  const fields = {
    cartesian: [['nx', 'nx', 3, 1, 10], ['ny', 'ny', 3, 1, 10], ['nz', 'nz', 2, 1, 8]],
    quantile: [['nx', 'nx', 3, 1, 10], ['ny', 'ny', 3, 1, 10], ['nz', 'nz', 2, 1, 8]],
    cylindrical: [['nr', 'nr', 3, 1, 8], ['ntheta', 'nθ', 6, 2, 24], ['nz', 'nz', 1, 1, 6]],
    octree: [['max_particles', 'max_pts', 120, 15, 2000], ['max_depth', 'prof.', 3, 1, 5]],
    voronoi: [['n_cells', 'n_cells', 16, 2, 64]],
    physics: [['n_cells', 'n_cells', 16, 2, 64], ['velocity_weight', 'w_V', 0.5, 0, 5, 0.1]],
    physics_full_vel: [['n_cells', 'n_cells', 16, 2, 64], ['velocity_weight', 'w_V', 0.5, 0, 5, 0.1]],
    spectral: [['n_cells', 'n_cells', 16, 2, 40], ['velocity_weight', 'w_V', 0.5, 0, 5, 0.1]],
    gmm: [['n_cells', 'n_cells', 14, 2, 40], ['velocity_weight', 'w_V', 0.5, 0, 5, 0.1]],
    dbscan: [['eps', 'eps(z)', 0.45, 0.05, 1.5, 0.02], ['min_samples', 'min_pts', 4, 2, 50]],
    spectral_biclustering: [['n_cells', 'n_row', 12, 2, 32], ['n_col_clusters', 'n_col', 2, 2, 6]],
    adaptive: [['y_split', 'y_split', 0.55, 0.2, 0.85, 0.05]],
  }[method] || [];

  let html = fields.map(([k, lbl, val, mn, mx, st]) => `
    <div class="rfield">
      <label>${lbl}</label>
      <input type="number" data-kw="${k}" value="${val}" min="${mn}" max="${mx}" step="${st||1}">
    </div>
  `).join('');

  if (method === 'cylindrical') {
    html += `<div class="rfield"><label>Mode</label>
      <select data-kw="radial_mode"><option value="equal_area">equal_area</option><option value="equal_dr">equal_dr</option></select></div>`;
  }
  if (method === 'octree') {
    html += `<div class="rfield"><label>Coupes</label>
      <select data-kw="oblique_method">
        <option value="pca">pca</option>
        <option value="axis">axis</option>
        <option value="kmeans2">kmeans2</option>
        <option value="2medians">2medians</option>
        <option value="svm">svm</option>
        <option value="random">random</option>
      </select></div>`;
  }
  box.innerHTML = html;
}

function gatherKwargs(method) {
  const kw = {};
  document.querySelectorAll('#method-params [data-kw]').forEach(el => {
    const k = el.dataset.kw;
    kw[k] = el.tagName === 'SELECT' ? el.value : parseFloat(el.value);
  });
  if (method === 'adaptive') {
    kw.y_split_mode = 'quantile';
    kw.top_method = 'single';
    kw.bottom_method = 'cylindrical';
    kw.bottom_kwargs = { nr: 2, ntheta: 6, nz: 1 };
  } else if (method === 'multizone') {
    kw.y_mode = 'quantile';
    kw.zones = [
      { method: 'cartesian', y_min: 0.0, y_max: 0.5, kwargs: { nx: 2, ny: 1, nz: 2 } },
      { method: 'voronoi', y_min: 0.5, y_max: 1.0, kwargs: { n_cells: 6 } }
    ];
  }
  return kw;
}

function defaultKwargs(method) {
  return {
    cartesian: { nx: 3, ny: 3, nz: 2 },
    quantile: { nx: 3, ny: 3, nz: 2 },
    cylindrical: { nr: 3, ntheta: 6, nz: 1, radial_mode: 'equal_area' },
    octree: { max_particles: 120, max_depth: 3, oblique_method: 'pca' },
    voronoi: { n_cells: 16 },
    physics: { n_cells: 16, velocity_weight: 0.5 },
    physics_full_vel: { n_cells: 16, velocity_weight: 0.5 },
    spectral: { n_cells: 16, velocity_weight: 0.5 },
    gmm: { n_cells: 14, velocity_weight: 0.5 },
    dbscan: { eps: 0.45, min_samples: 4, velocity_weight: 0.6 },
    spectral_biclustering: { n_cells: 12, n_col_clusters: 2, velocity_weight: 0.5 },
    adaptive: { y_split: 0.55, y_split_mode: 'quantile', top_method: 'single', bottom_method: 'cylindrical', bottom_kwargs: { nr: 2, ntheta: 6, nz: 1 } },
    multizone: { y_mode: 'quantile', zones: [{ method: 'cartesian', y_min: 0, y_max: 0.5, kwargs: { nx: 2, ny: 1, nz: 2 } }, { method: 'voronoi', y_min: 0.5, y_max: 1, kwargs: { n_cells: 6 } }] },
    single: {}
  }[method] || {};
}

function currentRequest(method, kwargs) {
  return {
    dataset: document.getElementById('dataset').value,
    method,
    kwargs,
    nlt: parseInt(document.getElementById('nlt').value) || 3,
    step: parseInt(document.getElementById('step').value) || 30,
    dt: parseInt(document.getElementById('dt').value) || 1,
    tau: parseInt(document.getElementById('tau').value) || 12,
    start_index: parseInt(document.getElementById('start_index').value) || 0,
    no_species: document.getElementById('no-species').checked,
    target_species: document.getElementById('target-species').value,
    predict_full_duration: document.getElementById('full-duration').checked,
  };
}

async function fetchSimulate(method, kwargs) {
  const payload = currentRequest(method, kwargs);
  if (!APP.staticMode) {
    try {
      const resp = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data.error) throw new Error(data.error);
        return data;
      }
      if (resp.status !== 404 && resp.status !== 405) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || 'Erreur Rust');
      }
    } catch (e) {
      if (!String(e).includes('Failed to fetch') && !String(e).includes('NetworkError')) {
        throw e;
      }
    }
    APP.staticMode = true;
  }
  const ds = payload.dataset;
  const sp = payload.target_species || 'small';
  const candidates = [
    `static_data/sim_${ds}_${method}_${sp}.json`,
    `static_data/sim_synthetic_drum_${method}_${sp}.json`,
    `static_data/sim_synthetic_drum_cartesian_${sp}.json`,
  ];
  for (const url of candidates) {
    const r = await fetch(url);
    if (r.ok) return await r.json();
  }
  throw new Error('Donnée statique indisponible');
}

async function runPrimary() {
  const loader = document.getElementById('loader');
  loader.classList.add('show');
  try {
    const method = document.getElementById('method').value;
    const bundle = await fetchSimulate(method, gatherKwargs(method));
    APP.bundleA = bundle;
    APP.frameIdx = 0;
    APP.selectedCell = null;
    fitCameraToMesh(bundle.mesh);
    const slider = document.getElementById('time-slider');
    slider.max = Math.max(0, bundle.frames.length - 1);
    slider.value = 0;
    populateSelectors(bundle);
    renderAllViews();
  } catch (e) {
    alert('Erreur Rust: ' + e.message);
  } finally {
    loader.classList.remove('show');
  }
}

async function runDual() {
  const loader = document.getElementById('loader');
  loader.classList.add('show');
  try {
    const mA = document.getElementById('method').value;
    const mB = document.getElementById('method-b').value;
    const [bA, bB] = await Promise.all([
      fetchSimulate(mA, gatherKwargs(mA)),
      fetchSimulate(mB, defaultKwargs(mB)),
    ]);
    APP.bundleA = bA;
    APP.bundleB = bB;
    renderAllViews();
  } catch (e) {
    alert('Erreur Dual-Mesh: ' + e.message);
  } finally {
    loader.classList.remove('show');
  }
}

async function runBenchmark() {
  const loader = document.getElementById('loader');
  loader.classList.add('show');
  try {
    const method = document.getElementById('method').value;
    const payload = currentRequest(method, gatherKwargs(method));
    payload.methods = ['cartesian', 'cylindrical', 'quantile', 'octree', 'voronoi', 'physics', 'adaptive'];
    let data = null;
    if (!APP.staticMode) {
      const resp = await fetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }).catch(() => null);
      if (resp && resp.ok) {
        data = await resp.json();
        if (data.error) throw new Error(data.error);
      }
    }
    if (!data) {
      const sp = payload.target_species || 'small';
      const ds = payload.dataset || 'synthetic_drum';
      const r1 = await fetch(`static_data/benchmark_${ds}_${sp}.json`);
      const r = r1.ok ? r1 : await fetch(`static_data/benchmark_synthetic_drum_${sp}.json`);
      if (!r.ok) throw new Error('Benchmark statique indisponible');
      data = await r.json();
    }
    APP.benchRows = data.rows;
    renderBenchmark();
  } catch (e) {
    alert('Erreur Benchmark: ' + e.message);
  } finally {
    loader.classList.remove('show');
  }
}

function fitCameraToMesh(mesh) {
  const b = mesh.domain_bounds;
  const diag = Math.max(0.05, Math.hypot(b[1]-b[0], b[3]-b[2], b[5]-b[4]));
  APP.cam.target = [0.5*(b[0]+b[1]), 0.5*(b[2]+b[3]), 0.5*(b[4]+b[5])];
  APP.cam.distance = diag * 1.38;
}

function setCamPreset(p) {
  if (p === 'iso') { APP.cam.yaw = 0.68; APP.cam.pitch = 0.44; }
  if (p === 'xy')  { APP.cam.yaw = 0.0;  APP.cam.pitch = 0.0; }
  if (p === 'xz')  { APP.cam.yaw = 0.0;  APP.cam.pitch = 1.54; }
  if (p === 'yz')  { APP.cam.yaw = Math.PI/2; APP.cam.pitch = 0.0; }
  render3DAll();
}

function camBasis(cam) {
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const fwd = [-cp * sy, -sp, -cp * cy];
  const rx = -fwd[2], rz = fwd[0];
  const rl = Math.hypot(rx, rz) || 1e-12;
  const right = [rx / rl, 0, rz / rl];
  const up = [
    right[1]*fwd[2] - right[2]*fwd[1],
    right[2]*fwd[0] - right[0]*fwd[2],
    right[0]*fwd[1] - right[1]*fwd[0],
  ];
  const eye = [
    cam.target[0] - fwd[0]*cam.distance,
    cam.target[1] - fwd[1]*cam.distance,
    cam.target[2] - fwd[2]*cam.distance,
  ];
  return { right, up, fwd, eye };
}

function projectPt(pt, basis, w, h, fov) {
  const rx = pt[0] - basis.eye[0], ry = pt[1] - basis.eye[1], rz = pt[2] - basis.eye[2];
  const cx = rx*basis.right[0] + ry*basis.right[1] + rz*basis.right[2];
  const cy = rx*basis.up[0] + ry*basis.up[1] + rz*basis.up[2];
  const depth = rx*basis.fwd[0] + ry*basis.fwd[1] + rz*basis.fwd[2];
  if (depth <= 1e-4) return null;
  const focal = (0.5 * h) / Math.tan((fov * Math.PI / 180) * 0.5);
  const scale = focal / depth;
  return { x: 0.5 * w + cx * scale, y: 0.5 * h - cy * scale, depth, scale };
}

function nearestEvalRow(bundle, ts) {
  const times = bundle.hom_metrics.times_eval;
  if (!times.length) return 0;
  let best = 0, bestD = Infinity;
  for (let i = 0; i < times.length; i++) {
    const d = Math.abs(times[i] - ts);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function getCellMetricValue(bundle, cid, frame, colorMode) {
  const evalIdx = nearestEvalRow(bundle, frame.timestep);
  const cDem = (bundle.dem_conc_traj[evalIdx] && bundle.dem_conc_traj[evalIdx][cid]) ?? (frame.cell_conc_dem[cid] ?? 0.5);
  const cHom = (bundle.hom_conc_traj[evalIdx] && bundle.hom_conc_traj[evalIdx][cid]) ?? 0.5;
  const cInh = (bundle.inhom_conc_traj[evalIdx] && bundle.inhom_conc_traj[evalIdx][cid]) ?? 0.5;
  if (colorMode === 'dem_teneur') return { v: cDem, err: false };
  if (colorMode === 'hom_teneur') return { v: cHom, err: false };
  if (colorMode === 'inhom_teneur') return { v: cInh, err: false };
  if (colorMode === 'hom_err') return { v: Math.min(1, Math.abs(cHom - cDem) * 2.5), err: true };
  if (colorMode === 'inhom_err') return { v: Math.min(1, Math.abs(cInh - cDem) * 2.5), err: true };
  if (colorMode === 'cell_rmse') {
    const cr = bundle.inhom_metrics.cell_rmse_per_cell?.[cid] ?? 0;
    return { v: Math.min(1, cr * 8), err: true };
  }
  if (colorMode === 'population') {
    const maxPop = Math.max(1, ...frame.cell_pop);
    return { v: (frame.cell_pop[cid] || 0) / maxPop, err: false };
  }
  return { v: cid, err: false };
}

function renderViewport3D(canvas, bundle, recordHits = false) {
  if (!canvas || !bundle) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const basis = camBasis(APP.cam);
  const proj = p => projectPt(p, basis, W, H, APP.cam.fov);
  const frame = bundle.frames[Math.min(APP.frameIdx, bundle.frames.length - 1)] || bundle.frames[0];
  const colorMode = document.getElementById('color-mode').value;
  const showWire = document.getElementById('show-wire').checked;
  const showFaces = document.getElementById('show-faces').checked;
  const showSmall = document.getElementById('show-small').checked;
  const showLarge = document.getElementById('show-large').checked;
  const showVel = document.getElementById('show-vel').checked;
  const showEdges = document.getElementById('show-edges').checked;
  const showDrum = document.getElementById('show-drum').checked;

  // Drum wireframe
  if (showDrum) {
    const dr = bundle.mesh.drum;
    const N = 40;
    ctx.strokeStyle = 'rgba(207, 224, 240, 0.22)';
    ctx.lineWidth = 1;
    [dr.z_min, dr.z_max].forEach(z => {
      ctx.beginPath();
      for (let i = 0; i <= N; i++) {
        const a = (i / N) * Math.PI * 2;
        const p = proj([dr.radius * Math.cos(a), dr.radius * Math.sin(a), z]);
        if (!p) continue;
        if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    });
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      const p1 = proj([dr.radius * Math.cos(a), dr.radius * Math.sin(a), dr.z_min]);
      const p2 = proj([dr.radius * Math.cos(a), dr.radius * Math.sin(a), dr.z_max]);
      if (p1 && p2) { ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke(); }
    }
  }

  const items = [];
  if (recordHits) APP.projectedCells = [];

  for (const cell of bundle.mesh.cells) {
    const isSel = APP.selectedCell === cell.id;
    const pop = frame.cell_pop[cell.id] || 0;
    const metric = getCellMetricValue(bundle, cell.id, frame, colorMode);
    const baseHex = PALETTE[(cell.zone * 5 + cell.id) % PALETTE.length];
    const faceFill = colorMode === 'cell_id'
      ? hexToRgba(baseHex, isSel ? 0.38 : 0.11)
      : teneurColor(metric.v, isSel ? 0.44 : (pop > 0 ? 0.18 : 0.04), metric.err);

    const pc = proj(cell.centroid);
    if (recordHits && pc) {
      APP.projectedCells.push({ id: cell.id, x: pc.x, y: pc.y, pop, cell });
    }

    if (showFaces) {
      for (const poly of cell.faces) {
        const pts = [];
        let dSum = 0, ok = true;
        for (const v of poly) {
          const pv = proj(v);
          if (!pv) { ok = false; break; }
          pts.push(pv); dSum += pv.depth;
        }
        if (ok && pts.length >= 3) {
          items.push({ type: 'poly', depth: dSum / pts.length, pts, fill: faceFill, stroke: isSel ? '#ffffff' : null });
        }
      }
    }

    if (showWire) {
      const strokeCol = isSel
        ? '#ffffff'
        : (colorMode === 'cell_id' ? hexToRgba(baseHex, 0.58) : 'rgba(207, 224, 240, 0.30)');
      for (const [a, b] of cell.edges) {
        const pa = proj(a), pb = proj(b);
        if (pa && pb) {
          items.push({ type: 'line', depth: 0.5*(pa.depth+pb.depth) - 1e-4, pa, pb, col: strokeCol, lw: isSel ? 1.8 : 0.8 });
        }
      }
    }
  }

  for (const sp of (bundle.mesh.split_planes || [])) {
    const pts = sp.quad.map(proj).filter(Boolean);
    if (pts.length === 4) {
      const d = pts.reduce((a, p) => a + p.depth, 0) / 4;
      items.push({ type: 'poly', depth: d, pts, fill: 'rgba(125,211,189,0.14)', stroke: 'rgba(125,211,189,0.75)' });
    }
  }

  if (showEdges && bundle.top_edges_small) {
    const cenById = new Map(bundle.mesh.cells.map(c => [c.id, c.centroid]));
    for (const [si, dj, prob] of bundle.top_edges_small) {
      const cs = cenById.get(si), cd = cenById.get(dj);
      if (!cs || !cd) continue;
      const pa = proj(cs), pb = proj(cd);
      if (pa && pb) {
        items.push({
          type: 'arrow',
          depth: 0.5*(pa.depth+pb.depth) - 2e-4,
          pa, pb,
          col: `rgba(158, 192, 220, ${Math.min(0.85, 0.22 + prob * 1.3)})`,
          lw: 0.9 + prob * 4.0,
        });
      }
    }
  }

  // Render BOTH particle sizes faithfully at true 2:1 diameter ratio (`diameters[i] * 0.5 * scale`)
  if (frame && (showSmall || showLarge)) {
    const diams = frame.diameters || [];
    for (let i = 0; i < frame.coords.length; i++) {
      const isS = frame.is_small[i];
      if ((isS && !showSmall) || (!isS && !showLarge)) continue;
      const p = proj(frame.coords[i]);
      if (!p) continue;
      const d = diams[i] || (isS ? 0.004 : 0.008);
      // Exact physical radius scaled consistently (visual min 2.1px small / 4.2px large so 2:1 size always shows clearly)
      const rad = Math.max(isS ? 2.1 : 4.2, Math.min(isS ? 5.0 : 10.0, p.scale * d * 0.56));
      items.push({ type: 'sphere', depth: p.depth - 3e-4, x: p.x, y: p.y, r: rad, isSmall: isS });
      if (showVel && frame.velocities[i]) {
        const v = frame.velocities[i];
        const tip = proj([frame.coords[i][0] + v[0]*0.024, frame.coords[i][1] + v[1]*0.024, frame.coords[i][2] + v[2]*0.024]);
        if (tip) items.push({ type: 'line', depth: p.depth - 3.5e-4, pa: p, pb: tip, col: 'rgba(207,224,240,0.65)', lw: 1.0 });
      }
    }
  }

  items.sort((a, b) => b.depth - a.depth);
  for (const it of items) {
    if (it.type === 'poly') {
      ctx.beginPath();
      ctx.moveTo(it.pts[0].x, it.pts[0].y);
      for (let k = 1; k < it.pts.length; k++) ctx.lineTo(it.pts[k].x, it.pts[k].y);
      ctx.closePath();
      ctx.fillStyle = it.fill;
      ctx.fill();
      if (it.stroke) { ctx.strokeStyle = it.stroke; ctx.lineWidth = 1.3; ctx.stroke(); }
    } else if (it.type === 'line') {
      ctx.strokeStyle = it.col; ctx.lineWidth = it.lw;
      ctx.beginPath(); ctx.moveTo(it.pa.x, it.pa.y); ctx.lineTo(it.pb.x, it.pb.y); ctx.stroke();
    } else if (it.type === 'arrow') {
      ctx.strokeStyle = it.col; ctx.lineWidth = it.lw;
      ctx.beginPath(); ctx.moveTo(it.pa.x, it.pa.y); ctx.lineTo(it.pb.x, it.pb.y); ctx.stroke();
      const ang = Math.atan2(it.pb.y - it.pa.y, it.pb.x - it.pa.x);
      const hl = 5 + it.lw;
      ctx.beginPath();
      ctx.moveTo(it.pb.x, it.pb.y);
      ctx.lineTo(it.pb.x - hl*Math.cos(ang-0.42), it.pb.y - hl*Math.sin(ang-0.42));
      ctx.lineTo(it.pb.x - hl*Math.cos(ang+0.42), it.pb.y - hl*Math.sin(ang+0.42));
      ctx.closePath(); ctx.fillStyle = it.col; ctx.fill();
    } else if (it.type === 'sphere') {
      const g = ctx.createRadialGradient(it.x - it.r*0.3, it.y - it.r*0.3, it.r*0.12, it.x, it.y, it.r);
      if (it.isSmall) {
        g.addColorStop(0, '#dcfaf5');
        g.addColorStop(0.45, '#7dd3c8');
        g.addColorStop(1, '#2b6a64');
      } else {
        g.addColorStop(0, '#fff3db');
        g.addColorStop(0.45, '#eec078');
        g.addColorStop(1, '#7a5420');
      }
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(it.x, it.y, it.r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(10,12,16,0.55)'; ctx.lineWidth = 0.65; ctx.stroke();
    }
  }
}

function drawLineChart(canvas, xVals, series, blocks = [], modelRange = null) {
  if (!canvas || !xVals || !xVals.length) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const pad = { l: 46, r: 12, t: 22, b: 24 };
  const pw = Math.max(10, W - pad.l - pad.r), ph = Math.max(10, H - pad.t - pad.b);
  const xMin = xVals[0], xMax = xVals[xVals.length - 1] === xVals[0] ? xVals[0] + 1 : xVals[xVals.length - 1];
  let yMin = Infinity, yMax = -Infinity;
  for (const s of series) {
    for (const v of s.data) {
      if (v != null && isFinite(v)) { yMin = Math.min(yMin, v); yMax = Math.max(yMax, v); }
    }
  }
  if (!isFinite(yMin)) { yMin = 0; yMax = 1; }
  const ySpan = Math.max(1e-6, yMax - yMin);
  yMin = Math.max(0, yMin - 0.08 * ySpan);
  yMax = yMax + 0.1 * ySpan;

  const tx = x => pad.l + ((x - xMin) / (xMax - xMin)) * pw;
  const ty = y => pad.t + (1 - (y - yMin) / (yMax - yMin)) * ph;

  // Model construction phase band vs full DEM prediction
  if (modelRange) {
    const mx0 = Math.max(pad.l, tx(modelRange[0]));
    const mx1 = Math.min(W - pad.r, tx(modelRange[1]));
    if (mx1 > mx0) {
      ctx.fillStyle = 'rgba(125, 211, 189, 0.09)';
      ctx.fillRect(mx0, pad.t, mx1 - mx0, ph);
      ctx.strokeStyle = 'rgba(125, 211, 189, 0.28)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(mx0, pad.t); ctx.lineTo(mx0, pad.t + ph);
      ctx.moveTo(mx1, pad.t); ctx.lineTo(mx1, pad.t + ph);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  blocks.forEach((b, idx) => {
    const x0 = Math.max(pad.l, tx(b.t_start));
    const x1 = Math.min(W - pad.r, tx(b.t_end));
    if (x1 > x0) {
      ctx.fillStyle = idx % 2 === 0 ? 'rgba(207,224,240,0.05)' : 'rgba(125,211,189,0.06)';
      ctx.fillRect(x0, pad.t, x1 - x0, ph);
      ctx.fillStyle = 'rgba(182,192,206,0.58)';
      ctx.font = '9.5px monospace';
      ctx.fillText(`B${b.block_index}`, x0 + 3, pad.t + 11);
    }
  });

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.font = '10px monospace';
  ctx.fillStyle = '#8490a2';
  for (let i = 0; i <= 4; i++) {
    const yv = yMin + (i / 4) * (yMax - yMin);
    const yp = ty(yv);
    ctx.beginPath(); ctx.moveTo(pad.l, yp); ctx.lineTo(W - pad.r, yp); ctx.stroke();
    ctx.fillText(yv >= 100 ? yv.toFixed(0) : yv.toFixed(2), 4, yp + 3);
  }
  for (let i = 0; i <= 4; i++) {
    const xv = Math.round(xMin + (i / 4) * (xMax - xMin));
    const xp = tx(xv);
    ctx.beginPath(); ctx.moveTo(xp, pad.t); ctx.lineTo(xp, H - pad.b); ctx.stroke();
    ctx.fillText(String(xv), xp - 10, H - 7);
  }

  for (const s of series) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.lw || 1.9;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < xVals.length; i++) {
      const y = s.data[i];
      if (y == null || !isFinite(y)) continue;
      const px = tx(xVals[i]), py = ty(y);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
  }

  let lx = pad.l + 4;
  series.forEach(s => {
    ctx.fillStyle = s.color;
    ctx.fillRect(lx, 6, 12, 3.5);
    ctx.fillStyle = '#dce4ee';
    ctx.font = '10.5px sans-serif';
    ctx.fillText(s.label, lx + 15, 11);
    lx += ctx.measureText(s.label).width + 26;
  });
}

function drawMatrixHeatmap(canvas, matrix, isDiff = false) {
  if (!canvas || !matrix || !matrix.length) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const n = matrix.length;
  const size = Math.min(rect.width - 30, rect.height - 24);
  const ox = Math.round((rect.width - size) * 0.5), oy = 10;
  const cw = size / n;

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = matrix[r][c];
      if (v == null || !isFinite(v)) {
        ctx.fillStyle = 'rgba(15,18,24,0.65)';
      } else if (isDiff) {
        const k = Math.max(-1, Math.min(1, v * 3));
        if (k >= 0) ctx.fillStyle = `rgb(${Math.round(45+190*k)},${Math.round(52+130*k)},${Math.round(64+90*k)})`;
        else ctx.fillStyle = `rgb(${Math.round(45+165*(-k))},${Math.round(52+50*(-k))},${Math.round(64+60*(-k))})`;
      } else {
        const k = Math.min(1, Math.max(0, Math.pow(v, 0.55)));
        const R = Math.round(28 + 205 * k);
        const G = Math.round(38 + 185 * Math.sin(k * Math.PI * 0.78));
        const B = Math.round(56 + 150 * (1 - k * 0.45));
        ctx.fillStyle = `rgb(${R},${G},${B})`;
      }
      ctx.fillRect(ox + c * cw, oy + r * cw, Math.max(1, cw - 0.5), Math.max(1, cw - 0.5));
    }
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.16)';
  ctx.strokeRect(ox, oy, size, size);
  canvas._heatMeta = { ox, oy, size, n, matrix };
}

function drawRectHeatmap(canvas, grid, xTimes, cellIds) {
  if (!canvas || !grid || !grid.length || !grid[0].length) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 38, r: 10, t: 8, b: 22 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const rows = grid.length, cols = grid[0].length;
  let vmax = 1e-6;
  for (const r of grid) for (const v of r) if (v > vmax) vmax = v;
  const cw = pw / cols, ch = ph / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const k = Math.min(1, grid[r][c] / vmax);
      const R = Math.round(32 + 200 * Math.pow(k, 0.7));
      const G = Math.round(45 + 145 * Math.sin(k * Math.PI));
      const B = Math.round(62 + 40 * (1 - k));
      ctx.fillStyle = `rgb(${R},${G},${B})`;
      ctx.fillRect(pad.l + c * cw, pad.t + (rows - 1 - r) * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
  ctx.fillStyle = '#8490a2';
  ctx.font = '9.5px monospace';
  ctx.fillText(`C${cellIds[rows-1] ?? 0}`, 4, pad.t + 10);
  ctx.fillText(`C${cellIds[0] ?? 0}`, 4, H - pad.b - 2);
  ctx.fillText(String(xTimes[0]), pad.l, H - 6);
  ctx.fillText(String(xTimes[cols-1]), W - pad.r - 28, H - 6);
}

function drawBarBenchmark(canvas, rows) {
  if (!canvas || !rows || !rows.length) return;
  const rect = canvas.parentElement.getBoundingClientRect();
  if (rect.width < 10 || rect.height < 10) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 42, r: 12, t: 22, b: 30 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  let ymax = 1e-4;
  rows.forEach(r => { ymax = Math.max(ymax, r.hom_rmse_teneur, r.inhom_rmse_teneur); });
  ymax *= 1.15;
  const gw = pw / rows.length, bw = Math.max(6, gw * 0.34);

  ctx.font = '10px monospace';
  ctx.fillStyle = '#8490a2';
  for (let i = 0; i <= 3; i++) {
    const yv = (i / 3) * ymax;
    const yp = pad.t + (1 - i / 3) * ph;
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.beginPath(); ctx.moveTo(pad.l, yp); ctx.lineTo(W - pad.r, yp); ctx.stroke();
    ctx.fillText(yv.toFixed(3), 4, yp + 3);
  }
  rows.forEach((r, idx) => {
    const x0 = pad.l + idx * gw + gw * 0.14;
    const hH = (r.hom_rmse_teneur / ymax) * ph;
    const hI = (r.inhom_rmse_teneur / ymax) * ph;
    ctx.fillStyle = 'rgba(158, 192, 220, 0.78)';
    ctx.fillRect(x0, pad.t + ph - hH, bw, hH);
    ctx.fillStyle = 'rgba(125, 211, 189, 0.85)';
    ctx.fillRect(x0 + bw + 3, pad.t + ph - hI, bw, hI);
    ctx.fillStyle = '#dce4ee';
    ctx.fillText(r.method.slice(0, 8), x0, H - 10);
  });
  ctx.fillStyle = 'rgba(158, 192, 220, 0.85)'; ctx.fillRect(pad.l, 6, 11, 4);
  ctx.fillStyle = '#dce4ee'; ctx.font = '10.5px sans-serif'; ctx.fillText('RMSE(Teneur) Homogène', pad.l + 15, 11);
  ctx.fillStyle = 'rgba(125, 211, 189, 0.90)'; ctx.fillRect(pad.l + 165, 6, 11, 4);
  ctx.fillStyle = '#dce4ee'; ctx.fillText('RMSE(Teneur) Inhomogène', pad.l + 180, 11);
}

function populateSelectors(b) {
  const bSel = document.getElementById('block-select');
  bSel.innerHTML = b.blocks.map(bl =>
    `<option value="${bl.block_index}">B${bl.block_index} [${bl.t_start}..${bl.t_end}]</option>`
  ).join('');
  const cSel = document.getElementById('teneur-cell-select');
  cSel.innerHTML = `<option value="all">Top-${Math.min(5, b.top_cells.length)} cellules (superposées)</option>` +
    b.top_cells.map(c => `<option value="${c}">Cellule #${c}</option>`).join('');
  const span = Math.max(1, b.dem_t_max - b.dem_t_min);
  const left = ((b.model_t_start - b.dem_t_min) / span) * 100;
  const width = ((b.model_t_end - b.model_t_start) / span) * 100;
  const bar = document.getElementById('model-window-bar');
  bar.style.left = `${Math.max(0, Math.min(100, left))}%`;
  bar.style.width = `${Math.max(2, Math.min(100 - left, width))}%`;
}

function renderTelemetry(b) {
  const hm = b.hom_metrics, im = b.inhom_metrics;
  const fr = b.frames[APP.frameIdx] || b.frames[0];
  document.getElementById('telemetry').innerHTML = `
    <div class="tpill"><span class="k">Maillage</span><span class="v soft">${b.method} (${b.diagnostics.n_visited}/${b.mesh.n_cells})</span></div>
    <div class="tpill"><span class="k">Particules DEM</span><span class="v">${fr.n_small_total} (4mm) + ${fr.n_large_total} (8mm) = ${fr.n_total}</span></div>
    <div class="tpill"><span class="k">RMSE Teneur (${b.target_species})</span><span class="v soft">${hm.rmse_teneur.toFixed(4)}</span><span class="v ok">/ ${im.rmse_teneur.toFixed(4)}</span></div>
    <div class="tpill"><span class="k">Erreur L1 moy</span><span class="v soft">${hm.l1_mean_target.toFixed(4)}</span><span class="v ok">/ ${im.l1_mean_target.toFixed(4)}</span></div>
    <div class="tpill"><span class="k">Discrepancy RMS</span><span class="v soft">${hm.rms_discrepancy_mean.toFixed(4)}</span><span class="v ok">/ ${im.rms_discrepancy_mean.toFixed(4)}</span></div>
    <div class="tpill"><span class="k">RMSE RSD teneur</span><span class="v soft">${hm.rmse_rsd.toFixed(4)}</span><span class="v ok">/ ${im.rmse_rsd.toFixed(4)}</span></div>
    <div class="tpill"><span class="k">Horizon DEM</span><span class="v">[${b.dem_t_min}..${b.dem_t_max}] (${b.dem_total_timesteps} pas)</span></div>
  `;
}

function renderTables(b) {
  const hm = b.hom_metrics, im = b.inhom_metrics;
  const fmtT = v => v == null ? 'n/a' : `${v.toFixed(0)}`;
  document.getElementById('horizon-badge').textContent =
    `Modèle [${b.model_t_start}..${b.model_t_end}] → Prédiction [${b.pred_t_start}..${b.pred_t_end}]`;
  document.getElementById('summary-table-wrap').innerHTML = `
    <table>
      <thead><tr><th>Modèle</th><th>RMSE(Teneur)</th><th>MAE(Teneur)</th><th>L1 moy</th><th>RMS disc.</th><th>RMSE(RSD)</th><th>t50</th><th>Dérive N</th></tr></thead>
      <tbody>
        <tr><td>Homogène</td><td>${hm.rmse_teneur.toFixed(4)}</td><td>${hm.mae_teneur.toFixed(4)}</td>
            <td>${hm.l1_mean_target.toFixed(4)}</td><td>${hm.rms_discrepancy_mean.toFixed(4)}</td>
            <td>${hm.rmse_rsd.toFixed(4)}</td><td>${fmtT(hm.t50_markov)}</td><td>${hm.mass_drift_pct.toFixed(3)}%</td></tr>
        <tr><td class="ok">Inhomogène</td><td>${im.rmse_teneur.toFixed(4)}</td><td>${im.mae_teneur.toFixed(4)}</td>
            <td>${im.l1_mean_target.toFixed(4)}</td><td>${im.rms_discrepancy_mean.toFixed(4)}</td>
            <td>${im.rmse_rsd.toFixed(4)}</td><td>${fmtT(im.t50_markov)}</td><td>${im.mass_drift_pct.toFixed(3)}%</td></tr>
      </tbody>
    </table>`;

  document.getElementById('teneur-table-wrap').innerHTML = `
    <table>
      <thead><tr><th>Métrique (postprocess.py)</th><th>DEM réf.</th><th>Homogène</th><th>Inhomogène</th></tr></thead>
      <tbody>
        <tr><td>RMSE Teneur (${b.target_species})</td><td>0.0000</td><td>${hm.rmse_teneur.toFixed(4)}</td><td class="ok">${im.rmse_teneur.toFixed(4)}</td></tr>
        <tr><td>MAE Teneur (${b.target_species})</td><td>0.0000</td><td>${hm.mae_teneur.toFixed(4)}</td><td class="ok">${im.mae_teneur.toFixed(4)}</td></tr>
        <tr><td>Erreur Max Teneur</td><td>0.0000</td><td>${hm.max_err_teneur.toFixed(4)}</td><td class="ok">${im.max_err_teneur.toFixed(4)}</td></tr>
        <tr><td>Erreur L1 normalisée (moy)</td><td>0.0000</td><td>${hm.l1_mean_target.toFixed(4)}</td><td class="ok">${im.l1_mean_target.toFixed(4)}</td></tr>
        <tr><td>Discrepancy RMS (moy / max)</td><td>0.0000</td><td>${hm.rms_discrepancy_mean.toFixed(4)} / ${hm.rms_discrepancy_max.toFixed(4)}</td><td class="ok">${im.rms_discrepancy_mean.toFixed(4)} / ${im.rms_discrepancy_max.toFixed(4)}</td></tr>
        <tr><td>Temps mélange t50 / t90</td><td>${fmtT(hm.t50_dem)} / ${fmtT(hm.t90_dem)}</td><td>${fmtT(hm.t50_markov)} / ${fmtT(hm.t90_markov)}</td><td>${fmtT(im.t50_markov)} / ${fmtT(im.t90_markov)}</td></tr>
        <tr><td>Particules finales (4mm+8mm)</td><td>${hm.n_particles_total_dem.slice(-1)[0].toFixed(0)}</td><td>${hm.n_particles_total_mk.slice(-1)[0].toFixed(0)}</td><td>${im.n_particles_total_mk.slice(-1)[0].toFixed(0)}</td></tr>
        <tr><td>|λ|_max / Diag(P) moy</td><td>—</td><td>${hm.lambda_max_small.toFixed(5)} / ${hm.diag_mean_small.toFixed(3)}</td><td>${im.lambda_max_small.toFixed(5)} / ${im.diag_mean_small.toFixed(3)}</td></tr>
      </tbody>
    </table>`;
}

function renderAllPlots() {
  const b = APP.bundleA;
  if (!b) return;
  const hm = b.hom_metrics, im = b.inhom_metrics;
  const mRange = [b.model_t_start, b.model_t_end];

  // 1. Overview plot
  const om = document.getElementById('overview-metric').value;
  let sOver = [];
  if (om === 'rsd') {
    sOver = [
      { label: 'DEM (réf.)', data: hm.rsd_dem, color: '#e6ebf2', lw: 2.2 },
      { label: 'Homogène', data: hm.rsd_markov, color: '#9ec0dc', lw: 1.9, dash: [5,4] },
      { label: 'Inhomogène', data: im.rsd_markov, color: '#7dd3bd', lw: 2.1 },
    ];
  } else if (om === 'l1') {
    sOver = [
      { label: 'L1 Homogène', data: hm.l1_error_small, color: '#9ec0dc', lw: 2, dash: [5,4] },
      { label: 'L1 Inhomogène', data: im.l1_error_small, color: '#7dd3bd', lw: 2.1 },
      { label: 'L1 Total Hom', data: hm.l1_error_total, color: '#eec078', lw: 1.4 },
    ];
  } else if (om === 'rms_disc') {
    sOver = [
      { label: 'RMS disc. Homogène', data: hm.rms_discrepancy_target, color: '#9ec0dc', lw: 2, dash: [5,4] },
      { label: 'RMS disc. Inhomogène', data: im.rms_discrepancy_target, color: '#7dd3bd', lw: 2.1 },
    ];
  } else if (om === 'n_part') {
    sOver = [
      { label: 'N_tot DEM', data: hm.n_particles_total_dem, color: '#e6ebf2', lw: 2.2 },
      { label: 'N_tot Hom', data: hm.n_particles_total_mk, color: '#9ec0dc', lw: 1.9, dash: [5,4] },
      { label: 'N_tot Inhom', data: im.n_particles_total_mk, color: '#7dd3bd', lw: 2.1 },
    ];
  } else if (om === 'entropy') {
    sOver = [
      { label: 'H(t) DEM', data: hm.entropy_dem, color: '#e6ebf2', lw: 2.2 },
      { label: 'H(t) Hom', data: hm.entropy_markov, color: '#9ec0dc', lw: 1.9, dash: [5,4] },
      { label: 'H(t) Inhom', data: im.entropy_markov, color: '#7dd3bd', lw: 2.1 },
    ];
  } else {
    sOver = [
      { label: 'I_D(t) DEM', data: hm.danckwerts_dem, color: '#e6ebf2', lw: 2.2 },
      { label: 'I_D(t) Hom', data: hm.danckwerts_markov, color: '#9ec0dc', lw: 1.9, dash: [5,4] },
      { label: 'I_D(t) Inhom', data: im.danckwerts_markov, color: '#7dd3bd', lw: 2.1 },
    ];
  }
  drawLineChart(document.getElementById('chart-overview'), hm.times_eval, sOver, b.blocks, mRange);

  // 2. Teneur cell chart + N_particles + Top-pop
  const cSel = document.getElementById('teneur-cell-select').value;
  const cellsToShow = (cSel === 'all' || !cSel) ? b.top_cells.slice(0, 5) : [parseInt(cSel)];
  const tSeries = [];
  cellsToShow.forEach((cid, i) => {
    const col = PALETTE[i % PALETTE.length];
    tSeries.push({ label: `DEM C${cid}`, data: b.dem_conc_traj.map(r => r[cid]), color: col, lw: 2.0 });
    tSeries.push({ label: `Hom C${cid}`, data: b.hom_conc_traj.map(r => r[cid]), color: col, lw: 1.4, dash: [4,3] });
    if (cellsToShow.length === 1) {
      tSeries.push({ label: `Inhom C${cid}`, data: b.inhom_conc_traj.map(r => r[cid]), color: '#7dd3bd', lw: 2.2 });
    }
  });
  drawLineChart(document.getElementById('chart-teneur-cell'), hm.times_eval, tSeries, b.blocks, mRange);

  const npScope = document.getElementById('npart-species').value;
  const npSeries = npScope === 'small' ? [
    { label: 'DEM 4mm', data: hm.n_particles_small_dem, color: '#e6ebf2', lw: 2 },
    { label: 'Hom 4mm', data: hm.n_particles_small_mk, color: '#9ec0dc', lw: 1.8, dash: [5,4] },
    { label: 'Inhom 4mm', data: im.n_particles_small_mk, color: '#7dd3bd', lw: 2 },
  ] : npScope === 'large' ? [
    { label: 'DEM 8mm', data: hm.n_particles_large_dem, color: '#e6ebf2', lw: 2 },
    { label: 'Hom 8mm', data: hm.n_particles_large_mk, color: '#9ec0dc', lw: 1.8, dash: [5,4] },
    { label: 'Inhom 8mm', data: im.n_particles_large_mk, color: '#7dd3bd', lw: 2 },
  ] : [
    { label: 'DEM Total', data: hm.n_particles_total_dem, color: '#e6ebf2', lw: 2 },
    { label: 'Hom Total', data: hm.n_particles_total_mk, color: '#9ec0dc', lw: 1.8, dash: [5,4] },
    { label: 'Inhom Total', data: im.n_particles_total_mk, color: '#7dd3bd', lw: 2 },
  ];
  drawLineChart(document.getElementById('chart-npart'), hm.times_eval, npSeries, [], mRange);

  const popSeries = [];
  b.top_cells.slice(0, 4).forEach((cid, i) => {
    const col = PALETTE[i % PALETTE.length];
    popSeries.push({ label: `DEM C${cid}`, data: b.top_cell_pop_dem.map(r => r[i]), color: col, lw: 1.9 });
    popSeries.push({ label: `Inhom C${cid}`, data: b.top_cell_pop_inhom.map(r => r[i]), color: col, lw: 1.4, dash: [4,3] });
  });
  drawLineChart(document.getElementById('chart-top-pop'), hm.times_eval, popSeries, [], mRange);

  // 3. Errors & Matrices
  const l1s = document.getElementById('l1-scope').value;
  drawLineChart(document.getElementById('chart-l1'), hm.times_eval, [
    { label: 'L1 Homogène', data: l1s === 'total' ? hm.l1_error_total : (b.target_species === 'large' ? hm.l1_error_large : hm.l1_error_small), color: '#9ec0dc', lw: 2, dash: [5,4] },
    { label: 'L1 Inhomogène', data: l1s === 'total' ? im.l1_error_total : (b.target_species === 'large' ? im.l1_error_large : im.l1_error_small), color: '#7dd3bd', lw: 2.2 },
  ], b.blocks, mRange);

  const discGrid = document.getElementById('heat-disc-chain').value === 'hom'
    ? b.discrepancy_heatmap_hom : b.discrepancy_heatmap_inhom;
  drawRectHeatmap(document.getElementById('heat-discrepancy'), discGrid, b.heatmap_times, b.active_cell_ids);

  const matMode = document.getElementById('mat-view').value;
  const blkIdx = parseInt(document.getElementById('block-select').value) || 0;
  const isLarge = b.target_species === 'large';
  if (matMode === 'hom') {
    drawMatrixHeatmap(document.getElementById('heat-matrix'), isLarge ? b.p_hom_large : b.p_hom_small, false);
  } else if (matMode === 'block' && b.blocks[blkIdx]) {
    const bl = b.blocks[blkIdx];
    drawMatrixHeatmap(document.getElementById('heat-matrix'), isLarge ? bl.p_large : bl.p_small, false);
  } else if (matMode === 'diff' && b.blocks.length >= 2) {
    const k0 = Math.min(blkIdx, b.blocks.length - 2);
    const a = isLarge ? b.blocks[k0].p_large : b.blocks[k0].p_small;
    const c = isLarge ? b.blocks[k0+1].p_large : b.blocks[k0+1].p_small;
    const diff = a.map((row, r) => row.map((v, j) => (c[r][j] ?? 0) - (v ?? 0)));
    drawMatrixHeatmap(document.getElementById('heat-matrix'), diff, true);
  }

  // p_ij(t) polynomial evolution curves
  if (b.p_laws && b.p_laws.length) {
    const xs = b.p_laws[0].block_times;
    const plSeries = [];
    b.p_laws.slice(0, 4).forEach((l, i) => {
      const col = PALETTE[i % PALETTE.length];
      plSeries.push({ label: `p(${l.src}→${l.dst}) R²=${l.r2.toFixed(2)}`, data: l.values, color: col, lw: 2 });
      plSeries.push({ label: `fit d${l.degree}`, data: l.predicted, color: col, lw: 1.2, dash: [3,3] });
    });
    drawLineChart(document.getElementById('chart-plaws'), xs, plSeries);
  }

  // 4. Dual comparison
  if (APP.bundleB) {
    const bB = APP.bundleB;
    document.getElementById('dual-a-lbl').textContent = `${b.method} (${b.mesh.n_cells}c)`;
    drawLineChart(document.getElementById('chart-dual'), hm.times_eval, [
      { label: `DEM réf.`, data: hm.rsd_dem, color: '#e6ebf2', lw: 2 },
      { label: `${b.method} Inhom`, data: im.rsd_markov, color: '#7dd3bd', lw: 2.1 },
      { label: `${bB.method} Inhom`, data: bB.inhom_metrics.rsd_markov, color: '#eec078', lw: 2.1 },
      { label: `${b.method} L1`, data: im.l1_error_small, color: '#9ec0dc', lw: 1.4, dash: [4,3] },
      { label: `${bB.method} L1`, data: bB.inhom_metrics.l1_error_small, color: '#e5989b', lw: 1.4, dash: [4,3] },
    ], [], mRange);
  }
}

function renderBenchmark() {
  const rows = APP.benchRows;
  if (!rows) return;
  drawBarBenchmark(document.getElementById('chart-bench'), rows);
  document.getElementById('bench-table-wrap').innerHTML = `
    <table>
      <thead><tr><th>Méthode</th><th>Vis./Cell</th><th>N_DEM</th><th>N_Inhom</th><th>RMSE Teneur (H/I)</th><th>MAE Teneur (H/I)</th><th>L1 moy (H/I)</th><th>RMS disc (H/I)</th><th>RMSE RSD (H/I)</th><th>t50 (H/I)</th></tr></thead>
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
          <td>${r.t50_hom==null?'-':r.t50_hom.toFixed(0)} / ${r.t50_inhom==null?'-':r.t50_inhom.toFixed(0)}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;
}

function render3DAll() {
  const b = APP.bundleA;
  if (!b) return;
  const frame = b.frames[Math.min(APP.frameIdx, b.frames.length - 1)] || b.frames[0];
  document.getElementById('time-label').textContent = `t = ${frame.timestep}`;
  document.getElementById('cnt-small').textContent = frame.n_small_total;
  document.getElementById('cnt-large').textContent = frame.n_large_total;
  document.getElementById('cnt-total').textContent = frame.n_total;
  document.getElementById('hud-info').innerHTML =
    `<b style="color:var(--text);">${b.label}</b> · t=${frame.timestep}/${b.dem_t_max}<br>` +
    `Modèle [${b.model_t_start}..${b.model_t_end}] · Critère: teneur <b>${b.target_species}</b><br>` +
    `Teneur RMSE: Hom ${b.hom_metrics.rmse_teneur.toFixed(4)} · Inhom <span class="ok">${b.inhom_metrics.rmse_teneur.toFixed(4)}</span>`;

  renderViewport3D(document.getElementById('canvas3d'), b, true);
  if (document.getElementById('pane-benchmark').classList.contains('active')) {
    renderViewport3D(document.getElementById('canvas3d-a'), b, false);
    if (APP.bundleB) renderViewport3D(document.getElementById('canvas3d-b'), APP.bundleB, false);
  }
}

function renderAllViews() {
  if (!APP.bundleA) return;
  renderTelemetry(APP.bundleA);
  renderTables(APP.bundleA);
  render3DAll();
  renderAllPlots();
}

function wire3DEvents(canvas) {
  canvas.addEventListener('mousedown', e => {
    APP.drag.active = true;
    APP.drag.mode = (e.button === 2 || e.shiftKey) ? 'pan' : 'orbit';
    APP.drag.x = e.clientX; APP.drag.y = e.clientY;
  });
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  window.addEventListener('mousemove', e => {
    if (!APP.drag.active) return;
    const dx = e.clientX - APP.drag.x, dy = e.clientY - APP.drag.y;
    APP.drag.x = e.clientX; APP.drag.y = e.clientY;
    if (APP.drag.mode === 'orbit') {
      APP.cam.yaw += dx * 0.008;
      APP.cam.pitch = Math.max(-1.54, Math.min(1.54, APP.cam.pitch + dy * 0.008));
    } else {
      const b = camBasis(APP.cam);
      const scale = APP.cam.distance * 0.0018;
      for (let d = 0; d < 3; d++) APP.cam.target[d] -= dx * scale * b.right[d] - dy * scale * b.up[d];
    }
    render3DAll();
  });
  window.addEventListener('mouseup', () => { APP.drag.active = false; });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    APP.cam.distance = Math.max(0.03, Math.min(20, APP.cam.distance * (e.deltaY > 0 ? 1.1 : 0.9)));
    render3DAll();
  }, { passive: false });
  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 26 * 26;
    for (const c of APP.projectedCells) {
      const d2 = (c.x - x)**2 + (c.y - y)**2;
      if (d2 < bd) { bd = d2; best = c; }
    }
    APP.selectedCell = best ? best.id : null;
    const box = document.getElementById('cell-info');
    if (best && APP.bundleA) {
      const fr = APP.bundleA.frames[APP.frameIdx] || APP.bundleA.frames[0];
      const ei = nearestEvalRow(APP.bundleA, fr.timestep);
      const tD = APP.bundleA.dem_conc_traj[ei]?.[best.id] ?? fr.cell_conc_dem[best.id] ?? 0;
      const tH = APP.bundleA.hom_conc_traj[ei]?.[best.id] ?? 0;
      const tI = APP.bundleA.inhom_conc_traj[ei]?.[best.id] ?? 0;
      const crm = APP.bundleA.inhom_metrics.cell_rmse_per_cell?.[best.id] ?? 0;
      box.innerHTML = `<b style="color:#fff">Cellule #${best.id}</b> (zone ${best.cell.zone})<br>` +
        `Particules N_${best.id}(t=${fr.timestep}) : <b>${best.pop}</b><br>` +
        `Teneur (${APP.bundleA.target_species}) : DEM <b>${tD.toFixed(3)}</b> · Hom <b>${tH.toFixed(3)}</b> · Inhom <b class="ok">${tI.toFixed(3)}</b><br>` +
        `Discrepancy RMSE cellule : <b>${crm.toFixed(4)}</b>`;
      box.classList.add('show');
    } else {
      box.classList.remove('show');
    }
    render3DAll();
  });
}

function wireMatrixHover(canvas) {
  canvas.addEventListener('mousemove', e => {
    const m = canvas._heatMeta;
    if (!m) return;
    const r = canvas.getBoundingClientRect();
    const c = Math.floor((e.clientX - r.left - m.ox) / (m.size / m.n));
    const row = Math.floor((e.clientY - r.top - m.oy) / (m.size / m.n));
    const lbl = document.getElementById('hover-p-lbl');
    if (row >= 0 && row < m.n && c >= 0 && c < m.n) {
      const v = m.matrix[row][c];
      lbl.textContent = `(${row}→${c}) = ${v == null ? 'NaN' : v.toFixed(4)}`;
    }
  });
}

async function init() {
  let data = null;
  try {
    const res = await fetch('/api/methods');
    if (res.ok) data = await res.json();
  } catch (_) {}
  if (!data) {
    APP.staticMode = true;
    const res = await fetch('static_data/methods.json');
    data = await res.json();
    const pill = document.querySelector('.brand-pill');
    if (pill) pill.textContent = 'Liquid Glass · 3D · GitHub Pages';
  }
  APP.methods = data.methods;

  const opts = APP.methods.map(m => `<option value="${m.id}">${m.name}</option>`).join('');
  document.getElementById('method').innerHTML = opts;
  document.getElementById('method-b').innerHTML = opts;
  document.getElementById('method-b').value = 'voronoi';
  renderMethodParams('cartesian');

  const planResp = await fetch(APP.staticMode ? 'static_data/egui_plan.txt' : '/api/egui_plan');
  document.getElementById('egui-plan-text').textContent = await planResp.text();

  document.querySelectorAll('.rtab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.rtab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('pane-' + btn.dataset.tab).classList.add('active');
      if (btn.dataset.tab === 'benchmark' && !APP.bundleB) runDual();
      else setTimeout(() => { renderAllViews(); renderBenchmark(); }, 20);
    });
  });

  document.getElementById('method').addEventListener('change', e => renderMethodParams(e.target.value));
  document.getElementById('run-btn').addEventListener('click', runPrimary);
  document.getElementById('run-dual-btn').addEventListener('click', runDual);
  document.getElementById('run-bench-btn').addEventListener('click', runBenchmark);
  document.getElementById('dataset').addEventListener('change', e => {
    document.getElementById('start_index').value = e.target.value === 'real_chunks' ? 157 : 40;
    runPrimary();
  });
  ['color-mode','show-small','show-large','show-wire','show-faces','show-vel','show-edges','show-drum'].forEach(id =>
    document.getElementById(id).addEventListener('change', render3DAll)
  );
  ['overview-metric','teneur-cell-select','npart-species','l1-scope','heat-disc-chain','mat-view','block-select'].forEach(id =>
    document.getElementById(id).addEventListener('change', renderAllPlots)
  );
  document.getElementById('time-slider').addEventListener('input', e => {
    APP.frameIdx = parseInt(e.target.value) || 0;
    render3DAll();
  });
  document.getElementById('play-btn').addEventListener('click', () => {
    APP.playing = !APP.playing;
    document.getElementById('play-btn').textContent = APP.playing ? 'Pause' : 'Lecture';
    if (APP.playing) {
      APP.playTimer = setInterval(() => {
        if (!APP.bundleA) return;
        APP.frameIdx = (APP.frameIdx + 1) % APP.bundleA.frames.length;
        document.getElementById('time-slider').value = APP.frameIdx;
        render3DAll();
      }, 130);
    } else {
      clearInterval(APP.playTimer);
    }
  });

  wire3DEvents(document.getElementById('canvas3d'));
  wire3DEvents(document.getElementById('canvas3d-a'));
  wire3DEvents(document.getElementById('canvas3d-b'));
  wireMatrixHover(document.getElementById('heat-matrix'));
  window.addEventListener('resize', () => { renderAllViews(); renderBenchmark(); });

  await runPrimary();
}

init();
</script>
</body>
</html>
"##;
