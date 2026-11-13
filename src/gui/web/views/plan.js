async function loadPlanText() {
  const el = document.getElementById('egui-plan-text');
  if (!el) return;
  try {
    const planResp = await fetch(APP.staticMode ? 'static_data/egui_plan.txt' : '/api/egui_plan');
    if (planResp.ok) {
      el.textContent = await planResp.text();
    }
  } catch (_) {}
}
