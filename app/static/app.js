"use strict";

const COLORS = {
  green: "#0b6e4f",
  blue: "#0b2a4a",
  blueLight: "#9ecae1",
  gold: "#d9a441",
  purple: "#8e44ad",
  grid: "#d8e1e7",
  muted: "#5b7183",
};

const SENEGAL_ZONES = [
  "Dakar", "Saint-Louis", "Matam", "Podor",
  "Tambacounda", "Kaolack", "Louga", "Ziguinchor",
];

window.fmt = function (v, d = 2) {
  const n = Number(v);
  return Number.isFinite(n) ? n.toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—";
};
function fmtInt(v) {
  return window.fmt(v, 0);
}

function currentCapacity() {
  const select = document.getElementById("capacity");
  return select ? parseFloat(select.value) || 100 : 100;
}
const tooltip = document.getElementById("tooltip");
const hoverPoints = {};
let lastPayload = null;
let lastImpactAll = null;

function prepare(id) {
  const canvas = document.getElementById(id);
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, rect.width);
  const height = Math.max(180, rect.height);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.font = "11px Segoe UI, Arial, sans-serif";
  hoverPoints[id] = [];
  return { ctx, w: width, h: height };
}

function scale(min, max, lo, hi) {
  if (!isFinite(min) || !isFinite(max) || max === min) return () => (lo + hi) / 2;
  return (v) => lo + ((v - min) / (max - min)) * (hi - lo);
}

function grid(ctx, w, h, pad, columns, rows) {
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= columns; i++) {
    const x = pad.l + ((w - pad.l - pad.r) * i) / columns;
    ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b);
  }
  for (let i = 0; i <= rows; i++) {
    const y = pad.t + ((h - pad.t - pad.b) * i) / rows;
    ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y);
  }
  ctx.stroke();
  ctx.beginPath();
  ctx.strokeStyle = "#8fa3b0";
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b);
  ctx.stroke();
}

function addHover(id, x, y, text) {
  hoverPoints[id].push({ x, y, text });
}

document.addEventListener("mousemove", (event) => {
  let best = null, bestDist = 18 * 18;
  for (const id of Object.keys(hoverPoints)) {
    const canvas = document.getElementById(id);
    if (!canvas) continue;
    const rect = canvas.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) continue;
    const px = event.clientX - rect.left, py = event.clientY - rect.top;
    for (const point of hoverPoints[id]) {
      const dist = (point.x - px) ** 2 + (point.y - py) ** 2;
      if (dist < bestDist) { bestDist = dist; best = point; }
    }
  }
  if (best) {
    tooltip.style.display = "block";
    tooltip.textContent = best.text;
    tooltip.style.left = Math.min(event.clientX + 14, window.innerWidth - 270) + "px";
    tooltip.style.top = Math.max(8, event.clientY - 42) + "px";
  } else {
    tooltip.style.display = "none";
  }
});

function drawBands(payload) {
  const { ctx, w, h } = prepare("chart-band");
  const structure = payload.band_structure;
  const bands = structure.bands, path = structure.path;
  let lo = Infinity, hi = -Infinity;
  bands.forEach((band) => band.forEach((v) => { lo = Math.min(lo, v); hi = Math.max(hi, v); }));
  const margin = Math.max(0.2, (hi - lo) * 0.08);
  lo = Math.min(lo - margin, -margin); hi = Math.max(hi + margin, margin);
  const pad = { l: 58, r: 16, t: 14, b: 32 };
  const xv = scale(0, 1, pad.l, w - pad.r);
  const yv = scale(lo, hi, h - pad.b, pad.t);
  grid(ctx, w, h, pad, 3, 4);
  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ["Γ", "X", "M", "Γ"].forEach((label, i) => ctx.fillText(label, xv(i / 3), h - pad.b + 6));
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = lo + ((hi - lo) * i) / 4;
    ctx.fillText(fmt(value), pad.l - 6, yv(value));
  }
  ctx.strokeStyle = COLORS.gold;
  ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(pad.l, yv(0)); ctx.lineTo(w - pad.r, yv(0)); ctx.stroke();
  ctx.setLineDash([]);
  const palette = [COLORS.blue, COLORS.green, COLORS.purple, COLORS.gold, "#d95f0e"];
  bands.forEach((band, index) => {
    ctx.strokeStyle = palette[index % palette.length];
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    path.forEach((p, i) => {
      const px = xv(p), py = yv(band[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      if (i % 4 === 0) addHover("chart-band", px, py, `E${index + 1} = ${fmt(band[i])} eV · k = ${fmt(p)}`);
    });
    ctx.stroke();
  });
  ctx.fillStyle = COLORS.blue;
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText(`${bands.length} bandes · E_g = ${fmt(structure.gap_ev)} eV`, pad.l + 6, pad.t + 4);
  document.getElementById("band-info").textContent = structure.model;
}

function drawBerry(payload) {
  const { ctx, w, h } = prepare("chart-berry");
  const berry = payload.berry;
  const field = berry.field, n = field.length;
  let max = 0;
  field.forEach((row) => row.forEach((v) => { max = Math.max(max, Math.abs(v)); }));
  max = max || 1;
  const pad = { l: 46, r: 52, t: 16, b: 30 };
  const size = Math.min((w - pad.l - pad.r) / n, (h - pad.t - pad.b) / n);
  const ox = pad.l + ((w - pad.l - pad.r) - size * n) / 2;
  const oy = pad.t + ((h - pad.t - pad.b) - size * n) / 2;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const value = field[i][j] / max;
      const a = Math.round(Math.min(1, Math.abs(value)) * 210);
      ctx.fillStyle = value >= 0 ? `rgb(${a + 40},${40 + Math.round(a * 0.2)},60)` : `rgb(35,${60 + Math.round(a * 0.3)},${a + 40})`;
      ctx.fillRect(ox + j * size, oy + i * size, size + 0.5, size + 0.5);
      addHover("chart-berry", ox + j * size + size / 2, oy + i * size + size / 2,
        `Ω(k) = ${fmt(field[i][j])} · plaquette (${i + 1}, ${j + 1})`);
    }
  }
  ctx.strokeStyle = "#8fa3b0";
  ctx.strokeRect(ox, oy, size * n, size * n);
  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  ctx.fillText("k₂", ox + size * n / 2, oy + size * n + 8);
  ctx.save();
  ctx.translate(ox - 14, oy + size * n / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("k₁", 0, 0); ctx.restore();
  const barX = w - pad.r + 12, barY = oy, barH = size * n;
  for (let i = 0; i < 60; i++) {
    const value = 1 - (2 * i) / 59;
    const a = Math.round(Math.abs(value) * 210);
    ctx.fillStyle = value >= 0 ? `rgb(${a + 40},${40 + Math.round(a * 0.2)},60)` : `rgb(35,${60 + Math.round(a * 0.3)},${a + 40})`;
    ctx.fillRect(barX, barY + (i * barH) / 60, 10, barH / 60 + 0.5);
  }
  ctx.textAlign = "left";
  ctx.fillText(fmt(max), barX, barY - 14);
  ctx.fillText(fmt(-max), barX, barY + barH + 2);
  ctx.fillStyle = COLORS.green;
  ctx.font = "bold 20px Segoe UI, Arial, sans-serif";
  ctx.fillText(`C = ${berry.chern}`, pad.l, pad.t - 2);
  document.getElementById("chern-info").textContent = `${berry.method} · C num. = ${fmt(berry.chern_numeric)}`;
}

function drawJV(payload) {
  const { ctx, w, h } = prepare("chart-jv");
  const jv = payload.jv, x = jv.voltage, y = jv.current_ma_cm2;
  const yMin = Math.min(0, ...y), yMax = Math.max(...y) * 1.08 || 1;
  const xMax = jv.voc * 1.05 || 1;
  const pad = { l: 58, r: 16, t: 16, b: 34 };
  const xv = scale(0, xMax, pad.l, w - pad.r);
  const yv = scale(yMin, yMax, h - pad.b, pad.t);
  grid(ctx, w, h, pad, 5, 4);
  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 5; i++) ctx.fillText(fmt((xMax * i) / 5), xv((xMax * i) / 5), h - pad.b + 6);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = yMin + ((yMax - yMin) * i) / 4;
    ctx.fillText(fmt(value), pad.l - 6, yv(value));
  }
  ctx.strokeStyle = COLORS.green; ctx.lineWidth = 2.2;
  ctx.beginPath();
  x.forEach((value, i) => {
    const px = xv(value), py = yv(y[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    addHover("chart-jv", px, py, `V = ${fmt(value)} V · J = ${fmt(y[i])} mA/cm²`);
  });
  ctx.stroke();
  let best = 0;
  x.forEach((v, i) => { if (v * y[i] > x[best] * y[best]) best = i; });
  ctx.fillStyle = COLORS.blueLight; ctx.strokeStyle = COLORS.blue;
  ctx.beginPath(); ctx.arc(xv(x[best]), yv(y[best]), 5, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
  ctx.fillStyle = COLORS.blue;
  ctx.textAlign = "left"; ctx.textBaseline = "top";
  ctx.fillText(`V_oc = ${fmt(jv.voc)} V · J_sc = ${fmt(jv.jsc_ma_cm2)} mA/cm²`, pad.l + 6, pad.t + 2);
  ctx.fillText(`FF = ${fmt(jv.ff)} · η = ${fmt(jv.efficiency_percent)} % · T_j = ${fmt(jv.junction_c)} °C`, pad.l + 6, pad.t + 18);
  document.getElementById("jv-info").textContent = jv.method;
}

function drawRagone(payload) {
  const { ctx, w, h } = prepare("chart-ragone");
  const ragone = payload.ragone;
  const px = ragone.power_w_kg, py = ragone.energy_wh_kg;
  const xMax = Math.max(...px) * 1.1 || 1, yMax = Math.max(...py) * 1.15 || 1;
  const pad = { l: 58, r: 18, t: 16, b: 34 };
  const xv = scale(0, xMax, pad.l, w - pad.r), yv = scale(0, yMax, h - pad.b, pad.t);
  grid(ctx, w, h, pad, 5, 4);
  ctx.fillStyle = COLORS.muted;
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  for (let i = 0; i <= 5; i++) ctx.fillText(fmt((xMax * i) / 5), xv((xMax * i) / 5), h - pad.b + 6);
  ctx.textAlign = "right"; ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) ctx.fillText(fmt((yMax * i) / 4), pad.l - 6, yv((yMax * i) / 4));
  ctx.strokeStyle = COLORS.purple; ctx.lineWidth = 2;
  ctx.beginPath();
  px.forEach((p, i) => { const cx = xv(p), cy = yv(py[i]); if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy); });
  ctx.stroke();
  px.forEach((p, i) => {
    const cx = xv(p), cy = yv(py[i]);
    ctx.fillStyle = i === 0 ? COLORS.gold : COLORS.purple;
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = COLORS.muted;
    ctx.textAlign = "left"; ctx.textBaseline = "bottom";
    ctx.fillText(`${fmtInt(ragone.current_a_kg[i])} A/kg`, cx + 7, cy - 5);
    addHover("chart-ragone", cx, cy, `P = ${fmt(p)} W/kg · E = ${fmt(py[i])} Wh/kg · I = ${fmt(ragone.current_a_kg[i])} A/kg`);
  });
  document.getElementById("ragone-info").textContent = `${fmt(py[0])} Wh/kg à ${fmt(ragone.temperature_c)} °C`;
}

function drawImpact(results) {
  const { ctx, w, h } = prepare("chart-impact");
  const names = results.map((r) => r.site);
  const pad = { l: 56, r: 20, t: 28, b: 34 };
  const half = (w - pad.l - pad.r - 22) / 2;
  const energyMax = Math.max(...results.map((r) => Math.max(r.champion_mwh_an, r.classique_mwh_an))) * 1.15 || 1;
  const costMax = Math.max(...results.map((r) => Math.max(r.lcoe_champion_fcfa_kwh, r.lcoe_classique_fcfa_kwh))) * 1.15 || 1;
  ["Énergie utile (MWh/an)", "Coût du kWh (FCFA/kWh)"].forEach((title, part) => {
    const start = pad.l + part * (half + 22);
    const max = part === 0 ? energyMax : costMax;
    const yv = scale(0, max, h - pad.b, pad.t + 16);
    ctx.fillStyle = COLORS.blue;
    ctx.font = "bold 12px Segoe UI, Arial, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    ctx.fillText(title, start + half / 2, pad.t - 12);
    ctx.font = "11px Segoe UI, Arial, sans-serif";
    ctx.strokeStyle = COLORS.grid;
    ctx.beginPath();
    [0, 0.5, 1].forEach((f) => {
      const y = yv(max * f);
      ctx.moveTo(start, y); ctx.lineTo(start + half, y);
      ctx.fillStyle = COLORS.muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
      ctx.fillText(fmt(max * f), start - 4, y);
    });
    ctx.stroke();
    const groupWidth = half / results.length;
    results.forEach((r, i) => {
      const cx = start + (i + 0.5) * groupWidth;
      const champ = part === 0 ? r.champion_mwh_an : r.lcoe_champion_fcfa_kwh;
      const base = part === 0 ? r.classique_mwh_an : r.lcoe_classique_fcfa_kwh;
      [[champ, COLORS.green, -1], [base, COLORS.blueLight, 1]].forEach(([value, color, side]) => {
        const bw = groupWidth * 0.26;
        const bx = cx + side * bw * 0.55 - bw / 2;
        const by = yv(value);
        ctx.fillStyle = color;
        ctx.fillRect(bx, by, bw, h - pad.b - by);
        addHover("chart-impact", bx + bw / 2, by,
          `${r.site} · ${part === 0 ? "Champion" : "Classique"} : ${fmt(value)} ${part === 0 ? "MWh/an" : "FCFA/kWh"}`);
      });
      ctx.fillStyle = COLORS.muted;
      ctx.textAlign = "center"; ctx.textBaseline = "top";
      ctx.fillText(names[i], cx, h - pad.b + 6);
    });
  });
  ctx.fillStyle = COLORS.gold;
  ctx.font = "bold 9px Segoe UI, Arial, sans-serif";
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  results.forEach((r, i) => {
    const x = pad.l + (i + 0.5) * (half / results.length);
    const millions = fmtInt(r.economies_fcfa_an / 1e6);
    const y = pad.t + 2 + (results.length > 4 && i % 2 === 1 ? 11 : 0);
    ctx.fillText(`+${millions} M`, x, y);
    addHover("chart-impact", x, y + 5,
      `${r.site} · économies : +${millions} M FCFA/an`);
  });
}

function setStatus(message, kind = "") {
  const status = document.getElementById("status");
  status.textContent = message;
  status.className = "status" + (kind ? " " + kind : "");
}

function updateKpis(impact, capacityKwc = 100) {
  const factor = capacityKwc / 100;
  const kwc = fmtInt(capacityKwc);
  document.getElementById("kpi-mwh").textContent = `+${fmt(impact.gain_mwh_an * factor)}`;
  document.getElementById("kpi-co2").textContent = fmt(impact.co2_evitees_t_an * factor);
  document.getElementById("kpi-fcfa").textContent = `${fmtInt(impact.economies_fcfa_an * factor / 1e6)} M`;
  document.getElementById("kpi-mwh-sub").textContent = `MWh/an à ${kwc} kWc`;
  document.getElementById("kpi-co2-sub").textContent = `tonnes/an à ${kwc} kWc`;
  document.getElementById("kpi-fcfa-sub").textContent = `FCFA/an à ${kwc} kWc`;
  const lcoe = document.getElementById("kpi-lcoe");
  lcoe.textContent = fmt(impact.lcoe_champion_fcfa_kwh);
  lcoe.title = `Classique : ${fmt(impact.lcoe_classique_fcfa_kwh)} FCFA/kWh`;
  document.getElementById("kpi-lcoe-sub").textContent = `FCFA/kWh · stable à l'échelle`;
}

function optionFor(material) {
  const option = document.createElement("option");
  option.value = material.id;
  option.textContent = `${material.id} · ${material.formula}${material.is_champion ? " (champion)" : ""}`;
  return option;
}

async function loadCatalog() {
  const response = await fetch("/api/catalog");
  if (!response.ok) throw new Error("Catalogue indisponible");
  const payload = await response.json();
  const select = document.getElementById("material");
  select.innerHTML = "";
  const championGroup = document.createElement("optgroup");
  championGroup.label = "Matériaux champions UAM";
  payload.materials.filter((m) => m.is_champion).forEach((m) => championGroup.appendChild(optionFor(m)));
  const otherGroup = document.createElement("optgroup");
  otherGroup.label = `Autres candidats (${payload.count - 2})`;
  payload.materials.filter((m) => !m.is_champion).forEach((m) => otherGroup.appendChild(optionFor(m)));
  select.appendChild(championGroup);
  select.appendChild(otherGroup);
  select.value = "UAM-1384";
}

async function loadImpactAll(temperature) {
  const results = await Promise.all(SENEGAL_ZONES.map(async (city) => {
    const response = await fetch(`/api/impact?city=${encodeURIComponent(city)}&temperature_c=${temperature}`);
    if (!response.ok) throw new Error("Bilan d'impact indisponible");
    return response.json();
  }));
  drawImpact(results);
  return results;
}

function render(payload) {
  drawBands(payload);
  drawBerry(payload);
  drawJV(payload);
  drawRagone(payload);
  updateKpis(payload.impact, currentCapacity());
  if (lastImpactAll) drawImpact(lastImpactAll);
}

async function runPipeline() {
  const button = document.getElementById("run");
  const material = document.getElementById("material").value;
  const city = document.getElementById("city").value;
  const temperature = parseFloat(document.getElementById("temperature").value);
  button.disabled = true;
  setStatus("Pipeline automatique en cours : bandes, Berry/Chern, J(V), Ragone et impact…");
  try {
    const response = await fetch("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ material_id: material, temperature_c: temperature, city }),
    });
    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.detail || "Erreur du pipeline");
    }
    lastPayload = await response.json();
    render(lastPayload);
    lastImpactAll = await loadImpactAll(temperature);
    setStatus(`Pipeline terminé · ${material} · ${city} · ${fmt(temperature)} °C · ${lastPayload.material.role} · ${lastPayload.site.irradiance_w_m2} W/m²`, "ok");
  } catch (error) {
    setStatus("Erreur : " + error.message, "error");
  } finally {
    button.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  const slider = document.getElementById("temperature");
  const citySelect = document.getElementById("city");
  slider.addEventListener("input", () => {
    document.getElementById("temp-value").textContent = slider.value + " °C";
  });
  citySelect.addEventListener("change", () => {
    const preset = citySelect.options[citySelect.selectedIndex].dataset.temp;
    if (preset) {
      slider.value = preset;
      document.getElementById("temp-value").textContent = preset + " °C";
    }
  });
  const capacitySelect = document.getElementById("capacity");
  capacitySelect.addEventListener("change", () => {
    const kwc = currentCapacity();
    const usage = capacitySelect.options[capacitySelect.selectedIndex].dataset.usage || "Installation solaire";
    document.getElementById("capacity-note").textContent = `${usage} · ${fmtInt(kwc)} kWc`;
    if (lastPayload) {
      updateKpis(lastPayload.impact, kwc);
      setStatus(`Puissance mise à jour : ${fmtInt(kwc)} kWc — ${usage}. Les 4 indicateurs ont été recalculés en direct.`, "ok");
    } else {
      setStatus(`Puissance sélectionnée : ${fmtInt(kwc)} kWc — ${usage}.`);
    }
  });
  document.getElementById("run").addEventListener("click", runPipeline);
  document.getElementById("pdf").addEventListener("click", () => {
    const link = document.createElement("a");
    link.href = "/api/report/pdf";
    link.download = "UAM_TopoMat_Fiche_Champions.pdf";
    document.body.appendChild(link);
    link.click();
    link.remove();
    setStatus("Téléchargement de la fiche technique PDF officielle…", "ok");
  });
  window.addEventListener("resize", () => {
    if (lastPayload) render(lastPayload);
  });
  try {
    await loadCatalog();
    await runPipeline();
  } catch (error) {
    setStatus("Impossible d'initialiser l'interface : " + error.message, "error");
  }
});
