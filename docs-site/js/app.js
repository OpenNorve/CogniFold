// app.js — data-driven entry point. Loads memory_coverage.json and renders the
// coverage ring, the brain visualization, and the systems side panel.

import { renderBrain } from "./brain.js";

const STATUS_LABEL = { covered: "Modeled", partial: "Partial", planned: "Planned" };

async function loadData() {
  const res = await fetch("./data/memory_coverage.json", { cache: "no-cache" });
  if (!res.ok) throw new Error(`Failed to load coverage data: ${res.status}`);
  return res.json();
}

// ---- Coverage ring -------------------------------------------------------
function renderRing(pct) {
  const ring = document.getElementById("coverage-ring");
  const R = 130;
  const C = 2 * Math.PI * R;
  ring.innerHTML = `
    <svg viewBox="0 0 300 300" aria-hidden="true">
      <defs>
        <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stop-color="#38e8ff"/>
          <stop offset="55%" stop-color="#9b6bff"/>
          <stop offset="100%" stop-color="#ff5cc8"/>
        </linearGradient>
      </defs>
      <circle class="track" cx="150" cy="150" r="${R}"/>
      <circle class="progress" cx="150" cy="150" r="${R}"
              stroke-dasharray="${C}" stroke-dashoffset="${C}"/>
    </svg>
    <div class="center">
      <div class="pct"><span class="num">0</span><span class="sym">%</span></div>
      <div class="label">Human brain memory modeled</div>
    </div>`;

  const progress = ring.querySelector(".progress");
  const num = ring.querySelector(".num");
  const target = C - (pct / 100) * C;

  // Animate when the ring scrolls into view (or immediately if reduced motion).
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const animate = () => {
    requestAnimationFrame(() => { progress.style.strokeDashoffset = String(target); });
    if (reduce) { num.textContent = String(pct); return; }
    const dur = 1800, start = performance.now();
    const tick = (t) => {
      const k = Math.min(1, (t - start) / dur);
      const eased = 1 - Math.pow(1 - k, 3);
      num.textContent = String(Math.round(eased * pct));
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };

  const io = new IntersectionObserver((entries, obs) => {
    for (const e of entries) if (e.isIntersecting) { animate(); obs.disconnect(); }
  }, { threshold: 0.4 });
  io.observe(ring);
}

// ---- Systems side panel --------------------------------------------------
function renderSystems(systems, brain, tooltip) {
  const panel = document.getElementById("systems-panel");
  panel.innerHTML = "";

  for (const sys of systems) {
    const card = document.createElement("article");
    card.className = `sys-card s-${sys.status}`;
    card.dataset.system = sys.id;
    card.setAttribute("tabindex", "0");
    card.innerHTML = `
      <div class="sys-head">
        <div>
          <div class="sys-name">${escapeHtml(sys.name)}</div>
          <div class="sys-region">${escapeHtml(sys.brain_region)} · ${escapeHtml(sys.taxonomy_group)}</div>
        </div>
        <span class="chip ${sys.status}">${STATUS_LABEL[sys.status]}</span>
      </div>
      <div class="sys-detail">
        <p>${escapeHtml(sys.description)}</p>
        <div class="row"><b>Mechanism:</b> ${escapeHtml(sys.cognifold_mechanism)}</div>
        <div class="row evidence">${escapeHtml(sys.evidence)}</div>
      </div>`;

    const enter = (ev) => {
      brain.highlightBySystem(sys.id);
      showTooltip(tooltip, sys, ev);
    };
    const leave = () => { brain.highlightBySystem(null); hideTooltip(tooltip); };

    card.addEventListener("mouseenter", enter);
    card.addEventListener("mousemove", (ev) => positionTooltip(tooltip, ev));
    card.addEventListener("mouseleave", leave);
    card.addEventListener("focus", () => brain.highlightBySystem(sys.id));
    card.addEventListener("blur", () => brain.highlightBySystem(null));

    // click to expand details
    card.addEventListener("click", () => card.classList.toggle("active"));
    card.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); card.classList.toggle("active"); }
    });

    panel.appendChild(card);
  }
}

function highlightSystemCard(sid, on) {
  const card = document.querySelector(`.sys-card[data-system="${sid}"]`);
  if (card) card.classList.toggle("active-region", on);
}

// ---- Tooltip helpers -----------------------------------------------------
function showTooltip(tip, sys, ev) {
  const color = { covered: "#38e8ff", partial: "#ffb347", planned: "#4a5380" }[sys.status];
  tip.innerHTML = `
    <div class="t-title">${escapeHtml(sys.name)}</div>
    <div class="t-status" style="color:${color}">${STATUS_LABEL[sys.status]} · ${escapeHtml(sys.brain_region)}</div>
    <div style="margin-top:6px;color:var(--text-dim)">${escapeHtml(sys.cognifold_mechanism)}</div>`;
  tip.classList.add("show");
  positionTooltip(tip, ev);
}
function positionTooltip(tip, ev) {
  const pad = 16;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  const r = tip.getBoundingClientRect();
  if (x + r.width > window.innerWidth - 10) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 10) y = ev.clientY - r.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTooltip(tip) { tip.classList.remove("show"); }

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- Scroll reveal -------------------------------------------------------
function setupReveal() {
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }, { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach((el) => io.observe(el));
}

// ---- Boot ----------------------------------------------------------------
async function main() {
  setupReveal();
  const tooltip = document.getElementById("tooltip");

  let data;
  try {
    data = await loadData();
  } catch (err) {
    console.error(err);
    document.getElementById("systems-panel").innerHTML =
      `<p style="color:var(--text-dim)">Could not load coverage data. Serve this folder over HTTP (e.g. <code>python -m http.server</code>) rather than opening the file directly.</p>`;
    return;
  }

  // Headline numbers
  const pct = data.overall_coverage_pct ?? 0;
  document.querySelectorAll("[data-coverage-pct]").forEach((n) => { n.textContent = `~${pct}%`; });
  renderRing(pct);

  // Counts in hero meta
  const counts = data.systems.reduce((a, s) => { a[s.status] = (a[s.status] || 0) + 1; return a; }, {});
  setText("stat-covered", counts.covered || 0);
  setText("stat-partial", counts.partial || 0);
  setText("stat-total", data.systems.length);

  // Brain + systems
  const brainMount = document.getElementById("brain-mount");
  const brain = renderBrain(brainMount, data.systems, {
    onRegionEnter: (region, status, ev) => {
      // highlight all system cards hosted by this region
      region.systems.forEach((sid) => highlightSystemCard(sid, true));
      const host = data.systems.find((s) => region.systems.includes(s.id));
      if (host) showTooltip(tooltip, host, ev);
    },
    onRegionLeave: (region) => {
      region.systems.forEach((sid) => highlightSystemCard(sid, false));
      hideTooltip(tooltip);
    },
  });
  renderSystems(data.systems, brain, tooltip);
}

function setText(id, v) { const el = document.getElementById(id); if (el) el.textContent = v; }

main();
