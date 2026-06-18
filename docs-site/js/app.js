// app.js — data-driven entry point for the CogniFold site.
// Loads memory_coverage.json (single source of truth) and wires up: nav,
// scroll-reveal, the coverage ring, the interactive brain, the notebook
// cards + system index, the CLS architecture diagram, and the results ledger.
// rough.js is vendored locally and loaded with a dynamic import wrapped in
// try/catch, so a missing/broken rough.js degrades gracefully (static fallback)
// instead of white-screening the page. No build step.
import { renderBrain } from "./brain.js";

const SVG_NS = "http://www.w3.org/2000/svg";
// System font stacks for in-SVG text (no web fonts).
const SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
const MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
const reduceMotion =
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Real benchmark numbers only (no invented competitor figures, no LongMemEval).
const BENCHMARKS = [
  { bench: "CogEval", name: "Proactivity", val: "0.614", unit: "", note: "intent crystallization" },
  { bench: "BABILong", name: "long-context QA", val: "96.0", unit: "% EM" },
  { bench: "SafetyBench", name: "safety", val: "94.3", unit: "%" },
  { bench: "MuTual", name: "dialogue reasoning", val: "93.2", unit: "%" },
  { bench: "ToMi", name: "theory of mind", val: "91.6", unit: "% EM" },
  { bench: "LoCoMo", name: "long conversation", val: "82.8", unit: " J-Score", note: "episodic recall" },
  { bench: "StreamingQA", name: "streaming QA", val: "78.4", unit: "% EM" },
  { bench: "NarrativeQA", name: "narrative", val: "0.720", unit: " F1" },
  { bench: "MuSiQue", name: "multi-hop", val: "41.2", unit: "% EM", note: "F1 0.587" },
];

const STATUS_WORD = { covered: "inked", partial: "half-inked", planned: "pencilled" };

const el = (tag, attrs = {}) => {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};

document.addEventListener("DOMContentLoaded", init);

async function init() {
  setupNav();
  setupReveal();

  // --- load data (single source of truth) ---
  let data;
  try {
    const res = await fetch("data/memory_coverage.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    data = await res.json();
    if (!data || !Array.isArray(data.systems) || !data.systems.length) {
      throw new Error("malformed memory_coverage.json");
    }
  } catch (err) {
    console.error("Could not load memory_coverage.json — showing static fallback.", err);
    renderHardFallback();
    return;
  }

  const systemsById = new Map(data.systems.map((s) => [s.id, s]));

  // --- load rough.js (vendored). If it fails, we still render everything
  //     except the hand-drawn art, and fall back to a static brain panel. ---
  let rough = null;
  try {
    const mod = await import("./vendor/rough.esm.js");
    rough = mod.default || mod;
  } catch (err) {
    console.warn("rough.js unavailable — using static fallbacks for hand-drawn art.", err);
  }

  renderCoverageRing(data.overall_coverage_pct, rough);
  renderArchDiagram(rough);
  renderLedger();
  renderSystemIndex(data, onSelectSystem);
  showNotecard(null); // default state

  const host = document.getElementById("brainHost");
  let brain = null;
  if (rough) {
    try {
      brain = renderBrain(host, data, (systemId, regionId, isHover) => {
        onSelectSystem(systemId, !isHover);
      }, rough);
    } catch (err) {
      console.error("Brain render failed — using static fallback.", err);
      brain = null;
    }
  }
  if (!brain) renderBrainFallback(host, data);

  // wire system-index rows back to the brain (no-op guards if brain absent)
  document.querySelectorAll(".sys-row").forEach((row) => {
    const id = row.dataset.system;
    row.addEventListener("mouseenter", () => { brain && brain.focusSystem(id); onSelectSystem(id, false); });
    row.addEventListener("mouseleave", () => brain && brain.clearHighlight());
    row.addEventListener("focus", () => { brain && brain.focusSystem(id); onSelectSystem(id, false); });
    row.addEventListener("click", () => onSelectSystem(id, true));
  });

  function onSelectSystem(systemId, markActive) {
    const sys = systemsById.get(systemId);
    if (!sys) return;
    showNotecard(sys);
    document.querySelectorAll(".sys-row").forEach((r) =>
      r.classList.toggle("is-active", r.dataset.system === systemId)
    );
  }
}

// -------------------------------------------------------------- nav
function setupNav() {
  const nav = document.getElementById("nav");
  const menu = document.getElementById("navMenu");
  const links = document.getElementById("navLinks");

  const onScroll = () => nav.classList.toggle("is-scrolled", window.scrollY > 20);
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });

  menu.addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    menu.setAttribute("aria-expanded", String(open));
  });
  links.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => {
      nav.classList.remove("is-open");
      menu.setAttribute("aria-expanded", "false");
    })
  );
}

// -------------------------------------------------------------- scroll reveal
function setupReveal() {
  const els = document.querySelectorAll(".reveal");
  if (reduceMotion || !("IntersectionObserver" in window)) {
    els.forEach((e) => e.classList.add("is-in"));
    return;
  }
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-in");
          io.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
  );
  els.forEach((e) => io.observe(e));
}

// -------------------------------------------------------------- coverage ring
function renderCoverageRing(pct, rough) {
  const host = document.getElementById("heroRing");
  host.classList.add("cov-ring");
  const size = 320, r = 128, cx = size / 2, cy = size / 2;
  const circ = 2 * Math.PI * r;

  const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, "aria-hidden": "true" });

  // outer vitruvian guide ring (hand-drawn when rough.js is present)
  if (rough) {
    const rc = rough.svg(svg);
    svg.appendChild(rc.circle(cx, cy, r * 2 + 34, {
      stroke: "#9c4a2f", strokeWidth: 0.8, roughness: 2.4, bowing: 1.2, fill: "none",
    }));
  } else {
    svg.appendChild(el("circle", {
      cx, cy, r: r + 17, fill: "none", stroke: "#9c4a2f", "stroke-width": 0.8, opacity: 0.6,
    }));
  }

  // track
  const track = el("circle", {
    cx, cy, r, fill: "none", stroke: "rgba(43,33,24,0.14)", "stroke-width": 14,
  });
  svg.appendChild(track);

  // crayon-textured progress arc
  const prog = el("circle", {
    cx, cy, r, fill: "none",
    stroke: "#8a6a2c", "stroke-width": 14, "stroke-linecap": "round",
    transform: `rotate(-90 ${cx} ${cy})`,
    "stroke-dasharray": circ,
    "stroke-dashoffset": circ,
    filter: "url(#crayon)",
  });
  svg.appendChild(prog);

  host.appendChild(svg);

  const num = document.createElement("div");
  num.className = "cov-ring__num";
  num.innerHTML = `<span class="cov-ring__pct">~${pct}<span style="font-size:0.5em">%</span></span><span class="cov-ring__cap">of human memory systems modeled</span>`;
  host.appendChild(num);

  const target = circ * (1 - pct / 100);
  if (reduceMotion) {
    prog.setAttribute("stroke-dashoffset", target);
  } else {
    // animate when scrolled into view
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          prog.animate(
            [{ strokeDashoffset: circ }, { strokeDashoffset: target }],
            { duration: 1600, easing: "cubic-bezier(0.22,1,0.36,1)", fill: "forwards" }
          );
          io.disconnect();
        }
      });
    }, { threshold: 0.4 });
    io.observe(host);
  }
}

// -------------------------------------------------------------- notebook card
function showNotecard(sys) {
  const card = document.getElementById("notecard");
  if (!sys) {
    card.innerHTML = `
      <div class="notecard__head">
        <h3 class="notecard__name">Field Notes</h3>
      </div>
      <p class="notecard__desc">Twelve human memory systems, mapped to the CogniFold topology.</p>
      <p class="notecard__hint">Hover or tab a brain region to open its notes &rarr;</p>`;
    return;
  }
  const evidenceParts = sys.evidence.split(";").map((s) => s.trim());
  card.innerHTML = `
    <div class="notecard__head">
      <h3 class="notecard__name">${escapeHtml(sys.name)}</h3>
      <span class="notecard__status" data-status="${sys.status}">${STATUS_WORD[sys.status]}</span>
    </div>
    <div class="notecard__tags">
      <span class="notecard__tag">${escapeHtml(sys.taxonomy_group)}</span>
      <span class="notecard__tag">${escapeHtml(sys.brain_region)}</span>
      <span class="notecard__tag">w ${sys.weight}</span>
    </div>
    <p class="notecard__desc">${escapeHtml(sys.description)}</p>
    <div class="notecard__row"><b>CogniFold mechanism</b>${escapeHtml(sys.cognifold_mechanism)}</div>
    <div class="notecard__row"><b>Evidence</b>${evidenceParts.map((p) => `<code>${escapeHtml(p)}</code>`).join(" ")}</div>`;
}

// -------------------------------------------------------------- system index
function renderSystemIndex(data, onSelect) {
  const host = document.getElementById("systemIndex");
  host.innerHTML = "";
  data.systems.forEach((s) => {
    const row = document.createElement("button");
    row.className = "sys-row";
    row.type = "button";
    row.dataset.system = s.id;
    row.dataset.status = s.status;
    row.setAttribute("role", "listitem");
    row.setAttribute("aria-label", `${s.name}, ${s.status}, ${s.brain_region}`);
    row.innerHTML = `
      <span class="sys-row__dot" aria-hidden="true"></span>
      <span class="sys-row__name">${escapeHtml(s.name)}</span>
      <span class="sys-row__region">${escapeHtml(shortRegion(s.brain_region))}</span>`;
    host.appendChild(row);
  });
}

function shortRegion(region) {
  return region.split(/[\/(]/)[0].trim().split(" ")[0];
}

// A minimal stand-in for rough.svg() that draws plain (straight) SVG shapes.
// Used when rough.js is unavailable so the architecture diagram still renders.
function plainRc() {
  const dash = (o) => (o && o.strokeLineDash ? o.strokeLineDash.join(" ") : null);
  const stroke = (o = {}) => ({
    fill: o.fill && o.fill !== "none" && !o.fillStyle ? o.fill : "none",
    stroke: o.stroke || "#2b2118",
    "stroke-width": o.strokeWidth || 1,
    "fill-opacity": o.fillStyle ? 0.18 : (o.fill && o.fill !== "none" ? 1 : 0),
  });
  return {
    line(x1, y1, x2, y2, o) {
      const n = el("line", { x1, y1, x2, y2, ...stroke(o) });
      const d = dash(o); if (d) n.setAttribute("stroke-dasharray", d);
      return n;
    },
    circle(cx, cy, diam, o) {
      const n = el("circle", { cx, cy, r: diam / 2, ...stroke(o) });
      if (o && o.fillStyle && o.fill) { n.setAttribute("fill", o.fill); }
      return n;
    },
    curve(pts, o) {
      const d = pts.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(" ");
      return el("path", { d, ...stroke(o), fill: "none" });
    },
  };
}

// -------------------------------------------------------------- arch diagram
function renderArchDiagram(rough) {
  const host = document.getElementById("archDiagram");
  host.innerHTML = "";
  const W = 1000, H = 360;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "The folding loop: EVENT folds into CONCEPT, which crystallizes into INTENT, all threaded by TIME." });
  const rc = rough ? rough.svg(svg) : plainRc();

  const nodes = [
    { x: 175, y: 170, label: "EVENT", sub: "hippocampal", color: "#9c4a2f" },
    { x: 500, y: 170, label: "CONCEPT", sub: "neocortical", color: "#b9842f" },
    { x: 825, y: 170, label: "INTENT", sub: "prefrontal", color: "#3f4f6b" },
  ];

  // arrows between nodes
  const arrow = (x1, y, x2, label) => {
    const g = el("g", {});
    g.appendChild(rc.line(x1, y, x2, y, { stroke: "#1d1d1f", strokeWidth: 1.6, roughness: 1, bowing: 0.8 }));
    g.appendChild(rc.line(x2, y, x2 - 13, y - 7, { stroke: "#1d1d1f", strokeWidth: 1.6, roughness: 0.8 }));
    g.appendChild(rc.line(x2, y, x2 - 13, y + 7, { stroke: "#1d1d1f", strokeWidth: 1.6, roughness: 0.8 }));
    const t = el("text", { x: (x1 + x2) / 2, y: y - 16, "text-anchor": "middle",
      "font-family": SANS, "font-size": "15", "font-weight": "500", fill: "#515154" });
    t.textContent = label;
    g.appendChild(t);
    return g;
  };
  svg.appendChild(arrow(255, 170, 415, "fold"));
  svg.appendChild(arrow(580, 170, 745, "crystallize"));

  // feedback arc back (CONCEPT informs new EVENTs)
  svg.appendChild(rc.curve(
    [[825, 235], [650, 320], [400, 320], [175, 235]],
    { stroke: "#86868b", strokeWidth: 1.4, roughness: 1, bowing: 1 }
  ));
  const fb = el("text", { x: 500, y: 340, "text-anchor": "middle",
    "font-family": SANS, "font-size": "14", fill: "#86868b" });
  fb.textContent = "intent reshapes attention — the loop closes";
  svg.appendChild(fb);

  // nodes
  nodes.forEach((n, i) => {
    const g = el("g", {});
    g.appendChild(rc.circle(n.x, n.y, 130, {
      stroke: n.color, strokeWidth: 2, roughness: 0.9, bowing: 0.8,
      fill: n.color, fillStyle: "hachure", fillWeight: 0.8, hachureGap: 9,
    }));
    const t1 = el("text", { x: n.x, y: n.y - 1, "text-anchor": "middle",
      "font-family": SANS, "font-weight": "700", "font-size": "24", fill: "#1d1d1f" });
    t1.textContent = n.label;
    const t2 = el("text", { x: n.x, y: n.y + 20, "text-anchor": "middle",
      "font-family": MONO, "font-size": "10", fill: "#515154",
      "letter-spacing": "0.12em" });
    t2.textContent = n.sub.toUpperCase();
    svg.append(g, t1, t2);
  });

  // TIME axis threading underneath
  const timeY = 40;
  svg.appendChild(rc.line(120, timeY, 880, timeY, {
    stroke: "#86868b", strokeWidth: 1, roughness: 1, bowing: 1, strokeLineDash: [2, 8],
  }));
  const tlabel = el("text", { x: 500, y: timeY - 12, "text-anchor": "middle",
    "font-family": MONO, "font-size": "11", "letter-spacing": "0.1em", fill: "#86868b" });
  tlabel.textContent = "TIME · the fourth axis";
  svg.appendChild(tlabel);
  // tick marks down to each node
  nodes.forEach((n) => {
    svg.appendChild(rc.line(n.x, timeY, n.x, 100, {
      stroke: "#3f4f6b", strokeWidth: 0.8, roughness: 2.4, strokeLineDash: [2, 6],
    }));
  });

  host.appendChild(svg);
}

// -------------------------------------------------------------- ledger
function renderLedger() {
  const grid = document.getElementById("ledgerGrid");
  grid.innerHTML = "";
  BENCHMARKS.forEach((b, i) => {
    const card = document.createElement("article");
    card.className = "bench-card reveal";
    card.innerHTML = `
      <div class="bench-card__bench">${escapeHtml(b.bench)}</div>
      <div class="bench-card__metric">
        <span class="bench-card__val">${escapeHtml(b.val)}</span>
        <span class="bench-card__unit">${escapeHtml(b.unit)}</span>
      </div>
      <div class="bench-card__name">${escapeHtml(b.name)}</div>
      ${b.note ? `<div class="bench-card__note">${escapeHtml(b.note)}</div>` : ""}`;
    grid.appendChild(card);
  });
  // re-observe the freshly added reveal cards
  setupReveal();
}

// -------------------------------------------------------------- fallbacks
// Static brain panel: shown when rough.js is missing or the brain throws.
// Renders the systems as a tasteful inline list inside the brain stage so the
// section is never blank.
function renderBrainFallback(host, data) {
  host.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "brain-fallback";
  wrap.setAttribute("role", "list");
  wrap.setAttribute("aria-label", "Memory systems mapped to brain regions");
  wrap.innerHTML =
    `<p class="brain-fallback__note">Interactive diagram unavailable. Memory systems:</p>` +
    data.systems
      .map(
        (s) => `<div class="brain-fallback__row" data-status="${s.status}" role="listitem">
          <span class="sys-row__dot" aria-hidden="true"></span>
          <span class="brain-fallback__name">${escapeHtml(s.name)}</span>
          <span class="brain-fallback__region">${escapeHtml(s.brain_region)}</span>
          <span class="brain-fallback__status" data-status="${s.status}">${escapeHtml(s.status)}</span>
        </div>`
      )
      .join("");
  host.appendChild(wrap);
}

// Hard fallback: data itself could not be loaded. Replace the whole brain
// section body with the coverage number + a short note so it is never blank.
function renderHardFallback() {
  const ring = document.getElementById("heroRing");
  if (ring) {
    ring.classList.add("cov-ring");
    ring.innerHTML =
      `<div class="cov-ring__num"><span class="cov-ring__pct">~60<span style="font-size:0.5em">%</span></span>` +
      `<span class="cov-ring__cap">of human memory systems modeled</span></div>`;
  }
  const host = document.getElementById("brainHost");
  if (host) {
    host.innerHTML =
      `<div class="brain-fallback"><p class="brain-fallback__note">` +
      `Coverage data could not be loaded. CogniFold models roughly 60% of ` +
      `human memory systems — see the architecture and results below.</p></div>`;
  }
}

// -------------------------------------------------------------- util
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
