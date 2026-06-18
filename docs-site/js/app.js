// app.js — data-driven entry point for the CogniFold site.
// Loads memory_coverage.json (single source of truth) and wires up: nav,
// scroll-reveal, the coverage ring, the interactive brain, the notebook
// cards + system index, the CLS architecture diagram, and the results ledger.
// All visuals are native SVG (no external libraries, no build step). A
// data-load failure degrades gracefully to a static panel.
import { renderBrain } from "./brain.js";

const SVG_NS = "http://www.w3.org/2000/svg";
// System font stacks for in-SVG text (no web fonts).
const SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
const MONO = "ui-monospace, 'SF Mono', Menlo, Consolas, monospace";
const reduceMotion =
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Real benchmark numbers only (no invented competitor figures, no LongMemEval).
// `note` is a concise, factual context line describing what the benchmark probes.
const BENCHMARKS = [
  { bench: "CogEval", name: "Proactivity", val: "0.614", unit: "", note: "Tests whether the agent acts on intents at the right moment — CogniFold's core differentiator." },
  { bench: "BABILong", name: "Long-context QA", val: "96.0", unit: "% EM", note: "Reasoning over facts buried in very long contexts; exact-match accuracy." },
  { bench: "SafetyBench", name: "Safety", val: "94.3", unit: "%", note: "Multiple-choice safety judgments across harm categories." },
  { bench: "MuTual", name: "Dialogue reasoning", val: "93.2", unit: "%", note: "Multi-turn dialogue inference — choosing the coherent next response." },
  { bench: "ToMi", name: "Theory of mind", val: "91.6", unit: "% EM", note: "Tracking other agents' beliefs and false beliefs across a narrative." },
  { bench: "LoCoMo", name: "Long conversation", val: "82.8", unit: " J-Score", note: "Episodic recall over very long multi-session conversations." },
  { bench: "StreamingQA", name: "Streaming QA", val: "78.4", unit: "% EM", note: "Answering questions over a time-ordered stream of incoming documents." },
  { bench: "NarrativeQA", name: "Narrative", val: "0.720", unit: " F1", note: "Comprehension questions over full stories and scripts." },
  { bench: "MuSiQue", name: "Multi-hop", val: "41.2", unit: "% EM", note: "Compositional multi-hop questions; also F1 0.587." },
];

const STATUS_WORD = { covered: "covered", partial: "partial", planned: "planned" };

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

  // All visuals are native SVG (no external libraries), so they always render.
  renderCoverageRing(data.overall_coverage_pct);
  renderArchDiagram();
  renderLedger();
  renderSystemsTable(data, onSelectSystem);
  showNotecard(null); // default state

  const host = document.getElementById("brainHost");
  let brain = null;
  try {
    brain = renderBrain(host, data, (systemId, regionId, isHover) => {
      onSelectSystem(systemId, !isHover);
    });
  } catch (err) {
    console.error("Brain render failed — using static fallback.", err);
    brain = null;
  }
  if (!brain) renderBrainFallback(host, data);

  function onSelectSystem(systemId, markActive) {
    const sys = systemsById.get(systemId);
    if (!sys) return;
    showNotecard(sys);
    document.querySelectorAll(".systbl tbody tr").forEach((r) =>
      r.classList.toggle("is-active", r.dataset.system === systemId)
    );
    if (brain) brain.focusSystem(systemId);
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
// Clean neural-ink progress ring (native SVG, accent gradient stroke).
function renderCoverageRing(pct) {
  const host = document.getElementById("heroRing");
  host.classList.add("cov-ring");
  const size = 320, r = 128, cx = size / 2, cy = size / 2;
  const circ = 2 * Math.PI * r;

  const svg = el("svg", { viewBox: `0 0 ${size} ${size}`, "aria-hidden": "true" });

  // accent gradient for the progress arc
  const defs = el("defs");
  defs.innerHTML =
    `<linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1">` +
    `<stop offset="0%" stop-color="#0a84ff"/><stop offset="100%" stop-color="#5ac8fa"/></linearGradient>`;
  svg.appendChild(defs);

  // track
  svg.appendChild(el("circle", {
    cx, cy, r, fill: "none", stroke: "rgba(0,0,0,0.08)", "stroke-width": 12,
  }));

  // progress arc
  const prog = el("circle", {
    cx, cy, r, fill: "none",
    stroke: "url(#ringGrad)", "stroke-width": 12, "stroke-linecap": "round",
    transform: `rotate(-90 ${cx} ${cy})`,
    "stroke-dasharray": circ,
    "stroke-dashoffset": circ,
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

// -------------------------------------------------------------- systems table
// Full reference table: all 12 systems visible at once (name · taxonomy ·
// region · status · mechanism). Rows sync with the brain + side index on hover.
function renderSystemsTable(data, onSelect) {
  const body = document.getElementById("systemsTableBody");
  if (!body) return;
  body.innerHTML = "";
  data.systems.forEach((s) => {
    const tr = document.createElement("tr");
    tr.dataset.system = s.id;
    tr.dataset.status = s.status;
    tr.innerHTML = `
      <td class="systbl__name"><span class="systbl__sys"><span class="systbl__dot" aria-hidden="true"></span>${escapeHtml(s.name)}</span></td>
      <td class="systbl__tax">${escapeHtml(s.taxonomy_group)}</td>
      <td class="systbl__region">${escapeHtml(s.brain_region)}</td>
      <td><span class="systbl__status" data-status="${s.status}">${escapeHtml(s.status)}</span></td>
      <td class="systbl__mech">${escapeHtml(s.cognifold_mechanism)}</td>`;
    tr.addEventListener("mouseenter", () => onSelect(s.id, false));
    tr.addEventListener("click", () => onSelect(s.id, true));
    body.appendChild(tr);
  });
}

// -------------------------------------------------------------- arch diagram
// Clean neural-ink diagram, native SVG (no rough.js).
function renderArchDiagram() {
  const host = document.getElementById("archDiagram");
  host.innerHTML = "";
  const W = 1000, H = 360;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "The folding loop: EVENT folds into CONCEPT, which crystallizes into INTENT, all threaded by TIME." });

  const nodes = [
    { x: 175, y: 170, label: "EVENT", sub: "hippocampal", color: "#0a84ff" },
    { x: 500, y: 170, label: "CONCEPT", sub: "neocortical", color: "#2a9ad6" },
    { x: 825, y: 170, label: "INTENT", sub: "prefrontal", color: "#3f4f6b" },
  ];

  // arrows between nodes
  const arrow = (x1, y, x2, label) => {
    const g = el("g", {});
    g.appendChild(el("line", { x1, y1: y, x2, y2: y, stroke: "#1d1d1f", "stroke-width": 1.6 }));
    g.appendChild(el("line", { x1: x2, y1: y, x2: x2 - 13, y2: y - 7, stroke: "#1d1d1f", "stroke-width": 1.6, "stroke-linecap": "round" }));
    g.appendChild(el("line", { x1: x2, y1: y, x2: x2 - 13, y2: y + 7, stroke: "#1d1d1f", "stroke-width": 1.6, "stroke-linecap": "round" }));
    const t = el("text", { x: (x1 + x2) / 2, y: y - 16, "text-anchor": "middle",
      "font-family": SANS, "font-size": "15", "font-weight": "500", fill: "#515154" });
    t.textContent = label;
    g.appendChild(t);
    return g;
  };
  svg.appendChild(arrow(255, 170, 415, "fold"));
  svg.appendChild(arrow(580, 170, 745, "crystallize"));

  // feedback arc back (CONCEPT informs new EVENTs)
  svg.appendChild(el("path", {
    d: "M 825 235 C 700 315 300 315 175 235",
    fill: "none", stroke: "#c4c7cc", "stroke-width": 1.4, "stroke-dasharray": "2 6",
  }));
  const fb = el("text", { x: 500, y: 340, "text-anchor": "middle",
    "font-family": SANS, "font-size": "14", fill: "#86868b" });
  fb.textContent = "intent reshapes attention — the loop closes";
  svg.appendChild(fb);

  // nodes — soft accent fill + crisp ring
  nodes.forEach((n) => {
    const g = el("g", {});
    g.appendChild(el("circle", { cx: n.x, cy: n.y, r: 65, fill: n.color, "fill-opacity": 0.1, stroke: n.color, "stroke-width": 2 }));
    const t1 = el("text", { x: n.x, y: n.y - 1, "text-anchor": "middle",
      "font-family": SANS, "font-weight": "700", "font-size": "24", fill: "#1d1d1f" });
    t1.textContent = n.label;
    const t2 = el("text", { x: n.x, y: n.y + 20, "text-anchor": "middle",
      "font-family": MONO, "font-size": "10", fill: "#515154", "letter-spacing": "0.12em" });
    t2.textContent = n.sub.toUpperCase();
    svg.append(g, t1, t2);
  });

  // TIME axis threading underneath
  const timeY = 40;
  svg.appendChild(el("line", { x1: 120, y1: timeY, x2: 880, y2: timeY,
    stroke: "#86868b", "stroke-width": 1, "stroke-dasharray": "2 8" }));
  const tlabel = el("text", { x: 500, y: timeY - 12, "text-anchor": "middle",
    "font-family": MONO, "font-size": "11", "letter-spacing": "0.1em", fill: "#86868b" });
  tlabel.textContent = "TIME · the fourth axis";
  svg.appendChild(tlabel);
  nodes.forEach((n) => {
    svg.appendChild(el("line", { x1: n.x, y1: timeY, x2: n.x, y2: 100,
      stroke: "#c4c7cc", "stroke-width": 0.8, "stroke-dasharray": "2 6" }));
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
// Static brain panel: shown if the brain render throws, so it is never blank.
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
