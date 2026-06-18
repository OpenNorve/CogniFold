// brain.js — the interactive "Memory Brain".
// A hand-drawn (rough.js) anatomical brain rendered in SVG, with a crayon-
// textured neuron network overlaid. Regions map to memory systems from the
// data file; firing/pulse animation runs on interaction and as an idle
// ambient shimmer. Honors prefers-reduced-motion.
//
// The rough.js instance is passed in from app.js (vendored locally) rather than
// imported here, so a missing rough.js degrades gracefully instead of crashing
// the whole module graph — app.js falls back to a static panel in that case.

const SVG_NS = "http://www.w3.org/2000/svg";
const W = 760;
const H = 560;

const PALETTE = {
  covered: "#8a6a2c",
  partial: "#8a7d52",
  planned: "#9a8c74",
  ink: "#2b2118",
  sanguine: "#9c4a2f",
  ochre: "#b9842f",
  indigo: "#3f4f6b",
};

const reduceMotion =
  window.matchMedia &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// --- Region geometry (lateral view, hand-tuned) -------------------------
// Each region: closed polygon (for rough fill), a label anchor, and the
// memory systems it hosts.
const REGIONS = [
  {
    id: "prefrontal",
    label: "prefrontal cortex",
    systems: ["working", "prospective"],
    labelAt: [150, 150],
    poly: [[120, 250], [110, 190], [140, 150], [210, 140], [250, 175], [235, 235], [185, 270]],
  },
  {
    id: "neocortex",
    label: "neocortex",
    systems: ["semantic", "priming"],
    labelAt: [430, 95],
    poly: [[235, 150], [300, 110], [400, 95], [500, 110], [560, 160], [520, 215], [420, 215], [320, 210], [250, 190]],
  },
  {
    id: "hippocampus",
    label: "hippocampus",
    systems: ["episodic", "temporal"],
    labelAt: [355, 365],
    poly: [[300, 320], [360, 300], [420, 315], [445, 345], [410, 372], [345, 372], [305, 350]],
  },
  {
    id: "basal",
    label: "basal ganglia",
    systems: ["procedural"],
    labelAt: [470, 300],
    poly: [[440, 270], [500, 260], [545, 285], [535, 325], [480, 330], [450, 305]],
  },
  {
    id: "amygdala",
    label: "amygdala",
    systems: ["affective", "conditioning"],
    labelAt: [275, 395],
    poly: [[250, 360], [295, 350], [320, 375], [300, 405], [255, 400], [240, 380]],
  },
  {
    id: "cerebellum",
    label: "cerebellum",
    systems: ["procedural"],
    labelAt: [600, 380],
    poly: [[540, 320], [620, 310], [675, 350], [675, 420], [610, 450], [545, 420], [525, 365]],
  },
  {
    id: "sensory",
    label: "sensory cortices",
    systems: ["sensory"],
    labelAt: [555, 130],
    poly: [[505, 115], [575, 130], [600, 170], [560, 200], [515, 185], [510, 145]],
  },
];

// Annotated nodes (not anatomical regions): consolidation & forgetting.
// Rendered as small margin annotations with sketchy leader lines.
const ANNOT_NODES = [
  { id: "consolidation", at: [320, 470], leadTo: [400, 360], label: "consolidation" },
  { id: "forgetting", at: [600, 480], leadTo: [560, 400], label: "forgetting" },
];

// Brain outer hull (the iconic side-profile cortex) for the inked outline.
const HULL = [
  [110, 250], [105, 185], [150, 130], [240, 100], [360, 86],
  [480, 92], [575, 120], [630, 165], [660, 215],
  [675, 345], [678, 420], [620, 458], [545, 430],
  [470, 445], [400, 455], [320, 470], [255, 440],
  [220, 405], [180, 380], [150, 340], [122, 300],
];

// --- helpers ------------------------------------------------------------
function el(tag, attrs = {}) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function aggregateStatus(systemIds, systemsById) {
  // covered if any covered, else partial if any partial, else planned.
  let best = "planned";
  for (const id of systemIds) {
    const s = systemsById.get(id)?.status;
    if (s === "covered") return "covered";
    if (s === "partial") best = "partial";
  }
  return best;
}

function centroid(poly) {
  const n = poly.length;
  let x = 0, y = 0;
  for (const [px, py] of poly) { x += px; y += py; }
  return [x / n, y / n];
}

// Build a small neuron (soma + dendrites + axon) at a point, hand-drawn.
function makeNeuron(rc, cx, cy, color, scale = 1) {
  const g = el("g", { class: "neuron" });
  const r = 9 * scale;
  // soma
  g.appendChild(rc.circle(cx, cy, r * 2, {
    stroke: color, strokeWidth: 1.4, roughness: 2.1, bowing: 1.8,
    fill: color, fillStyle: "hachure", fillWeight: 0.7, hachureGap: 3,
  }));
  // dendrites (radiating wobbly lines)
  const dend = 5;
  for (let i = 0; i < dend; i++) {
    const a = (i / dend) * Math.PI * 2 + 0.4;
    const len = (16 + Math.random() * 10) * scale;
    g.appendChild(rc.line(cx, cy, cx + Math.cos(a) * len, cy + Math.sin(a) * len, {
      stroke: color, strokeWidth: 1, roughness: 2.4, bowing: 2.5,
    }));
  }
  return g;
}

// --- main render --------------------------------------------------------
export function renderBrain(host, data, onSelect, rough) {
  const systemsById = new Map(data.systems.map((s) => [s.id, s]));
  host.innerHTML = "";

  const svg = el("svg", {
    viewBox: `0 0 ${W} ${H}`,
    role: "group",
    "aria-label": "Hand-drawn anatomical brain. Tab through regions to inspect memory systems.",
  });
  host.appendChild(svg);

  const rc = rough.svg(svg);

  // layers
  const lNet = el("g", { class: "net-layer", filter: "url(#crayon-soft)" });
  const lHull = el("g", { class: "hull-layer", filter: "url(#crayon)" });
  const lRegions = el("g", { class: "regions-layer" });
  const lAnnot = el("g", { class: "annot-layer" });
  svg.append(lNet, lHull, lRegions, lAnnot);

  // ---- inked brain hull ----
  const hull = rc.polygon(
    HULL,
    { stroke: PALETTE.ink, strokeWidth: 2.4, roughness: 1.8, bowing: 1.4,
      fill: "rgba(176,138,62,0.05)", fillStyle: "solid" }
  );
  lHull.appendChild(hull);
  // a couple of inner gyri folds (sketchy)
  lHull.appendChild(rc.curve(
    [[160, 220], [240, 180], [330, 205], [410, 175], [500, 200], [580, 180]],
    { stroke: PALETTE.ink, strokeWidth: 1, roughness: 2.4, bowing: 2.2 }
  ));
  lHull.appendChild(rc.curve(
    [[150, 300], [250, 280], [350, 300], [450, 280], [560, 300]],
    { stroke: PALETTE.ink, strokeWidth: 1, roughness: 2.4, bowing: 2.2 }
  ));
  // brain stem
  lHull.appendChild(rc.curve(
    [[330, 470], [330, 510], [360, 540]],
    { stroke: PALETTE.ink, strokeWidth: 2, roughness: 1.6, bowing: 1.5 }
  ));

  // ---- neuron network overlay ----
  const realRegions = REGIONS.filter((r) => r.poly);
  const neuronPts = [];
  realRegions.forEach((reg) => {
    const [cx, cy] = centroid(reg.poly);
    const status = aggregateStatus(reg.systems, systemsById);
    neuronPts.push({ x: cx, y: cy, status, color: PALETTE[status], regionId: reg.id });
  });
  // a few extra free-floating neurons for density
  const filler = [[210, 200], [380, 150], [470, 180], [300, 250], [430, 250], [500, 360], [250, 320]];
  filler.forEach(([x, y], i) => {
    neuronPts.push({ x, y, status: "covered", color: PALETTE.covered, regionId: null, scale: 0.6 });
  });

  // synaptic connections (curved, crayon) — connect nearby neurons
  const connections = [];
  for (let i = 0; i < neuronPts.length; i++) {
    for (let j = i + 1; j < neuronPts.length; j++) {
      const a = neuronPts[i], b = neuronPts[j];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d < 150 && Math.random() < 0.55) {
        const status =
          a.status === "planned" || b.status === "planned"
            ? "planned"
            : a.status === "partial" || b.status === "partial"
            ? "partial"
            : "covered";
        const dash = status === "planned" ? "2 7" : status === "partial" ? "6 6" : null;
        const path = el("path", {
          d: `M${a.x} ${a.y} Q${(a.x + b.x) / 2 + (Math.random() * 30 - 15)} ${(a.y + b.y) / 2 + (Math.random() * 30 - 15)} ${b.x} ${b.y}`,
          fill: "none",
          stroke: PALETTE[status],
          "stroke-width": status === "covered" ? 1.4 : 1,
          "stroke-linecap": "round",
          opacity: status === "planned" ? 0.4 : 0.7,
          class: "synapse",
        });
        if (dash) path.setAttribute("stroke-dasharray", dash);
        lNet.appendChild(path);
        connections.push({ path, a: i, b: j, status });
      }
    }
  }

  // neurons
  const neuronEls = neuronPts.map((p) => {
    const g = makeNeuron(rc, p.x, p.y, p.color, p.scale || 1);
    g.dataset.region = p.regionId || "";
    g.dataset.status = p.status;
    if (p.status === "planned") g.style.opacity = "0.45";
    lNet.appendChild(g);
    return g;
  });

  // ---- interactive regions ----
  const regionEls = new Map();
  realRegions.forEach((reg) => {
    const status = aggregateStatus(reg.systems, systemsById);
    const color = PALETTE[status];
    const g = el("g", {
      class: "region",
      tabindex: "0",
      role: "button",
      "aria-label": `${reg.label}, ${status}. Activate to read field notes.`,
    });
    g.dataset.region = reg.id;

    // hand-drawn region fill
    const fillStyle = status === "covered" ? "hachure" : status === "partial" ? "cross-hatch" : "solid";
    const opts = {
      stroke: color,
      strokeWidth: status === "planned" ? 1.1 : 2,
      roughness: status === "planned" ? 2.8 : 1.9,
      bowing: 2,
      fill: status === "planned" ? "transparent" : color,
      fillStyle,
      fillWeight: status === "covered" ? 1.4 : 0.9,
      hachureGap: status === "covered" ? 5 : 7,
    };
    if (status === "planned") opts.strokeLineDash = [3, 5];
    g.appendChild(rc.polygon(reg.poly, opts));

    // transparent hit area on top for robust pointer/focus
    g.appendChild(el("polygon", {
      class: "region__hit",
      points: reg.poly.map((p) => p.join(",")).join(" "),
    }));

    // label
    const [lx, ly] = reg.labelAt;
    const label = el("text", { class: "region__label", x: lx, y: ly, "text-anchor": "middle" });
    label.textContent = reg.label;
    g.appendChild(label);

    regionEls.set(reg.id, { g, status, systems: reg.systems });
    lRegions.appendChild(g);

    const fire = () => fireRegion(reg.id);
    const select = () => {
      onSelect(reg.systems[0], reg.id);
      fireRegion(reg.id);
    };
    g.addEventListener("mouseenter", () => { highlight(reg.id); onSelect(reg.systems[0], reg.id, true); });
    g.addEventListener("mouseleave", () => highlight(null));
    g.addEventListener("focus", () => { highlight(reg.id); onSelect(reg.systems[0], reg.id, true); });
    g.addEventListener("click", select);
    g.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(); }
    });
  });

  // ---- annotation nodes (consolidation / forgetting) ----
  ANNOT_NODES.forEach((n) => {
    const sys = systemsById.get(n.id);
    const status = sys?.status || "partial";
    const color = PALETTE[status];
    // leader line
    lAnnot.appendChild(rc.line(n.at[0], n.at[1], n.leadTo[0], n.leadTo[1], {
      stroke: PALETTE.sanguine, strokeWidth: 0.9, roughness: 2.6, bowing: 3,
    }));
    // little arrowhead
    lAnnot.appendChild(el("circle", { cx: n.leadTo[0], cy: n.leadTo[1], r: 2.5, fill: PALETTE.sanguine }));
    const t = el("text", {
      class: "region__label", x: n.at[0], y: n.at[1], "text-anchor": "middle",
      fill: PALETTE.sanguine, "font-size": "13",
    });
    t.textContent = n.label;
    t.style.cursor = "pointer";
    t.setAttribute("tabindex", "0");
    t.setAttribute("role", "button");
    t.setAttribute("aria-label", `${n.label}, ${status}. Activate to read field notes.`);
    const pick = () => onSelect(n.id, null);
    t.addEventListener("click", pick);
    t.addEventListener("mouseenter", () => onSelect(n.id, null, true));
    t.addEventListener("focus", () => onSelect(n.id, null, true));
    t.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
    });
    lAnnot.appendChild(t);
  });

  // ---- highlight / dim logic ----
  function highlight(activeId) {
    regionEls.forEach(({ g }, id) => {
      g.classList.toggle("is-active", id === activeId);
      g.classList.toggle("is-dim", activeId !== null && id !== activeId);
    });
  }

  // ---- firing animation: pulse travels along connections from a region ----
  function fireRegion(regionId) {
    if (reduceMotion) return;
    // light the region's neuron + propagate along touching synapses
    neuronEls.forEach((g) => {
      if (g.dataset.region === regionId) pulseNeuron(g);
    });
    connections.forEach((c) => {
      const an = neuronPts[c.a], bn = neuronPts[c.b];
      if (an.regionId === regionId || bn.regionId === regionId) {
        pulseSynapse(c.path);
      }
    });
  }

  function pulseNeuron(g) {
    g.setAttribute("filter", "url(#ink-glow)");
    g.animate(
      [{ opacity: 1 }, { opacity: 0.55 }, { opacity: 1 }],
      { duration: 700, easing: "ease-in-out" }
    ).onfinish = () => g.removeAttribute("filter");
  }

  function pulseSynapse(path) {
    const len = path.getTotalLength ? path.getTotalLength() : 100;
    path.style.strokeDasharray = `${len}`;
    const anim = path.animate(
      [
        { strokeDashoffset: len, stroke: PALETTE.ochre, strokeWidth: 2.6, opacity: 1 },
        { strokeDashoffset: 0, stroke: PALETTE.ochre, strokeWidth: 2.6, opacity: 1 },
      ],
      { duration: 650, easing: "cubic-bezier(0.4,0,0.2,1)" }
    );
    anim.onfinish = () => {
      // restore original dash style
      const orig =
        path.dataset.dash || (path.getAttribute("stroke-dasharray") || "");
      path.style.strokeDasharray = "";
      path.style.strokeDashoffset = "";
      path.style.stroke = "";
      path.style.strokeWidth = "";
      path.style.opacity = "";
    };
  }

  // ---- idle ambient "thinking" shimmer ----
  let idleTimer = null;
  if (!reduceMotion) {
    const tick = () => {
      // fire one random covered/partial neuron softly
      const live = neuronEls.filter((g) => g.dataset.status !== "planned");
      if (live.length) {
        const g = live[Math.floor(Math.random() * live.length)];
        pulseNeuron(g);
        // fire a couple of its connections
        const reg = g.dataset.region;
        if (reg) {
          connections
            .filter((c) => neuronPts[c.a].regionId === reg || neuronPts[c.b].regionId === reg)
            .slice(0, 2)
            .forEach((c) => pulseSynapse(c.path));
        }
      }
      idleTimer = setTimeout(tick, 1400 + Math.random() * 1600);
    };
    idleTimer = setTimeout(tick, 1200);
  }

  // public API for app.js (e.g. system-index hover → light region)
  return {
    focusSystem(systemId) {
      // find region hosting this system
      let target = null;
      for (const [rid, info] of regionEls) {
        if (info.systems.includes(systemId)) { target = rid; break; }
      }
      highlight(target);
      if (target) fireRegion(target);
    },
    clearHighlight() { highlight(null); },
    destroy() { if (idleTimer) clearTimeout(idleTimer); },
  };
}
