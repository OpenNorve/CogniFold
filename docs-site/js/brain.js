// brain.js — renders a stylized anatomical brain (SVG) with labeled regions.
// Each region maps to one or more memory-system ids. Regions light up by the
// aggregated status of the systems they host, and expose hover highlighting.

// Region geometry: a lateral (side) view of the brain. Paths are hand-tuned
// blobs approximating the major lobes / structures. Coordinates live in a
// 0..600 x 0..420 viewBox.
export const REGIONS = [
  {
    id: "prefrontal",
    label: "Prefrontal cortex",
    systems: ["working", "prospective"],
    // front of brain (left side of viewBox)
    path: "M70 150 C60 110 95 70 150 64 C180 60 205 74 214 100 C200 130 196 165 192 196 C170 206 130 214 104 206 C80 198 76 176 70 150 Z",
    labelXY: [118, 138],
  },
  {
    id: "neocortex",
    label: "Neocortex",
    systems: ["semantic"],
    // top / parietal-occipital dome
    path: "M214 100 C232 60 300 44 372 56 C448 68 498 110 500 158 C470 168 420 170 372 168 C310 166 250 158 200 150 C202 132 208 114 214 100 Z",
    labelXY: [350, 100],
  },
  {
    id: "hippocampus",
    label: "Hippocampus",
    systems: ["episodic", "temporal", "consolidation"],
    // central medial-temporal seahorse region
    path: "M250 200 C270 188 320 188 348 200 C372 210 378 234 360 248 C330 262 282 262 256 248 C238 238 234 210 250 200 Z",
    labelXY: [305, 226],
  },
  {
    id: "temporal-lobe",
    label: "Temporal lobe",
    systems: ["sensory", "priming"],
    // lower-front lobe
    path: "M150 230 C140 268 168 300 220 306 C268 312 300 296 306 270 C282 264 256 256 238 246 C210 232 178 226 150 230 Z",
    labelXY: [212, 286],
  },
  {
    id: "amygdala",
    label: "Amygdala",
    systems: ["affective", "conditioning"],
    // small almond near hippocampus
    path: "M236 252 C232 240 252 234 266 240 C278 246 276 262 262 266 C248 270 240 264 236 252 Z",
    labelXY: [214, 264],
  },
  {
    id: "neocortex-back",
    label: "Cortical assoc.",
    systems: ["forgetting"],
    // occipital back lobe
    path: "M448 168 C492 168 522 192 520 226 C518 256 488 274 452 270 C424 266 408 244 410 218 C412 192 426 172 448 168 Z",
    labelXY: [468, 222],
  },
  {
    id: "cerebellum",
    label: "Cerebellum",
    systems: ["procedural"],
    // bottom-back ridged structure
    path: "M380 268 C420 262 470 276 480 308 C486 334 456 356 410 356 C368 356 340 338 338 312 C336 290 352 272 380 268 Z",
    labelXY: [418, 322],
  },
  {
    id: "basal-ganglia",
    label: "Basal ganglia",
    systems: ["procedural"],
    // deep central nucleus
    path: "M300 232 C322 226 350 234 356 252 C360 268 342 282 318 282 C296 282 282 268 284 252 C286 240 292 234 300 232 Z",
    labelXY: [322, 258],
  },
];

const STATUS_COLOR = { covered: "#38e8ff", partial: "#ffb347", planned: "#4a5380" };

// Aggregate a region's status from the systems it hosts.
function regionStatus(region, systemsById) {
  const statuses = region.systems
    .map((sid) => systemsById[sid]?.status)
    .filter(Boolean);
  if (statuses.some((s) => s === "covered")) return "covered";
  if (statuses.some((s) => s === "partial")) return "partial";
  return "planned";
}

const SVGNS = "http://www.w3.org/2000/svg";
function el(name, attrs = {}) {
  const e = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

/**
 * Render the brain into `mount`.
 * @returns {{ highlightBySystem: (sid:string|null)=>void, regionForSystem: (sid:string)=>string|undefined }}
 */
export function renderBrain(mount, systems, { onRegionEnter, onRegionLeave } = {}) {
  const systemsById = Object.fromEntries(systems.map((s) => [s.id, s]));

  const svg = el("svg", {
    viewBox: "0 0 600 420",
    role: "img",
    "aria-label": "Anatomical brain showing which memory systems CogniFold models",
  });

  // glow filter
  const defs = el("defs");
  const filter = el("filter", { id: "brainGlow", x: "-40%", y: "-40%", width: "180%", height: "180%" });
  const blur = el("feGaussianBlur", { in: "SourceGraphic", stdDeviation: "5", result: "b" });
  const merge = el("feMerge");
  merge.appendChild(el("feMergeNode", { in: "b" }));
  merge.appendChild(el("feMergeNode", { in: "SourceGraphic" }));
  filter.appendChild(blur);
  filter.appendChild(merge);
  defs.appendChild(filter);
  svg.appendChild(defs);

  // faint full-brain silhouette behind regions for cohesion
  const silhouette = el("path", {
    d: "M70 150 C55 95 110 52 180 50 C250 48 320 42 400 50 C480 58 528 104 522 160 C530 210 512 262 480 300 C470 345 420 366 360 360 C300 372 250 360 210 320 C160 318 130 300 120 270 C90 262 70 234 66 196 C62 178 64 164 70 150 Z",
    fill: "rgba(155,107,255,0.04)",
    stroke: "rgba(155,107,255,0.18)",
    "stroke-width": "1.5",
  });
  svg.appendChild(silhouette);

  // central connective threads (subtle neural net feel)
  const threads = [
    [118, 138, 305, 226], [305, 226, 350, 100], [305, 226, 322, 258],
    [322, 258, 418, 322], [214, 264, 305, 226], [468, 222, 350, 100],
    [118, 138, 212, 286],
  ];
  for (const [x1, y1, x2, y2] of threads) {
    svg.appendChild(el("line", {
      x1, y1, x2, y2, stroke: "rgba(56,232,255,0.10)", "stroke-width": "1",
    }));
  }

  const regionEls = {};
  const systemToRegion = {};

  for (const region of REGIONS) {
    const status = regionStatus(region, systemsById);
    const color = STATUS_COLOR[status];

    const g = el("g", { class: `region ${status === "planned" ? "dim" : "active"}` });
    g.style.color = color; // drives currentColor in CSS glow
    g.dataset.region = region.id;

    const lobe = el("path", {
      d: region.path,
      fill: status === "planned" ? "rgba(74,83,128,0.12)" : `${color}22`,
      stroke: color,
    });
    lobe.classList.add("lobe");

    // pulsing animation for partial regions
    if (status === "partial") {
      const anim = el("animate", {
        attributeName: "fill-opacity", values: "0.35;0.85;0.35",
        dur: "2.6s", repeatCount: "indefinite",
      });
      lobe.appendChild(anim);
    }

    const [lx, ly] = region.labelXY;
    const label = el("text", { x: lx, y: ly, "text-anchor": "middle" });
    label.classList.add("region-label");
    label.textContent = region.label;

    g.appendChild(lobe);
    g.appendChild(label);

    g.addEventListener("mouseenter", (ev) => onRegionEnter?.(region, status, ev));
    g.addEventListener("mouseleave", () => onRegionLeave?.(region));

    svg.appendChild(g);
    regionEls[region.id] = g;
    for (const sid of region.systems) systemToRegion[sid] = region.id;
  }

  mount.appendChild(svg);

  return {
    highlightBySystem(sid) {
      for (const g of Object.values(regionEls)) g.classList.remove("highlight");
      if (!sid) return;
      const rid = systemToRegion[sid];
      if (rid && regionEls[rid]) regionEls[rid].classList.add("highlight");
    },
    highlightRegion(rid) {
      for (const g of Object.values(regionEls)) g.classList.remove("highlight");
      if (rid && regionEls[rid]) regionEls[rid].classList.add("highlight");
    },
    regionForSystem(sid) {
      return systemToRegion[sid];
    },
  };
}
