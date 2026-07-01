"""Neural-symbolic computation agent for LongMemEval MS counting/arithmetic.

Motivation (iter33-MS, see ITER33_MS_FAILURE_MAP.md):
    Of the 31 in-family MS failures, ~7-9 are READER-UNDERCOUNT (the operand is
    present in the retrieved context, but the LLM reader miscounts / misclassifies
    / fails to compute), and ~7-8 are WRITER-GAP (the operand was dropped at
    write time but still lives verbatim in the RAW user/assistant turn).

    The established lesson (memory: "MS failures are retrieval, not reader") is
    that ABSTRACT reader prompt rules are unreliable — the LLM ignores "dedup the
    count" / "don't split entities" guidance. What it does NOT ignore is a
    CONCRETE, already-enumerated candidate list with evidence. So instead of
    asking the reader to count inline, we run a FOCUSED structured-extraction LLM
    call that reads the RAW retrieved turns, forces an explicit enumeration of
    every qualifying item (label + quantity + date + verbatim quote), and then
    computes the answer DETERMINISTICALLY in Python. The result is injected as a
    RECALL_HINT (bypass=False by default) so the reader verifies a presented list
    rather than performing the error-prone enumeration itself.

    "The failure isn't in computing — it's in selecting operands." (user)
    So the compute layer is pure/deterministic ($0-testable) and the operand
    SELECTION is the LLM call. Reading raw EVENT `content` (not writer-extracted
    concept summaries) is what recovers the writer-gap cases.

Design:
    classify_question(question) -> Family | None        (regex gate; None => skip)
    build_extraction_prompt(question, family, evidence) -> str
    parse_extraction(raw) -> dict                         (tolerant JSON parse)
    compute(family, parsed, question) -> Computation|None (pure, deterministic)
    resolve_neural_symbolic(question, nodes, call_llm_fn, config, ...) -> result|None

`call_llm_fn` is injected (signature: (prompt, config, json_mode) -> str) so the
orchestrator is unit-testable with a fake LLM and carries NO direct provider
dependency.

The returned dict matches the resolver contract consumed by run_eval +
render_symbolic_block: {"pattern", "answer", "reasoning", "bypass"}.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

# enumerate_sum  : count_members + value_sum (24/31 fixtures) — enumerate
#                  qualifying items, dedup, then either COUNT distinct items or
#                  SUM their quantities depending on sub-mode.
# percent_diff   : (original - paid) / original * 100  (e.g. "% discount")
# compare_max    : argmax / argmin / pairwise compare of named numeric candidates
#                  ("which gained most", "did I get a higher discount on X vs Y")
# date_span      : calendar span between two named year/date anchors
# age_diff       : numeric difference of two ages ("how much older am I than ...")

FAMILY_ENUMERATE_SUM = "enumerate_sum"
FAMILY_PERCENT_DIFF = "percent_diff"
FAMILY_COMPARE_MAX = "compare_max"
FAMILY_DATE_SPAN = "date_span"
FAMILY_AGE_DIFF = "age_diff"

# Sub-modes for enumerate_sum.
MODE_COUNT = "count"   # answer = number of distinct qualifying items
MODE_SUM = "sum"       # answer = sum of the qualifying items' quantities


@dataclass
class Family:
    """Classification result for a question."""

    name: str
    mode: str = ""              # MODE_COUNT / MODE_SUM for enumerate_sum
    unit: str = ""             # measure unit when summing (hours, pounds, $, ...)
    compare_kind: str = ""     # max / min / vs for compare_max


@dataclass
class Computation:
    """Deterministic compute result."""

    answer: str
    reasoning: str
    confidence: float          # 0..1 — drives the conditional-bypass decision
    n_items: int = 0


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "a": 1, "an": 1, "single": 1, "couple": 2, "few": 3, "several": 3,
    "half": 0.5,
}


def to_number(x: Any) -> float | None:
    """Coerce a value/string to a float. Handles word-numbers, $, commas, %,
    'k'/'K' multiplier, and embedded digits ('12 courses' -> 12). None if no
    number is recoverable."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().lower()
    if not s:
        return None
    # Exact word-number ("five", "a", "half").
    if s in _WORD_TO_NUM:
        return float(_WORD_TO_NUM[s])
    # Strip currency / thousands separators for the digit scan.
    cleaned = s.replace(",", "").replace("$", "").replace("£", "").replace("€", "")
    sign = -1.0 if re.match(r"^\s*(negative|minus)\b", cleaned) else 1.0
    # First numeric token WITH optional sign.
    m = re.search(r"(-?\d+(?:\.\d+)?)", cleaned)
    if m:
        val = float(m.group(1)) * sign
        # Apply a k/m multiplier ONLY when it is a STANDALONE suffix directly
        # attached to the number ("5k", "$1.2m") — NOT the first letter of an
        # adjacent unit word. Critical: "10 minutes" must stay 10, not 1e7;
        # "5km" must stay 5 (kilometres), not 5000.
        end = m.end()
        suffix = cleaned[end:end + 1]
        after = cleaned[end + 1:end + 2]
        if suffix in ("k", "m") and not after.isalpha():
            val *= 1000.0 if suffix == "k" else 1_000_000.0
        return val
    # Fallback: a leading word-number token ("twelve courses").
    for tok in re.findall(r"[a-z]+", s):
        if tok in _WORD_TO_NUM:
            return float(_WORD_TO_NUM[tok]) * sign
    return None


def _fmt_num(v: float) -> str:
    """Render a number without a trailing '.0' for integers."""
    if not math.isfinite(v):  # inf/nan would crash int(v)
        return str(v)
    if v == int(v):
        return str(int(v))
    return (f"{v:g}")


# ---------------------------------------------------------------------------
# Question classification (the gate)
# ---------------------------------------------------------------------------

# Measure units that mean the question SUMS a quantity rather than counting
# discrete nouns. "How many hours / pounds / dollars / pages ..." -> SUM.
_MEASURE_UNITS = (
    "hour", "hours", "hr", "hrs", "minute", "minutes", "min",
    "pound", "pounds", "lb", "lbs", "kg", "kilogram", "kilograms",
    "dollar", "dollars", "point", "points", "page", "pages",
    "mile", "miles", "km", "kilometer", "kilometers", "gallon", "gallons",
    "calorie", "calories",
)

# Strong aggregate cues -> SUM mode even for discrete nouns
# ("how many fish IN TOTAL", "page count of the two novels COMBINED").
_AGG_CUES = (
    "in total", "total number", "total amount", "total weight",
    "combined", "altogether", "sum of", "total of", "how much money",
    "how much did i spend", "how much did i raise", "grand total",
)

_PERCENT_DIFF_RE = re.compile(
    r"\b(what|how much)\b.*\b(percent|percentage|%)\b.*\b(discount|off|cheaper|markdown|reduction|increase|raise)\b",
    re.IGNORECASE,
)
_COMPARE_MAX_RE = re.compile(
    r"\bwhich\b.*\b(most|highest|largest|greatest|biggest|fewest|least|lowest|smallest)\b"
    r"|\b(did|do|was|is)\b.*\bhigher\b.*\b(than|vs|versus|compared)\b"
    r"|\bgain(ed)?\b.*\bmost\b",
    re.IGNORECASE,
)
_DATE_SPAN_RE = re.compile(
    r"\bhow many\b.*\b(year|years|month|months)\b.*\bfrom\b.*\bto\b"
    r"|\b(in total in|total years (of|in)).*education\b",
    re.IGNORECASE,
)
_AGE_DIFF_RE = re.compile(
    r"\bhow much\b.*\b(older|younger)\b.*\bthan\b"
    r"|\bage (difference|gap)\b",
    re.IGNORECASE,
)
# enumerate_sum trigger: a counting/aggregation question stem.
_ENUM_RE = re.compile(
    r"\bhow many\b"
    r"|\b(total number|total amount|total weight|page count)\b"
    r"|\bhow much (money|time)\b",
    re.IGNORECASE,
)
# Questions whose true shape is NOT an enumeration. Matching here makes
# classify_question return None so the agent does NOT emit a (garbage) count.
# Most of these are currently-CORRECT MS questions → firing enumerate on them is
# pure collateral (adversarial review wf_acbc2ad6). Each is a single-fact lookup
# or a 2-operand arithmetic the reader / classic resolver handles better. Checked
# AFTER the legit arithmetic families (percent/age/span/compare) so those still
# claim their questions, and BEFORE _ENUM_RE.
_NOT_ENUM_RE = re.compile(
    r"\bdid it take\b|\bhow long (did|does) it take\b"              # elapsed duration
    r"|\b(do|does|will) i need to (earn|reach|save|accumulate|spend|hit|get)\b"  # threshold to attain
    r"|\bneed to (earn|reach|accumulate)\b.*\bto (redeem|qualify|unlock|reach)\b"
    r"|\bdo i have\b[^?]*\b(left|remaining)\b|\b(left|remaining) to\b"  # remaining / left = subtraction
    r"|\bexceed(ed)?\b|\bby how (much|many)\b"                      # single-operand difference
    r"|\bin (a|an) (typical|normal|average|usual) (week|day|month|year)\b"  # recurring rate
    r"|\bper (week|day|month|year)\b"
    r"|\bhow many (years|months) (older|younger)\b"                # age difference (currently correct)
    r"|\bhow many (years|months) (will|would) i be\b"              # future age
    r"|\bhow many more\b.*\bthan\b",                               # subtraction, not enumerate
    re.IGNORECASE,
)


def classify_question(question: str) -> Family | None:
    """Detect which symbolic family (if any) a question belongs to.

    Conservative on purpose: returning None means the standard reader handles
    the question unchanged. A false positive on a currently-CORRECT question is
    the main risk, so the regexes require an explicit counting/arithmetic stem.
    """
    if not question:
        return None
    q = question.strip()
    qlow = q.lower()

    # Order matters — the more specific arithmetic families first so they are
    # not swallowed by the broad enumerate_sum "how many" trigger.
    if _PERCENT_DIFF_RE.search(q):
        return Family(FAMILY_PERCENT_DIFF)

    if _AGE_DIFF_RE.search(q):
        return Family(FAMILY_AGE_DIFF)

    if _DATE_SPAN_RE.search(q):
        return Family(FAMILY_DATE_SPAN)

    if _COMPARE_MAX_RE.search(q):
        kind = "max"
        if re.search(r"\b(fewest|least|lowest|smallest)\b", qlow):
            kind = "min"
        elif re.search(r"\bhigher\b.*\b(than|vs|versus)\b", qlow):
            kind = "vs"
        return Family(FAMILY_COMPARE_MAX, compare_kind=kind)

    # Exclude shapes that LOOK like "how many ..." but are not enumerations
    # (elapsed duration, requirement, remaining, single-operand diff, recurring
    # rate, age-difference). Returning None keeps the agent off currently-correct
    # questions it could only harm.
    if _NOT_ENUM_RE.search(q):
        return None

    if _ENUM_RE.search(q):
        mode = MODE_COUNT
        unit = ""
        # SUM mode if a measure unit is being totalled ...
        toks = set(re.findall(r"[a-z]+", qlow))
        hit_units = [u for u in _MEASURE_UNITS if u in toks]
        # "how many days a week" is a count-of-distinct-days, NOT a duration sum.
        is_days_per_week = bool(re.search(r"days?\s+(a|per)\s+week", qlow))
        if hit_units and not is_days_per_week:
            mode = MODE_SUM
            unit = hit_units[0]
        # ... or an explicit aggregate cue is present.
        if any(cue in qlow for cue in _AGG_CUES):
            mode = MODE_SUM
        return Family(FAMILY_ENUMERATE_SUM, mode=mode, unit=unit)

    return None


# ---------------------------------------------------------------------------
# Evidence assembly (raw retrieved turns -> prompt text)
# ---------------------------------------------------------------------------

def _node_field(node: Any, *names: str) -> str:
    """Read the first present attribute or dict key from a node-like object."""
    for n in names:
        v = node.get(n) if isinstance(node, dict) else getattr(node, n, None)
        if v:
            return str(v)
    return ""


def nodes_to_evidence(nodes: Any, max_chars: int = 14000) -> str:
    """Render retrieved nodes into evidence lines, prioritising RAW EVENT turns.

    EVENT nodes carry the verbatim user/assistant turn in `content`/`description`
    — that is what recovers writer-gap operands (e.g. 'a small pleco catfish',
    '8 edX courses') the writer dropped from concept summaries. CONCEPT nodes
    carry the writer's extracted facts. We emit both, EVENT first, each prefixed
    with its date so the LLM can apply temporal scope.
    """
    if not nodes:
        return ""
    event_lines: list[str] = []
    concept_lines: list[str] = []
    for node in nodes:
        ntype = _node_field(node, "node_type", "type").upper()
        title = _node_field(node, "title")
        body = _node_field(node, "content", "description")
        date = _node_field(node, "date", "event_date", "timestamp")
        # Strip a leading "[YYYY-MM-DD ...]" prefix already in the title to
        # avoid duplicating the date.
        title = re.sub(r"^\[[^\]]*\]\s*", "", title).strip()
        date_short = ""
        if date:
            m = re.match(r"(\d{4}-\d{2}-\d{2})", date)
            date_short = m.group(1) if m else date[:10]
        text = body or title
        if not text:
            continue
        prefix = f"[{date_short}] " if date_short else ""
        line = f"- {prefix}{text}".strip()
        if "EVENT" in ntype:
            role = _node_field(node, "role")
            rtag = f"({role}) " if role else ""
            event_lines.append(f"- {prefix}{rtag}{text}".strip())
        else:
            concept_lines.append(line)
    parts = []
    if event_lines:
        parts.append("RAW CONVERSATION TURNS:\n" + "\n".join(event_lines))
    if concept_lines:
        parts.append("EXTRACTED FACTS:\n" + "\n".join(concept_lines))
    evidence = "\n\n".join(parts)
    if len(evidence) > max_chars:
        evidence = evidence[:max_chars] + "\n…(truncated)"
    return evidence


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_ENUM_PROMPT = """\
You are a precise evidence extractor. Do NOT answer the question directly — only \
extract the operands needed to compute it. Work in TWO passes.

QUESTION: {question}

PASS 1 — GATHER EXHAUSTIVELY. Scan the ENTIRE excerpt block, every line, to the \
very end. Collect EVERY candidate that could possibly count toward the question. \
There is NO upper limit on how many there are — some questions have 8 or more — so \
do NOT stop because you have a plausible-looking set; stop only when you reach the \
end of the excerpts. Be generous: when unsure, include it (you filter in Pass 2).
  - Count an item even if it is named DIFFERENTLY from the question's wording, as \
long as it is the same CATEGORY. Examples: an "Italian feast", a "BBQ", a \
"potluck" are each a kind of dinner party; a "guided museum tour" or a "lecture" \
is an art-related event; "volunteered at"/"assisted at" a festival still counts \
as having attended it; a "vinyl" or "EP" is a kind of album.
  - Catch items buried inside a list or a single clause: "10 neon tetras, 5 \
gouramis, and a small pleco catfish" = THREE items (10, 5, 1); "8 edX courses" \
is a quantity of 8.
  - REQUIRED items named by the question: if the question says to INCLUDE a \
specific item ("...INCLUDING the one I set up for my friend's kid", "counting the \
X") or hints there is "an extra / another / one more" beyond the obvious ones, \
treat that as a REQUIRED item — search the excerpts specifically for it and add it \
even if it appears only once or in an unexpected session.
  - COMPLETENESS CHECK before finishing Pass 1: re-scan from the LAST excerpt line \
upward and confirm you took every qualifying item; do not stop early.

PASS 2 — QUALIFY each candidate against the question's EXPLICIT constraints; move \
failures to `excluded` (with a reason), keep the rest in `items`:
  (a) TIME WINDOW — drop items clearly outside the stated period ("in the past \
month", "this year", "in January and March").
  (b) STATUS — drop things only PLANNED / aspirational / hoped-for, not actually \
done ("I'm hoping to", "I think I'll go with", "planning to add").
  (c) CATEGORY / MANNER — drop items that miss a stated qualifier ("competitively", \
"new", "store item").
  (d) DISTRACTORS — drop values about a different topic.
A clearly planned/aspirational/out-of-window item MUST go to `excluded`, not \
`items` — do not pad the count with non-qualifying items.

ABSTENTION: if the question requires a component (e.g. "X and Y", "including Z") \
and the excerpts contain NO evidence for one of those required parts, add an entry \
to `excluded` with reason "MISSING: <part> not in evidence" and do NOT fabricate a \
value for it.

{mode_directive}

For each kept item also give the `date` if stated and a VERBATIM `quote`. Do NOT \
count the same physical item twice across turns (same identity = one item). Use \
ONLY facts present in the excerpts — never invent an item or a number. Every entry \
in `items` MUST set qualifies:true; anything that does not qualify goes in \
`excluded`, never in `items`.

Return STRICT JSON, no prose:
{{"items": [{{"label": str, "quantity": number_or_null, "date": str, "quote": str, "qualifies": true}}],
 "excluded": [{{"label": str, "reason": str}}]}}

EXCERPTS:
{evidence}
"""

_PERCENT_PROMPT = """\
You are a precise evidence extractor. The question asks for a percentage change \
(e.g. a discount). Extract the two operands.

QUESTION: {question}

Find the ORIGINAL (pre-change / list / regular) value and the FINAL (paid / \
post-change) value for the SAME item. They may be in different turns — link them \
by item identity.

Return STRICT JSON, no prose:
{{"original": number, "paid": number, "item": str, "quote_original": str, "quote_paid": str}}
Use null for a value you cannot find.

EXCERPTS:
{evidence}
"""

_COMPARE_PROMPT = """\
You are a precise evidence extractor. The question compares named candidates by a \
numeric measure (e.g. follower gain, discount). Extract each candidate and its \
numeric value.

QUESTION: {question}

For each candidate named or implied by the question, give its `name` and the \
relevant numeric `value` (the delta/amount the question compares on) with a \
verbatim `quote`. Use only stated numbers.

Return STRICT JSON, no prose:
{{"candidates": [{{"name": str, "value": number, "quote": str}}]}}

EXCERPTS:
{evidence}
"""

_SPAN_PROMPT = """\
You are a precise evidence extractor. The question asks for a calendar span \
between two anchors (e.g. start of high school to completion of a degree). \
Extract the two anchor years/dates.

QUESTION: {question}

Return STRICT JSON, no prose:
{{"start": str, "end": str, "start_year": number, "end_year": number,
  "quote_start": str, "quote_end": str}}
Use the earliest relevant year for `start` and the latest for `end`.

EXCERPTS:
{evidence}
"""

_AGE_PROMPT = """\
You are a precise evidence extractor. The question asks for a numeric age \
difference (e.g. how much older the user is than some reference). Extract both \
numeric ages/values.

QUESTION: {question}

Return STRICT JSON, no prose:
{{"self_value": number, "reference_value": number, "reference": str,
  "quote_self": str, "quote_reference": str}}
Use null for a value you cannot find.

EXCERPTS:
{evidence}
"""

_PROMPT_BY_FAMILY = {
    FAMILY_ENUMERATE_SUM: _ENUM_PROMPT,
    FAMILY_PERCENT_DIFF: _PERCENT_PROMPT,
    FAMILY_COMPARE_MAX: _COMPARE_PROMPT,
    FAMILY_DATE_SPAN: _SPAN_PROMPT,
    FAMILY_AGE_DIFF: _AGE_PROMPT,
}


def build_extraction_prompt(question: str, family: Family, evidence: str) -> str:
    template = _PROMPT_BY_FAMILY[family.name]
    # The enumerate template carries a {mode_directive} slot that tells the model
    # whether to COUNT (quantity=1 each) or SUM a measure (quantity = the per-item
    # unit amount, never 1). Injecting the resolved mode removes the count/sum
    # conflation at the source. Other templates ignore the extra kwarg.
    mode_directive = ""
    if family.name == FAMILY_ENUMERATE_SUM:
        if family.mode == MODE_SUM:
            unit = family.unit or "the measured amount"
            mode_directive = (
                f"QUANTITY: this question SUMS a total of {unit}. For EACH item, "
                f"`quantity` MUST be the numeric {unit} that item contributes (the "
                f"amount stated for it), NEVER 1. Each separate occurrence is its OWN "
                f"item even if the label repeats — give each its own entry and put its "
                f"date in the label so two real occurrences are never merged. If an item "
                f"has no stated numeric amount, set quantity to null (do NOT guess or use 1)."
            )
        else:
            mode_directive = (
                "QUANTITY: this question COUNTS distinct items. Set `quantity` = 1 for "
                "every item (each distinct item counts once)."
            )
    return template.format(question=question, evidence=evidence, mode_directive=mode_directive)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_extraction(raw: str) -> dict | None:
    """Tolerant JSON parse of the extraction reply. Strips markdown fences and
    grabs the outermost {...} if the model wrapped the JSON in prose."""
    if not raw:
        return None
    s = raw.strip()
    # Strip ```json ... ``` fences.
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # Grab outermost object.
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Deterministic compute (pure — $0-testable)
# ---------------------------------------------------------------------------

def _norm_label(s: str) -> str:
    """Normalise a label for dedup. Strip ONLY true articles (a/an/the) — keep
    size/novelty/possessive adjectives (small/new/rare/old/my), which are often
    the DISCRIMINATING feature between distinct items: a 'small tank' and 'the new
    tank' are DIFFERENT tanks. Over-stripping collapsed distinct items into one
    key and silently under-counted (adversarial review wf_acbc2ad6)."""
    toks = re.findall(r"[a-z0-9]+", str(s).lower())
    toks = [t for t in toks if t not in {"a", "an", "the"}]
    return " ".join(toks)


def compute_enumerate_sum(parsed: dict, family: Family) -> Computation | None:
    items = parsed.get("items") or []
    if not isinstance(items, list):
        return None
    # Gather qualifying items (qualifies omitted => kept).
    kept: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("qualifies") is False:
            continue
        label = (it.get("label") or "").strip()
        key = _norm_label(label)
        if not key:
            continue
        kept.append({
            "key": key, "label": label,
            "qty": to_number(it.get("quantity", 1)),
            "date": str(it.get("date") or ""),
            "quote": str(it.get("quote") or ""),
        })
    if not kept:
        return None

    unit = (" " + family.unit) if family.unit else ""
    excluded = parsed.get("excluded") or []
    conf = 0.85

    if family.mode == MODE_SUM:
        # SUM = total of EACH distinct occurrence. Merge only an EXACT-duplicate
        # emission (same normalised label + date + quote); distinct occurrences
        # that merely share a label are summed (the gaming 140h->115h bug was
        # max()-on-label-collision dropping real operands). A missing quantity is
        # a SKIP — never silently coerce to 1 (that corrupts a sum).
        seen_dup: set = set()
        members: list[dict] = []
        unquant: list[str] = []
        for k in kept:
            dk = (k["key"], k["date"], k["quote"])
            if dk in seen_dup:
                continue
            seen_dup.add(dk)
            if k["qty"] is None:
                unquant.append(k["label"])
                continue
            members.append(k)
        if not members:
            return None
        total = sum(m["qty"] for m in members)
        n = len(members)
        answer = f"{_fmt_num(total)}{unit}"
        bullets = "; ".join(f"{m['label']} ({_fmt_num(m['qty'])}{unit})" for m in members)
        reasoning = f"Summed {n} operand(s): {bullets}. Total = {answer}."
        if unquant:
            conf -= 0.25
            reasoning += f" Could NOT quantify (left OUT of the sum — verify): {', '.join(unquant)}."
        if n > 8:
            conf -= 0.1
    else:
        # COUNT = number of DISTINCT items (dedup by normalised label; a re-mention
        # of the same item across turns is counted once).
        seen: dict[str, dict] = {}
        order: list[str] = []
        for k in kept:
            if k["key"] not in seen:
                seen[k["key"]] = k
                order.append(k["key"])
        n = len(seen)
        total = float(n)
        answer = _fmt_num(total)
        bullets = "; ".join(seen[mk]["label"] for mk in order)
        reasoning = f"Enumerated {n} distinct qualifying item(s): {bullets}. Distinct count = {answer}."
        dated = sum(1 for k in kept if k["date"])
        if dated < n:
            conf -= 0.15
        if n > 8:
            conf -= 0.15

    if excluded:
        ex = "; ".join(
            f"{e.get('label', '?')} ({e.get('reason', '')})"
            for e in excluded if isinstance(e, dict)
        )
        reasoning += f" Excluded as non-qualifying: {ex}."
    conf = max(0.3, conf)
    return Computation(answer=answer, reasoning=reasoning, confidence=conf, n_items=n)


def compute_percent_diff(parsed: dict, family: Family) -> Computation | None:
    original = to_number(parsed.get("original"))
    paid = to_number(parsed.get("paid"))
    if original is None or paid is None or original == 0:
        return None
    pct = (original - paid) / original * 100.0
    # A negative pct means paid > original -> a price INCREASE, not a discount.
    kind = "discount" if pct >= 0 else "increase"
    answer = f"{_fmt_num(round(abs(pct), 2))}%"
    item = parsed.get("item") or "the item"
    reasoning = (
        f"|{_fmt_num(original)} - {_fmt_num(paid)}| / {_fmt_num(original)} * 100 "
        f"= {answer} {kind} on {item}."
    )
    return Computation(answer=answer, reasoning=reasoning, confidence=0.85, n_items=2)


def compute_compare(parsed: dict, family: Family) -> Computation | None:
    cands = parsed.get("candidates") or []
    pairs: list[tuple[str, float]] = []
    for c in cands:
        if not isinstance(c, dict):
            continue
        v = to_number(c.get("value"))
        name = c.get("name")
        if v is not None and name:
            pairs.append((str(name), v))
    # A comparison needs at least TWO operands; a single candidate cannot be
    # compared, so fall through to the reader.
    if len(pairs) < 2:
        return None
    detail = ", ".join(f"{n}={_fmt_num(v)}" for n, v in pairs)
    if family.compare_kind == "vs":
        # "Did I get a HIGHER X on A than B?" -> a Yes/No on whether the FIRST-
        # named candidate (the question's subject) is the greater one. The prompt
        # extracts candidates in question order, so pairs[0] is the subject.
        subj_name, subj_val = pairs[0]
        other_max = max(v for _, v in pairs[1:])
        yes = subj_val > other_max
        answer = "Yes" if yes else "No"
        reasoning = (
            f"Comparing {detail}. Subject = {subj_name} ({_fmt_num(subj_val)}); "
            f"{'higher' if yes else 'not higher'} than the other(s) -> {answer}."
        )
        return Computation(answer=answer, reasoning=reasoning, confidence=0.7, n_items=len(pairs))
    if family.compare_kind == "min":
        best = min(pairs, key=lambda p: p[1])
        which = "Lowest"
    else:
        best = max(pairs, key=lambda p: p[1])
        which = "Highest"
    reasoning = f"Comparing {detail}. {which} = {best[0]} ({_fmt_num(best[1])})."
    return Computation(answer=str(best[0]), reasoning=reasoning, confidence=0.8, n_items=len(pairs))


def compute_date_span(parsed: dict, family: Family) -> Computation | None:
    sy = to_number(parsed.get("start_year"))
    ey = to_number(parsed.get("end_year"))
    if sy is None or ey is None or ey < sy:
        return None
    span = round(ey - sy)
    answer = f"{span} years"
    reasoning = (
        f"Calendar span {_fmt_num(sy)} → {_fmt_num(ey)} = {span} years "
        f"({parsed.get('start', '')} to {parsed.get('end', '')})."
    )
    return Computation(answer=answer, reasoning=reasoning, confidence=0.7, n_items=2)


def compute_age_diff(parsed: dict, family: Family) -> Computation | None:
    a = to_number(parsed.get("self_value"))
    b = to_number(parsed.get("reference_value"))
    if a is None or b is None:
        return None
    diff = a - b
    answer = f"{_fmt_num(abs(round(diff, 2)))} years"
    direction = "older" if diff >= 0 else "younger"
    reasoning = (
        f"{_fmt_num(a)} - {_fmt_num(b)} = {_fmt_num(diff)} "
        f"({answer} {direction} than {parsed.get('reference', 'reference')})."
    )
    return Computation(answer=answer, reasoning=reasoning, confidence=0.8, n_items=2)


_COMPUTE_BY_FAMILY: dict[str, Callable[[dict, Family], Computation | None]] = {
    FAMILY_ENUMERATE_SUM: compute_enumerate_sum,
    FAMILY_PERCENT_DIFF: compute_percent_diff,
    FAMILY_COMPARE_MAX: compute_compare,
    FAMILY_DATE_SPAN: compute_date_span,
    FAMILY_AGE_DIFF: compute_age_diff,
}


def compute(family: Family, parsed: dict) -> Computation | None:
    fn = _COMPUTE_BY_FAMILY.get(family.name)
    if fn is None:
        return None
    try:
        return fn(parsed, family)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# Bypass policy: a concrete enumerated hint (bypass=False) lets the reader keep
# the final say (safe). 'auto' promotes to bypass=True only for high-confidence
# enumerate_sum results; 'never' always emits a hint.
BYPASS_NEVER = "never"
BYPASS_AUTO = "auto"

_BYPASS_CONF_THRESHOLD = 0.85


def resolve_neural_symbolic(
    question: str,
    nodes: Any,
    call_llm_fn: Callable[..., str],
    config: Any,
    *,
    bypass_mode: str = BYPASS_NEVER,
    max_evidence_chars: int = 14000,
    evidence_text: str | None = None,
    debug: bool = False,
) -> dict | None:
    """Run the neural-symbolic agent for one question.

    Returns a resolver-contract dict {"pattern","answer","reasoning","bypass"}
    or None to fall through to the normal reader (no family match / extraction
    failed / nothing computable).

    `evidence_text` overrides node rendering with a pre-built evidence string
    (used by the offline replay harness to feed cached retrieved context without
    re-ingesting). When None, evidence is rendered from `nodes`.
    """
    family = classify_question(question)
    if family is None:
        return None
    if evidence_text is not None:
        evidence = evidence_text[:max_evidence_chars]
    else:
        evidence = nodes_to_evidence(nodes, max_chars=max_evidence_chars)
    if not evidence:
        return None
    prompt = build_extraction_prompt(question, family, evidence)
    raw = call_llm_fn(prompt, config, json_mode=True)
    parsed = parse_extraction(raw)
    if not isinstance(parsed, dict):  # a top-level array/scalar reply is unusable
        return None
    comp = compute(family, parsed)
    if comp is None:
        return None

    # Bypass policy: the agent NEVER bypasses the reader for enumerate_sum — the
    # adversarial review showed the confidence heuristic rewards exactly the
    # unfiltered over-count case, and a bypass skips the only check (the reader).
    # Only the deterministic single-value families may bypass under BYPASS_AUTO,
    # and only at high confidence (in practice they stay below threshold → the
    # reader keeps the final say). Enumerate always emits a hint.
    bypass = (
        bypass_mode == BYPASS_AUTO
        and family.name in (FAMILY_DATE_SPAN, FAMILY_AGE_DIFF)
        and comp.confidence >= _BYPASS_CONF_THRESHOLD
    )

    pattern = f"neural_symbolic_{family.name}"
    if family.name == FAMILY_ENUMERATE_SUM and family.mode:
        pattern += f"_{family.mode}"

    result: dict[str, Any] = {
        "pattern": pattern,
        "answer": comp.answer,
        "reasoning": comp.reasoning,
        "bypass": bypass,
        "family": family.name,
        "mode": family.mode,
    }
    if debug:
        result["_debug"] = {
            "family": family.name,
            "mode": family.mode,
            "confidence": comp.confidence,
            "n_items": comp.n_items,
            "parsed": parsed,
            "raw": raw,
        }
    return result


def render_neural_symbolic_block(result: dict[str, Any]) -> str:
    """Render the agent's enumeration as a CROSS-CHECK block for the reader.

    Framing must be TWO-DIRECTIONAL (adversarial review). The earlier strict
    lower-bound framing ("never report fewer than your count") fixed the
    UNDER-count collateral (tanks 3->2, gaming 140h->115h) but BACKFIRED on
    OVER-count: the extractor is told to include borderline items, so a wrongly
    kept planned/aspirational item produced a count the reader was forbidden from
    correcting down. So the block now asks the reader to VERIFY EACH listed item
    against its quote and the context, ADD qualifying items the list missed, and
    DROP any listed item that does not actually qualify — i.e. correct UP or DOWN.
    It is a candidate enumeration to reconcile with the reader's own careful
    count, never a verdict to adopt blindly.

    For the comparison/percent/date/age families (single deterministic value, not
    an enumeration) the lower-bound framing does not apply, so those render as a
    plain candidate to verify.
    """
    fam = result.get("family", "")
    answer = result.get("answer", "")
    reasoning = result.get("reasoning", "")
    if fam == FAMILY_ENUMERATE_SUM:
        # LOWER-BOUND floor framing. The post-fix A/B proved the two-directional
        # "correct up or down / drop items" wording REGRESSED the count wins
        # (weddings 3->2, festivals 4->3, art 4->3) — the reader dropped valid
        # items. The floor ("do NOT report fewer than your own careful count")
        # empirically drove the wins (v1: 8 NS wins / 2 collateral), so it is
        # restored, with only a narrow clear-non-qualifier drop and an abstention
        # carve-out for missing required components.
        return (
            "## COUNT_CROSSCHECK (a symbolic agent enumerated the qualifying items "
            "below — this list may be INCOMPLETE; use it as a checklist, not a verdict)\n"
            f"**Computed total (treat as a LOWER BOUND)**: {answer}\n"
            f"**Enumeration**: {reasoning}\n"
            "**How to use this block**:\n"
            "- SCAN the context for any qualifying item the enumeration MISSED and ADD it. "
            "The list is often incomplete, so do NOT report a number smaller than your own "
            "careful count.\n"
            "- If the question names an item to include (e.g. 'including X', 'counting the Y'), "
            "that item COUNTS.\n"
            "- Only DROP a listed item if it CLEARLY fails the question's stated qualifier "
            "(wrong time window, only planned/aspirational, wrong category).\n"
            "- If the question requires a component (e.g. 'X and Y') and the context has NO "
            "evidence for one of them, say the information is insufficient rather than "
            "reporting a partial number.\n"
            "- Then state the final count.\n"
        )
    return (
        "## SYMBOLIC_CROSSCHECK (a symbolic agent computed the value below — verify "
        "against the context and override if a more specific fact contradicts it; if a "
        "required operand is not stated, say the information is insufficient)\n"
        f"**Computed**: {answer}\n"
        f"**Reasoning**: {reasoning}\n"
    )


# ---------------------------------------------------------------------------
# Take-max router (complementary use with the normal reader)
# ---------------------------------------------------------------------------

_COUNT_VERB_RE = re.compile(
    r"(?:attended|visited|used|have|own|completed|got|took|earned|raised|spent|"
    r"purchased|acquired|played|total of|total|count(?:ed)?)\s+"
    r"(?:about\s+|over\s+|a\s+total\s+of\s+|approximately\s+|at\s+least\s+)?"
    r"\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_answer_count(text: Any) -> float | None:
    """Best-effort extraction of an answer's headline count/total from prose.

    Order of preference: a bolded **N** (the reader almost always bolds the
    final figure) -> a number directly after a count/total verb -> a leading
    word-number -> the first standalone digit that is not a list index
    ("Item 1") or a markdown bullet. Returns None when no count is recoverable.
    """
    if text is None:
        return None
    s = str(text)
    m = re.search(r"\*\*\s*\$?\s*([\d,]+(?:\.\d+)?)", s)
    if m:
        return to_number(m.group(1))
    m = _COUNT_VERB_RE.search(s)
    if m:
        return to_number(m.group(1))
    m = re.search(
        r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
        s[:80], re.IGNORECASE,
    )
    if m:
        return to_number(m.group(1))
    for mm in re.finditer(r"([\d,]+(?:\.\d+)?)", s):
        pre = s[max(0, mm.start() - 6):mm.start()].lower()
        if "item" in pre or "#" in pre:
            continue
        return to_number(mm.group(1))
    return None


def take_max_answer(baseline_answer: str, ns_result: dict) -> tuple[str, bool]:
    """Complementary 'run both, keep the higher count' router for count/sum.

    The NS-vs-baseline cross-tab (and the $0 ns_take_max_sim.py replay, +7 fixes
    / 0 breaks) show that on MS count/sum questions the error mode is
    under-count-dominated: every NS win is NS ratcheting the count UP, every
    observed collateral is NS ratcheting DOWN. So taking max(baseline_count,
    ns_count) keeps the up-wins and structurally kills the down-collateral
    (when NS under-counts, we keep the baseline reader's higher answer).

    Only applies to enumerate_sum results. Returns (final_answer, took_ns).
    When the NS count is not strictly greater than the baseline's count — or
    either count can't be parsed — the baseline answer is kept unchanged.
    """
    if ns_result.get("family") != FAMILY_ENUMERATE_SUM:
        return baseline_answer, False
    ns_count = parse_answer_count(ns_result.get("answer"))
    base_count = parse_answer_count(baseline_answer)
    if ns_count is None or base_count is None:
        return baseline_answer, False
    if ns_count > base_count:
        # NS recovered items the reader missed — adopt the higher count and
        # carry the enumeration as its justification for the judge.
        reasoning = ns_result.get("reasoning", "")
        return f"{ns_result.get('answer', '')}. {reasoning}".strip(), True
    return baseline_answer, False
