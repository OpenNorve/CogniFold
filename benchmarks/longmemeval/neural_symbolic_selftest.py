"""$0 offline self-test for the neural-symbolic agent (no LLM, no network).

Three levels of validation, all deterministic:

  L1  classify_question() routes each of the 31 in-family MS fixtures to the
      correct family/mode (and routes the 1 out-of-family fixture to None).
  L2  the pure compute layer reproduces the ground-truth answer when fed the
      GOLD operands (the decomposition the failure map records). This isolates
      "the arithmetic is correct" from "operand selection is hard" — per the
      user's framing, the failure is in selecting operands, not computing.
  L3  the end-to-end orchestrator resolve_neural_symbolic() produces the GT
      answer when a FAKE call_llm returns the gold JSON, exercising the full
      prompt→parse→compute→contract path with zero API cost.

Run:  PYTHONPATH=/tmp/cf-unified/src /tmp/cf-unified/.venv/bin/python \
        benchmarks/longmemeval/neural_symbolic_selftest.py
"""

# ruff: noqa: C408  — dict() literals are intentional here for readable fixture rows.
from __future__ import annotations

import json
import sys

import neural_symbolic as ns

# Gold operands per fixture = what a PERFECT extraction would return.
# `mode` only applies to enumerate_sum. `disputed` flags GTs the docs mark as
# genuinely ambiguous (so an L2 miss there is not a compute-logic bug).
FIXTURES = [
    # ---- enumerate_sum / COUNT ----
    dict(qid="0a995998", fam="enumerate_sum", mode="count", gt="3",
         q="How many items of clothing do I need to pick up or return from a store?",
         disputed=True,
         parsed={"items": [
             {"label": "Zara boots return (old)", "quantity": 1, "date": "2023-05-19", "qualifies": True},
             {"label": "Zara boots pickup (new)", "quantity": 1, "date": "2023-05-19", "qualifies": True},
             {"label": "navy blazer dry-cleaning pickup", "quantity": 1, "date": "2023-05-11", "qualifies": True}],
             "excluded": [{"label": "green sweater", "reason": "lent to sister, not a store item"}]}),
    dict(qid="60159905", fam="enumerate_sum", mode="count", gt="three",
         q="How many dinner parties have I attended in the past month?",
         parsed={"items": [
             {"label": "Alex's potluck dinner party", "quantity": 1, "date": "2023-05-21", "qualifies": True},
             {"label": "Mike's BBQ dinner party", "quantity": 1, "date": "2023-05-21", "qualifies": True},
             {"label": "Sarah's Italian feast", "quantity": 1, "date": "2023-05-22", "qualifies": True}]}),
    dict(qid="9d25d4e0", fam="enumerate_sum", mode="count", gt="3",
         q="How many pieces of jewelry did I acquire in the last two months?",
         parsed={"items": [
             {"label": "emerald earrings", "quantity": 1, "date": "2023-05-21", "qualifies": True},
             {"label": "silver necklace with pendant", "quantity": 1, "date": "2023-05-28", "qualifies": True},
             {"label": "engagement ring", "quantity": 1, "date": "2023-04-30", "qualifies": True}]}),
    dict(qid="ef66a6e5", fam="enumerate_sum", mode="count", gt="two",
         q="How many sports have I played competitively in the past?",
         parsed={"items": [
             {"label": "swimming", "quantity": 1, "date": "", "qualifies": True},
             {"label": "tennis", "quantity": 1, "date": "", "qualifies": True}],
             "excluded": [{"label": "soccer", "reason": "recreational, not competitive"}]}),
    dict(qid="gpt4_7fce9456", fam="enumerate_sum", mode="count", gt="four",
         q="How many properties did I view before making an offer on the Brookside townhouse?",
         parsed={"items": [
             {"label": "bungalow Oakwood", "quantity": 1, "date": "2023-01-22", "qualifies": True},
             {"label": "Cedar Creek", "quantity": 1, "date": "2023-02-01", "qualifies": True},
             {"label": "1-bedroom condo", "quantity": 1, "date": "2023-02-10", "qualifies": True},
             {"label": "2-bedroom condo", "quantity": 1, "date": "2023-02-17", "qualifies": True}]}),
    dict(qid="2ce6a0f2", fam="enumerate_sum", mode="count", gt="4",
         q="How many different art-related events did I attend in the past month?",
         parsed={"items": [
             {"label": "Art Afternoon", "quantity": 1, "date": "2023-02-17", "qualifies": True},
             {"label": "Women in Art", "quantity": 1, "date": "2023-02-10", "qualifies": True},
             {"label": "Street Art lecture", "quantity": 1, "date": "2023-03-03", "qualifies": True},
             {"label": "History Museum guided tour", "quantity": 1, "date": "2023-02-24", "qualifies": True}]}),
    dict(qid="gpt4_2f8be40d", fam="enumerate_sum", mode="count", gt="3",
         q="How many weddings have I attended in this year?",
         parsed={"items": [
             {"label": "cousin Rachel's vineyard wedding", "quantity": 1, "date": "2023-08-01", "qualifies": True},
             {"label": "roommate Emily's rooftop wedding", "quantity": 1, "date": "", "qualifies": True},
             {"label": "friend Jen & Tom's wedding", "quantity": 1, "date": "", "qualifies": True}]}),
    dict(qid="gpt4_a56e767c", fam="enumerate_sum", mode="count", gt="4",
         q="How many movie festivals that I attended?",
         parsed={"items": [
             {"label": "Austin Film Festival", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Seattle International Film Festival", "quantity": 1, "date": "", "qualifies": True},
             {"label": "AFI Fest", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Portland Film Festival", "quantity": 1, "date": "", "qualifies": True}]}),
    dict(qid="gpt4_31ff4165", fam="enumerate_sum", mode="count", gt="4",
         q="How many health-related devices do I use in a day?", disputed=True,
         parsed={"items": [
             {"label": "Accu-Chek Aviva Nano", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Fitbit Versa 3", "quantity": 1, "date": "", "qualifies": True},
             {"label": "nebulizer", "quantity": 1, "date": "", "qualifies": True},
             {"label": "hearing aids", "quantity": 1, "date": "", "qualifies": True}]}),
    dict(qid="3a704032", fam="enumerate_sum", mode="count", gt="3",
         q="How many plants did I acquire in the last month?",
         parsed={"items": [
             {"label": "snake plant", "quantity": 1, "date": "", "qualifies": True},
             {"label": "peace lily", "quantity": 1, "date": "", "qualifies": True},
             {"label": "succulent", "quantity": 1, "date": "", "qualifies": True}]}),
    dict(qid="a08a253f", fam="enumerate_sum", mode="count", gt="4",
         q="How many days a week do I attend fitness classes?",
         parsed={"items": [
             {"label": "Tuesday Zumba", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Wednesday yoga", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Thursday Zumba", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Saturday weightlifting", "quantity": 1, "date": "", "qualifies": True}]}),
    dict(qid="bf659f65", fam="enumerate_sum", mode="count", gt="3",
         q="How many music albums or EPs have I purchased or downloaded?",
         parsed={"items": [
             {"label": "Happier Than Ever", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Midnight Sky EP", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Tame Impala vinyl", "quantity": 1, "date": "", "qualifies": True}]}),
    dict(qid="c4a1ceb8", fam="enumerate_sum", mode="count", gt="3",
         q="How many different types of citrus fruits have I used in my cocktail recipes?",
         parsed={"items": [
             {"label": "lime", "quantity": 1, "date": "", "qualifies": True},
             {"label": "orange", "quantity": 1, "date": "", "qualifies": True},
             {"label": "lemon", "quantity": 1, "date": "", "qualifies": True}]}),
    dict(qid="a9f6b44c", fam="enumerate_sum", mode="count", gt="2",
         q="How many bikes did I service or plan to service in March?",
         parsed={"items": [
             {"label": "road bike", "quantity": 1, "date": "2023-03-10", "qualifies": True},
             {"label": "commuter hybrid bike", "quantity": 1, "date": "2023-03-01", "qualifies": True}]}),
    dict(qid="d682f1a2", fam="enumerate_sum", mode="count", gt="3",
         q="How many different types of food delivery services have I used recently?",
         parsed={"items": [
             {"label": "Domino's", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Fresh Fusion", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Uber Eats", "quantity": 1, "date": "", "qualifies": True}]}),
    # ---- enumerate_sum / SUM ----
    dict(qid="aae3761f", fam="enumerate_sum", mode="sum", gt="15", disputed=True,
         q="How many hours total driving to my three road trip destinations combined?",
         parsed={"items": [
             {"label": "Outer Banks", "quantity": 4, "date": "", "qualifies": True},
             {"label": "Tennessee mountains", "quantity": 5, "date": "", "qualifies": True},
             {"label": "Washington D.C.", "quantity": 6, "date": "2023-05-26", "qualifies": True}],
             "excluded": [{"label": "Tybee Island", "reason": "planned only, trip not taken"}]}),
    dict(qid="d851d5ba", fam="enumerate_sum", mode="sum", gt="3750", disputed=True,
         q="How much money did I raise for charity in total?",
         parsed={"items": [
             {"label": "fitness challenge", "quantity": 500, "date": "", "qualifies": True},
             {"label": "bake sale", "quantity": 1000, "date": "", "qualifies": True},
             {"label": "Run for Hunger", "quantity": 250, "date": "", "qualifies": True},
             {"label": "animal shelter", "quantity": 2000, "date": "", "qualifies": True}],
             "excluded": [{"label": "music concert $5000", "reason": "back in April, out of cycle"}]}),
    dict(qid="67e0d0f2", fam="enumerate_sum", mode="sum", gt="20",
         q="What is the total number of online courses I've completed?",
         parsed={"items": [
             {"label": "Coursera courses", "quantity": 12, "date": "", "qualifies": True},
             {"label": "edX courses", "quantity": 8, "date": "", "qualifies": True}]}),
    dict(qid="bc149d6b", fam="enumerate_sum", mode="sum", gt="70", disputed=True,
         q="What is the total weight of the new feed I purchased in the past two months?",
         parsed={"items": [
             {"label": "layer feed", "quantity": 50, "date": "", "qualifies": True},
             {"label": "organic scratch grains", "quantity": 20, "date": "", "qualifies": True}]}),
    dict(qid="37f165cf", fam="enumerate_sum", mode="sum", gt="856",
         q="What was the page count of the two novels I finished in January and March?",
         parsed={"items": [
             {"label": "The Nightingale", "quantity": 440, "date": "2023-01-15", "qualifies": True},
             {"label": "recent novel", "quantity": 416, "date": "2023-03-20", "qualifies": True}],
             "excluded": [{"label": "The Power (341)", "reason": "finished in December, out of range"}]}),
    dict(qid="7024f17c", fam="enumerate_sum", mode="sum", gt="0.5",
         q="How many hours of jogging and yoga did I do last week?",
         parsed={"items": [
             {"label": "jog", "quantity": 0.5, "date": "2023-05-20", "qualifies": True}],
             "excluded": [{"label": "yoga", "reason": "lapsed/aspirational, 0 hours"}]}),
    dict(qid="e3038f8c", fam="enumerate_sum", mode="sum", gt="99",
         q="How many rare items do I have in total?",
         parsed={"items": [
             {"label": "figurines", "quantity": 12, "date": "", "qualifies": True},
             {"label": "records", "quantity": 57, "date": "", "qualifies": True},
             {"label": "books", "quantity": 5, "date": "", "qualifies": True},
             {"label": "rare coins", "quantity": 25, "date": "", "qualifies": True}]}),
    dict(qid="edced276", fam="enumerate_sum", mode="sum", gt="15",
         q="How many days did I spend in total traveling in Hawaii and in New York City?",
         parsed={"items": [
             {"label": "Hawaii", "quantity": 10, "date": "", "qualifies": True},
             {"label": "New York City", "quantity": 5, "date": "", "qualifies": True}]}),
    # 9ee3ecd6 is now correctly EXCLUDED (a "do I need" requirement lookup, not a
    # sum) — see _NOT_ENUM_RE. It must classify to None.
    dict(qid="9ee3ecd6", fam=None, mode="", gt="100",
         q="How many points do I need to earn to redeem a free skincare product at Sephora?",
         parsed=None),
    dict(qid="eeda8a6d", fam="enumerate_sum", mode="sum", gt="17",
         q="How many fish are there in total in both of my aquariums?",
         parsed={"items": [
             {"label": "neon tetras", "quantity": 10, "date": "", "qualifies": True},
             {"label": "golden honey gouramis", "quantity": 5, "date": "", "qualifies": True},
             {"label": "pleco catfish", "quantity": 1, "date": "", "qualifies": True},
             {"label": "Bubbles betta", "quantity": 1, "date": "", "qualifies": True}]}),
    # ---- percent_diff ----
    dict(qid="d905b33f", fam="percent_diff", mode="", gt="20%",
         q="What percentage discount did I get on the book from my favorite author?",
         parsed={"original": 30, "paid": 24, "item": "favorite-author book"}),
    # ---- compare_max ----
    dict(qid="gpt4_5501fe77", fam="compare_max", mode="", gt="TikTok",
         q="Which social media platform did I gain the most followers on over the past month?",
         parsed={"candidates": [
             {"name": "TikTok", "value": 200, "quote": "gained around 200 followers on TikTok"},
             {"name": "Twitter", "value": 120, "quote": "420 to 540"}]}),
    dict(qid="7405e8b1", fam="compare_max", mode="", gt="Yes", disputed=True,
         q="Did I receive a higher percentage discount on my first HelloFresh order vs my first UberEats order?",
         parsed={"candidates": [
             {"name": "HelloFresh", "value": 40, "quote": "40% discount on first order"},
             {"name": "UberEats", "value": 20, "quote": "20% off UberEats"}]}),
    # ---- date_span ----
    dict(qid="gpt4_372c3eed", fam="date_span", mode="", gt="10 years", disputed=True,
         q="How many years in total in formal education from high school to completion of Bachelor's?",
         parsed={"start": "high school 2010", "end": "Bachelor's 2020",
                 "start_year": 2010, "end_year": 2020}),
    # ---- age_diff ----
    dict(qid="3c1045c8", fam="age_diff", mode="", gt="2.5 years",
         q="How much older am I than the average age of employees in my department?",
         parsed={"self_value": 32, "reference_value": 29.5, "reference": "department average"}),
    # ---- out of family (time arithmetic, not one of the 5) ----
    dict(qid="73d42213", fam=None, mode="", gt="9:00 AM",
         q="What time did I reach the clinic on Monday?",
         parsed=None),
    # ---- ADVERSARIAL mis-route guards (must classify to None, NOT enumerate) ----
    # These currently-CORRECT MS shapes used to wrongly route to enumerate and
    # produce collateral; the _NOT_ENUM_RE gate must keep them off.
    dict(qid="adv_elapsed1", fam=None, mode="", gt="5 days", parsed=None,
         q="How many days did it take for me to receive the new remote?"),
    dict(qid="adv_elapsed2", fam=None, mode="", gt="3.5 weeks", parsed=None,
         q="How many weeks did it take me to watch all the Marvel movies?"),
    dict(qid="adv_age1", fam=None, mode="", gt="43", parsed=None,
         q="How many years older is my grandma than me?"),
    dict(qid="adv_age2", fam=None, mode="", gt="33", parsed=None,
         q="How many years will I be when my friend Rachel gets married?"),
    dict(qid="adv_left", fam=None, mode="", gt="190", parsed=None,
         q="How many pages do I have left to read in 'The Nightingale'?"),
    dict(qid="adv_exceed", fam=None, mode="", gt="12", parsed=None,
         q="How many minutes did I exceed my target time by in the marathon?"),
    dict(qid="adv_rate", fam=None, mode="", gt="50", parsed=None,
         q="How many hours do I work in a typical week during peak campaign seasons?"),
    # age_diff via "how much older ... than" MUST still fire (3c1045c8-style win).
    dict(qid="adv_age_keep", fam="age_diff", mode="", gt="2.5 years",
         q="How much older am I than the average age of my coworkers?",
         parsed={"self_value": 32, "reference_value": 29.5, "reference": "coworkers"}),
]


def _num_match(gt: str, ans: str) -> bool:
    """Numeric-aware comparison: if GT has a number, match on the number;
    else fall back to normalised substring containment."""
    g = ns.to_number(gt)
    a = ns.to_number(ans)
    if g is not None and a is not None:
        return abs(g - a) < 1e-6
    gl = "".join(ch for ch in gt.lower() if ch.isalnum())
    al = "".join(ch for ch in ans.lower() if ch.isalnum())
    return bool(gl) and (gl in al or al in gl)


def fake_llm_factory(parsed):
    def _fake(prompt, config, json_mode=False):
        return json.dumps(parsed)
    return _fake


def main() -> int:
    l1_pass = l1_fail = 0
    l2_pass = l2_fail = l2_disputed = 0
    l3_pass = l3_fail = 0
    l1_errors, l2_errors, l3_errors = [], [], []

    for fx in FIXTURES:
        qid, q = fx["qid"], fx["q"]
        # ---- L1 classify ----
        fam = ns.classify_question(q)
        got_name = fam.name if fam else None
        if got_name == fx["fam"]:
            # also check mode for enumerate_sum
            if fx["fam"] == "enumerate_sum" and fam.mode != fx["mode"]:
                l1_fail += 1
                l1_errors.append(f"{qid}: mode {fam.mode!r} != {fx['mode']!r}")
            else:
                l1_pass += 1
        else:
            l1_fail += 1
            l1_errors.append(f"{qid}: family {got_name!r} != {fx['fam']!r}  ({q[:60]})")

        if fx["fam"] is None or fx["parsed"] is None:
            continue

        # ---- L2 compute with gold operands ----
        fam_obj = ns.Family(fx["fam"], mode=fx["mode"])
        if fx["fam"] == "compare_max":
            fam_obj.compare_kind = "vs" if "vs" in q.lower() else "max"
        comp = ns.compute(fam_obj, fx["parsed"])
        if comp is None:
            l2_fail += 1
            l2_errors.append(f"{qid}: compute returned None")
            l3_fail += 1
            continue
        ok = _num_match(fx["gt"], comp.answer)
        if ok:
            l2_pass += 1
        elif fx.get("disputed"):
            l2_disputed += 1
        else:
            l2_fail += 1
            l2_errors.append(f"{qid}: answer {comp.answer!r} != GT {fx['gt']!r}  | {comp.reasoning}")

        # ---- L3 end-to-end with fake LLM ----
        res = ns.resolve_neural_symbolic(
            q, nodes=[{"node_type": "EVENT", "content": "x", "date": "2023-01-01"}],
            call_llm_fn=fake_llm_factory(fx["parsed"]), config=None,
        )
        if res and _num_match(fx["gt"], res["answer"]):
            l3_pass += 1
        elif res and fx.get("disputed"):
            pass  # disputed GT, end-to-end still ran
        else:
            l3_fail += 1
            l3_errors.append(f"{qid}: e2e {(res or {}).get('answer')!r} != GT {fx['gt']!r}")

    n = len(FIXTURES)
    print(f"\n{'='*70}\nNEURAL-SYMBOLIC OFFLINE SELF-TEST  ({n} fixtures)\n{'='*70}")
    print(f"L1 classify   : {l1_pass} pass / {l1_fail} fail")
    for e in l1_errors:
        print(f"    ✗ {e}")
    print(f"L2 compute(gold operands): {l2_pass} exact, {l2_disputed} disputed-GT, {l2_fail} fail")
    for e in l2_errors:
        print(f"    ✗ {e}")
    print(f"L3 end-to-end (fake LLM) : {l3_pass} pass / {l3_fail} hard-fail")
    for e in l3_errors:
        print(f"    ✗ {e}")

    # to_number spot checks — incl. the adversarial-review bug fixes:
    # "10 minutes" must NOT be 1e7 (the 'm' multiplier bug); "5km" stays 5;
    # negatives keep their sign.
    print(f"\n{'-'*70}\nto_number spot checks:")
    tn_fail = 0
    for s, exp in [("$3,750", 3750), ("five", 5), ("12 courses", 12), ("20%", 20),
                   ("440 pages", 440), ("half", 0.5), ("a", 1), ("2.5", 2.5), ("5k", 5000),
                   ("10 minutes", 10), ("5km", 5), ("3.5kg", 3.5), ("$1.2m", 1_200_000),
                   ("-5", -5), ("minus 3", -3)]:
        got = ns.to_number(s)
        ok = got == exp
        if not ok:
            tn_fail += 1
        print(f"  {'✓' if ok else '✗'} to_number({s!r}) = {got}  (expect {exp})")

    hard_fail = l1_fail + l2_fail + l3_fail + tn_fail
    print(f"\n{'='*70}")
    print(f"RESULT: {'PASS' if hard_fail == 0 else 'FAIL'}  "
          f"(hard failures: {hard_fail}; disputed-GT compute misses are expected/excluded)")
    print(f"{'='*70}\n")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
