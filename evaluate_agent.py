"""
evaluate_agent.py — evaluation pipeline for the churn retention agent.

What it measures
----------------
1. Automated metrics (deterministic, no LLM):
   - tool_selection : F1 of called-vs-expected tool set, plus forbidden-tool check
   - completeness   : fraction of required content patterns present in the reply
   - latency        : wall-clock seconds for the agent to answer

2. LLM-as-judge with an ANCHORED rubric (discrete labels with definitions, not a
   vague 1-10 scale) scoring four dimensions:
   - factual_correctness : correct | minor_error | major_error
   - tool_appropriateness: appropriate | suboptimal | inappropriate
   - actionability       : actionable | partially_actionable | not_actionable
   - hallucination       : none | minor | severe   (none is best)

"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

import churn_agent as ca

load_dotenv()

HERE = Path(__file__).resolve().parent
CASES_PATH = HERE / "eval_cases.json"
RESULTS_PATH = HERE / "eval_results.json"
import os
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")

# ----------------------------------------------------------------------
# Anchored rubric — the scoring contract handed to the judge
# ----------------------------------------------------------------------
RUBRIC = {
    "factual_correctness": {
        "correct": "Every factual claim (probabilities, tiers, offers, customer facts) matches the tool outputs; no contradictions.",
        "minor_error": "Substantively correct but one non-critical imprecision (rounding, slight rewording of a tier/figure).",
        "major_error": "A claim contradicts a tool output or states a material fact that the tools did not support.",
    },
    "tool_appropriateness": {
        "appropriate": "Called exactly the tools the task needed, in a sensible order; none missing, none gratuitous.",
        "suboptimal": "Task addressed, but a helpful tool was skipped or an unnecessary tool was called.",
        "inappropriate": "Wrong tool(s), skipped a required tool, or acted/answered without the lookup/prediction it needed.",
    },
    "actionability": {
        "actionable": "Gives a clear answer or next step the user can act on immediately.",
        "partially_actionable": "Relevant but vague, or missing a concrete recommendation/next step.",
        "not_actionable": "Unclear, generic, or does not address the request.",
    },
    "hallucination": {
        "none": "Every specific fact is grounded in a tool result or the user's input.",
        "minor": "A small unsupported detail that does not change the decision.",
        "severe": "Fabricated customer data, offers, probabilities, or capabilities.",
    },
}

# label -> numeric (for aggregation only; the label is the primary signal)
LABEL_SCORES = {
    "factual_correctness": {"correct": 1.0, "minor_error": 0.5, "major_error": 0.0},
    "tool_appropriateness": {"appropriate": 1.0, "suboptimal": 0.5, "inappropriate": 0.0},
    "actionability": {"actionable": 1.0, "partially_actionable": 0.5, "not_actionable": 0.0},
    "hallucination": {"none": 1.0, "minor": 0.5, "severe": 0.0},
}


# ----------------------------------------------------------------------
# Run one case through the agent and capture a full trace
# ----------------------------------------------------------------------
def extract_trace(messages: list[dict]) -> tuple[list[str], list[dict]]:
    """Return (ordered tool names called, [{name, args, result}, ...])."""
    pending: dict[str, dict] = {}
    order: list[str] = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                pending[tc["id"]] = {"name": tc["function"]["name"],
                                     "args": tc["function"]["arguments"], "result": None}
                order.append(tc["function"]["name"])
        elif m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if tcid in pending:
                pending[tcid]["result"] = m.get("content")
    return order, list(pending.values())


def run_case(case: dict) -> dict:
    agent = ca.Agent(verbose=False)
    t0 = time.perf_counter()
    out = agent.run(case["input"])
    latency = time.perf_counter() - t0
    tools_called, trace = extract_trace(out["messages"])
    return {"reply": out["reply"] or "", "tools_called": tools_called,
            "trace": trace, "latency_s": round(latency, 3)}


# ----------------------------------------------------------------------
# Automated metrics
# ----------------------------------------------------------------------
def tool_selection_metrics(expected: list[str], forbidden: list[str],
                           actual: list[str]) -> dict:
    exp, act, forb = set(expected), set(actual), set(forbidden)
    tp = len(exp & act)
    precision = tp / len(act) if act else (1.0 if not exp else 0.0)
    recall = tp / len(exp) if exp else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else (
        1.0 if not exp and not act else 0.0)
    forbidden_hit = sorted(forb & act)
    score = 0.0 if forbidden_hit else f1
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "exact_match": exp == act,
            "forbidden_called": forbidden_hit, "score": round(score, 3)}


def completeness_metric(patterns: list[str], reply: str) -> dict:
    hits = [p for p in patterns if re.search(p, reply, re.IGNORECASE)]
    frac = len(hits) / len(patterns) if patterns else 1.0
    missed = [p for p in patterns if p not in hits]
    return {"score": round(frac, 3), "matched": len(hits),
            "total": len(patterns), "missed_patterns": missed}


# ----------------------------------------------------------------------
# LLM-as-judge
# ----------------------------------------------------------------------
def build_judge_prompt(case: dict, run: dict) -> str:
    trace_str = json.dumps(
        [{"tool": t["name"], "args": t["args"], "result": t["result"]}
         for t in run["trace"]], indent=2, default=str)[:6000]
    return f"""\
You are a strict QA evaluator for a customer-retention AI agent. Score the agent
turn ONLY against the anchored rubric below. For each dimension choose EXACTLY
one label and justify it in one sentence grounded in the trace.

## User request
{case['input']}

## Expected behavior (ground truth for this case)
{case['expected_behavior']}

## Tools the agent actually called (with results)
{trace_str if run['trace'] else "(no tools were called)"}

## Agent's final reply to the user
{run['reply']}

## Anchored rubric — pick one label per dimension
{json.dumps(RUBRIC, indent=2)}

Return ONLY a JSON object with these keys:
{{
  "factual_correctness": "<label>",
  "tool_appropriateness": "<label>",
  "actionability": "<label>",
  "hallucination": "<label>",
  "rationale": "<2-3 sentences citing specifics>"
}}"""


def judge_case(case: dict, run: dict, client) -> dict:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": build_judge_prompt(case, run)}])
    verdict = json.loads(resp.choices[0].message.content)
    verdict["scores"] = {
        dim: LABEL_SCORES[dim].get(str(verdict.get(dim, "")).strip().lower())
        for dim in LABEL_SCORES}
    return verdict


# ----------------------------------------------------------------------
# Aggregation & reporting
# ----------------------------------------------------------------------
def _pctile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[idx]


def aggregate(results: list[dict], use_judge: bool) -> dict:
    tool_scores = [r["metrics"]["tool_selection"]["score"] for r in results]
    comp_scores = [r["metrics"]["completeness"]["score"] for r in results]
    lats = [r["run"]["latency_s"] for r in results]
    exact = sum(r["metrics"]["tool_selection"]["exact_match"] for r in results)
    forb = sum(bool(r["metrics"]["tool_selection"]["forbidden_called"]) for r in results)

    agg = {
        "n_cases": len(results),
        "tool_selection": {"mean_f1_score": round(statistics.mean(tool_scores), 3),
                           "exact_match_rate": round(exact / len(results), 3),
                           "forbidden_violations": forb},
        "completeness": {"mean": round(statistics.mean(comp_scores), 3)},
        "latency_s": {"mean": round(statistics.mean(lats), 2),
                      "p50": round(_pctile(lats, 0.5), 2),
                      "p95": round(_pctile(lats, 0.95), 2),
                      "max": round(max(lats), 2)},
    }
    if use_judge:
        jdims = {}
        for dim in LABEL_SCORES:
            vals = [r["judge"]["scores"][dim] for r in results
                    if r.get("judge") and r["judge"]["scores"].get(dim) is not None]
            labels = [r["judge"].get(dim) for r in results if r.get("judge")]
            jdims[dim] = {"mean_score": round(statistics.mean(vals), 3) if vals else None,
                          "label_counts": {l: labels.count(l) for l in set(labels)}}
        agg["judge"] = jdims
    return agg


def print_report(results: list[dict], agg: dict, use_judge: bool) -> None:
    print("\n" + "=" * 92)
    print("PER-CASE RESULTS")
    print("=" * 92)
    hdr = f"{'case_id':<30}{'category':<18}{'tool_f1':>8}{'compl':>7}{'lat_s':>7}"
    if use_judge:
        hdr += f"  {'fact':<12}{'tool_apt':<14}{'action':<20}{'halluc':<8}"
    print(hdr)
    print("-" * 92)
    for r in results:
        ts = r["metrics"]["tool_selection"]
        line = (f"{r['id']:<30}{r['category']:<18}{ts['score']:>8.2f}"
                f"{r['metrics']['completeness']['score']:>7.2f}{r['run']['latency_s']:>7.2f}")
        if use_judge and r.get("judge"):
            j = r["judge"]
            line += (f"  {j.get('factual_correctness',''):<12}"
                     f"{j.get('tool_appropriateness',''):<14}"
                     f"{j.get('actionability',''):<20}{j.get('hallucination',''):<8}")
        flag = "  ⚠ forbidden!" if ts["forbidden_called"] else ""
        print(line + flag)

    print("\n" + "=" * 92)
    print("AGGREGATE")
    print("=" * 92)
    print(json.dumps(agg, indent=2))


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    args = ap.parse_args()

    suite = json.loads(CASES_PATH.read_text())
    cases = suite["cases"][: args.limit] if args.limit else suite["cases"]
    use_judge = not args.no_judge

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set — the agent and judge need it.")

    judge_client = None
    if use_judge:
        from openai import OpenAI
        judge_client = OpenAI()

    results = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} ({case['category']}) ...", flush=True)
        run = run_case(case)
        metrics = {
            "tool_selection": tool_selection_metrics(
                case["expected_tools"], case.get("forbidden_tools", []), run["tools_called"]),
            "completeness": completeness_metric(case.get("must_include", []), run["reply"]),
        }
        rec = {"id": case["id"], "category": case["category"],
               "input": case["input"], "run": run, "metrics": metrics}
        if use_judge:
            try:
                rec["judge"] = judge_case(case, run, judge_client)
            except Exception as exc:
                rec["judge"] = {"error": f"{type(exc).__name__}: {exc}",
                                "scores": {d: None for d in LABEL_SCORES}}
        results.append(rec)

    agg = aggregate(results, use_judge)
    print_report(results, agg, use_judge)

    RESULTS_PATH.write_text(json.dumps(
        {"suite": suite["suite"], "judge_model": JUDGE_MODEL if use_judge else None,
         "aggregate": agg, "cases": results}, indent=2, default=str))
    print(f"\nSaved detailed results -> {RESULTS_PATH.name}")


if __name__ == "__main__":
    main()
