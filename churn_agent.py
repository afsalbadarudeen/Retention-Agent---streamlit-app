"""Customer-retention agent backed by the OpenAI API.

Tools register themselves with the ``@tool`` decorator, which also builds their
OpenAI function schema from the function's type hints. The agent loop only ever
talks to the registry, so adding a new tool means writing a new decorated
function -- nothing in ``Agent`` needs to change.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # read OPENAI_API_KEY / OPENAI_MODEL from a local .env if present

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "churn_model.joblib"
DATA_PATH = HERE / "cleaned_data_features.csv"
INTERACTION_LOG = HERE / "interactions_log.jsonl"
ESCALATION_LOG = HERE / "escalations_log.jsonl"
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# --- Tool registry -----------------------------------------------------------
TOOL_REGISTRY: dict[str, dict] = {}

_PY_TO_JSON = {str: "string", int: "integer", float: "number",
               bool: "boolean", dict: "object", list: "array"}


def _json_type(annotation: Any) -> str:
    return _PY_TO_JSON.get(annotation, "string")


def _build_parameters(fn: Callable, descriptions: dict[str, str]) -> dict:
    """Build the OpenAI parameter schema from a function's signature."""
    props, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        annotation = param.annotation if param.annotation is not inspect._empty else str
        jtype = _json_type(annotation)
        prop: dict[str, Any] = {"type": jtype}
        if name in descriptions:
            prop["description"] = descriptions[name]
        if jtype == "object":
            prop["additionalProperties"] = True
        elif jtype == "array":
            prop["items"] = {"type": "string"}
        props[name] = prop
        if param.default is inspect._empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def tool(descriptions: dict[str, str] | None = None) -> Callable:
    """Register a function as a tool. The tool name is the function name and its
    description is the docstring; ``descriptions`` documents individual args."""
    def decorator(fn: Callable) -> Callable:
        TOOL_REGISTRY[fn.__name__] = {
            "fn": fn,
            "schema": {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": _build_parameters(fn, descriptions or {}),
            },
        }
        return fn
    return decorator


# --- Shared resources (loaded once) ------------------------------------------
@lru_cache(maxsize=1)
def _model_artifact() -> dict:
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _customer_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, dtype={"customer_id": str})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# --- Tools -------------------------------------------------------------------
@tool(descriptions={
    "customer_data": "A customer's raw fields (e.g. tenure_months, contract_type, "
                     "monthly_charges, satisfaction_score). Get this from "
                     "lookup_customer; missing fields are imputed automatically."})
def predict_churn(customer_data: dict) -> dict:
    """Score a customer's churn risk with the saved model. Returns
    churn_probability (0-1), risk_tier (Low/Medium/High) and top_risk_factors."""
    if isinstance(customer_data, str):  # the model sometimes passes a JSON string
        customer_data = json.loads(customer_data)

    art = _model_artifact()
    pipe = art["pipeline"]
    num_f, cat_f = art["num_features"], art["cat_features"]

    def _norm(x):
        return x if (x is None or (isinstance(x, float) and pd.isna(x))) \
            else str(x).strip().lower()

    # Normalise messy categorical values (e.g. "MALE" -> "Male").
    d = dict(customer_data)
    for col, mapping in art["norm_maps"].items():
        if col in d and d[col] is not None:
            d[col] = mapping.get(_norm(d[col]), d[col])

    # Rebuild engineered features if the caller didn't supply them.
    tenure = d.get("tenure_months", np.nan)
    tickets = d.get("num_support_tickets", np.nan)
    if d.get("support_intensity") is None:
        d["support_intensity"] = (tickets / (tenure + 1)) \
            if pd.notna(tickets) and pd.notna(tenure) else np.nan
    if d.get("tenure_group") is None:
        d["tenure_group"] = (pd.cut([tenure], bins=art["tenure_bins"],
                                    labels=art["tenure_labels"], right=False)[0]
                             if pd.notna(tenure) else np.nan)

    row = pd.DataFrame([{c: d.get(c, np.nan) for c in num_f + cat_f}])
    for c in cat_f:
        row[c] = row[c].astype(object)

    proba = float(pipe.predict_proba(row)[0, 1])
    tiers = art["risk_tiers"]
    tier = "High" if proba >= tiers["high"] else "Medium" if proba >= tiers["medium"] else "Low"

    # Top risk factors = features that push this customer's prediction up.
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names = prep.get_feature_names_out()
    x = prep.transform(row)
    x = x.toarray()[0] if hasattr(x, "toarray") else np.asarray(x)[0]
    weights = clf.coef_[0] if hasattr(clf, "coef_") \
        else getattr(clf, "feature_importances_", np.zeros(len(names)))
    contrib = weights * x

    def _readable(feature_name):
        raw = feature_name.split("__", 1)[1] if "__" in feature_name else feature_name
        for c in cat_f:
            if raw.startswith(c + "_"):
                return f"{c} = {raw[len(c) + 1:]}"
        return raw

    factors = [_readable(names[i]) for i in np.argsort(contrib)[::-1] if contrib[i] > 0][:3]
    return {"churn_probability": round(proba, 4), "risk_tier": tier,
            "top_risk_factors": factors}


@tool(descriptions={"customer_id": "Customer identifier, e.g. 'TC-004711'."})
def lookup_customer(customer_id: str) -> dict:
    """Look up a customer's account profile in the CRM (mocked from the dataset).
    Returns {found: false} for an unknown id -- do not invent customer data."""
    df = _customer_df()
    cid = str(customer_id).strip().upper()
    hit = df[df["customer_id"].str.upper() == cid]
    if hit.empty:
        return {"found": False, "customer_id": customer_id,
                "message": "No customer matches that id. Confirm the id (format TC-XXXXXX)."}

    rec = hit.iloc[0].to_dict()
    rec.pop("churned", None)  # a live CRM wouldn't hold the future churn label
    profile = {k: (None if (isinstance(v, float) and pd.isna(v)) else v)
               for k, v in rec.items()}
    return {"found": True, "profile": profile}


_OFFER_CATALOG = {
    "High": [
        {"offer_id": "RET-H1", "name": "30% off for 12 months + free premium support",
         "monthly_value_usd": 30, "contract_required": "1 year"},
        {"offer_id": "RET-H2", "name": "Free device upgrade + loyalty credit",
         "monthly_value_usd": 25, "contract_required": "2 year"},
    ],
    "Medium": [
        {"offer_id": "RET-M1", "name": "15% off for 6 months",
         "monthly_value_usd": 12, "contract_required": "none"},
        {"offer_id": "RET-M2", "name": "Free add-on service for 3 months",
         "monthly_value_usd": 8, "contract_required": "none"},
    ],
    "Low": [
        {"offer_id": "RET-L1", "name": "Loyalty thank-you: 50GB bonus data",
         "monthly_value_usd": 3, "contract_required": "none"},
    ],
}


@tool(descriptions={"risk_tier": "One of 'Low', 'Medium', or 'High', usually from "
                                 "predict_churn."})
def get_retention_offers(risk_tier: str) -> dict:
    """Return the retention offers available for a risk tier. Higher risk unlocks
    richer offers."""
    tier = str(risk_tier).strip().title()
    if tier not in _OFFER_CATALOG:
        return {"error": f"Unknown risk_tier '{risk_tier}'. Use Low, Medium, or High.",
                "valid_tiers": list(_OFFER_CATALOG)}
    return {"risk_tier": tier, "offers": _OFFER_CATALOG[tier]}


@tool(descriptions={
    "customer_id": "Customer the interaction concerns.",
    "channel": "Contact channel: chat | phone | email | in_app.",
    "interaction_type": "e.g. retention_call, churn_assessment, complaint, inquiry.",
    "action_taken": "What the agent did, e.g. 'offered RET-H1'.",
    "outcome": "Result, e.g. offer_accepted | offer_declined | resolved | escalated | pending.",
    "agent_id": "Identifier of the agent handling the case.",
    "churn_probability": "Churn probability at the time of the interaction (optional).",
    "risk_tier": "Risk tier at the time of the interaction (optional).",
    "offer_id": "Offer presented, if any (optional).",
    "notes": "Free-text summary of the interaction (optional)."})
def log_interaction(customer_id: str, channel: str, interaction_type: str,
                    action_taken: str, outcome: str,
                    agent_id: str = "agent-retention-bot-v1",
                    churn_probability: float = None, risk_tier: str = None,
                    offer_id: str = None, notes: str = None) -> dict:
    """Write a customer interaction to the interactions log. Call this before
    ending a conversation."""
    record = {
        "schema_version": "1.0",
        "interaction_id": f"INT-{uuid.uuid4().hex[:12]}",
        "customer_id": customer_id,
        "timestamp_utc": _now_iso(),
        "agent_id": agent_id,
        "channel": channel,
        "interaction_type": interaction_type,
        "action_taken": action_taken,
        "outcome": outcome,
        "churn_probability": churn_probability,
        "risk_tier": risk_tier,
        "offer_id": offer_id,
        "notes": notes,
    }
    _append_jsonl(INTERACTION_LOG, record)
    return {"status": "logged", "interaction_id": record["interaction_id"], "record": record}


@tool(descriptions={
    "customer_id": "Customer being escalated (or 'unknown' if unidentified).",
    "reason": "Short reason code, e.g. 'high_value_high_risk', 'customer_requested_human', "
              "'tool_failure', 'out_of_policy', 'unresolved_ambiguity'.",
    "context_summary": "Summary of the conversation, the customer's situation, what was "
                       "attempted, and what the supervisor needs to decide.",
    "priority": "low | medium | high | urgent.",
    "churn_probability": "Latest churn probability, if known (optional).",
    "recommended_action": "What the agent suggests the supervisor do (optional)."})
def escalate_to_supervisor(customer_id: str, reason: str, context_summary: str,
                           priority: str = "medium", churn_probability: float = None,
                           recommended_action: str = None) -> dict:
    """Hand the case to a human supervisor with a context summary. Use for
    high-stakes, out-of-policy, or ambiguous cases, or when the customer asks for
    a human."""
    ticket = {
        "escalation_id": f"ESC-{uuid.uuid4().hex[:10]}",
        "customer_id": customer_id,
        "timestamp_utc": _now_iso(),
        "reason": reason,
        "priority": priority,
        "churn_probability": churn_probability,
        "recommended_action": recommended_action,
        "context_summary": context_summary,
        "status": "open",
    }
    _append_jsonl(ESCALATION_LOG, ticket)
    return {"status": "escalated", "escalation_id": ticket["escalation_id"],
            "priority": priority, "ticket": ticket}


# --- Agent -------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are "RetentionAssistant", an AI agent for a telecom customer-retention team.
Your job: assess churn risk and, where appropriate, present retention offers --
professionally, concisely, and within policy.

Standard workflow:
1. Identify the customer with lookup_customer. If the user gives only a name or a
   vague description (no valid TC-XXXXXX id), ASK for the customer id -- never guess
   or fabricate account data.
2. Assess risk with predict_churn using the looked-up profile.
3. If retention is warranted, fetch options with get_retention_offers for the
   returned risk_tier and recommend one suited to the customer.
4. ALWAYS record the interaction with log_interaction before you finish.

Escalate to a human via escalate_to_supervisor when ANY of these hold:
- High churn risk on a high-value / long-tenured customer (judgment + business impact).
- The customer explicitly asks for a human, is angry, or threatens to leave now.
- A tool fails or returns found=false / an error and you cannot proceed reliably.
- The request is outside retention scope (billing disputes, legal, cancellations).
- The input stays ambiguous after you have asked one clarifying question.

Handling ambiguity: prefer ONE concise clarifying question over an assumption. If
ambiguity blocks progress and cannot be resolved, escalate with a clear summary.
Think step by step and use tools rather than guessing facts."""


class Agent:
    """Runs the tool-calling loop against whatever tools are in the registry."""

    def __init__(self, model: str = MODEL_NAME, max_steps: int = 8, verbose: bool = False):
        self.model = model
        self.max_steps = max_steps
        self.verbose = verbose
        self.client = OpenAI()

    @staticmethod
    def _tool_specs() -> list[dict]:
        # Rebuilt from the registry on every call, so new tools show up for free.
        return [{"type": "function", "function": t["schema"]} for t in TOOL_REGISTRY.values()]

    @staticmethod
    def dispatch(name: str, arguments: dict) -> dict:
        """Run a tool by name. A failing tool returns an error dict rather than
        raising, so one bad call doesn't break the loop."""
        entry = TOOL_REGISTRY.get(name)
        if entry is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            return entry["fn"](**arguments)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def run(self, user_message: str, history: list[dict] | None = None) -> dict:
        """Answer a message, calling tools as needed. Returns the final reply and
        the full message list (handy for inspecting the tool trace)."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += history or []
        messages.append({"role": "user", "content": user_message})

        for _ in range(self.max_steps):
            response = self.client.chat.completions.create(
                model=self.model, messages=messages,
                tools=self._tool_specs(), tool_choice="auto", temperature=0)
            msg = response.choices[0].message

            if not msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content})
                return {"reply": msg.content, "messages": messages}

            messages.append({
                "role": "assistant", "content": msg.content,
                "tool_calls": [{"id": tc.id, "type": "function",
                                "function": {"name": tc.function.name,
                                             "arguments": tc.function.arguments}}
                               for tc in msg.tool_calls]})

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if self.verbose:
                    print(f"[tool] {tc.function.name}({args})")
                result = self.dispatch(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result, default=str)})

        return {"reply": "Max reasoning steps reached; escalate if unresolved.",
                "messages": messages}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python churn_agent.py "<your question>"')
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set.")
    print(Agent().run(" ".join(sys.argv[1:]))["reply"] or "")
