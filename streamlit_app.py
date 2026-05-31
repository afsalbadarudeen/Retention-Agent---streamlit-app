"""
streamlit_app.py — chat UI for the churn-retention agent.

- Chat interface for retention reps (multi-turn, with memory).
- Every tool call is shown visibly: name, order, arguments, and what it returned.
- Wires directly into the churn_agent backend (same TOOL_REGISTRY + Agent loop).

Run locally:  streamlit run streamlit_app.py
Deploy:       see DEPLOY.md (Streamlit Community Cloud).
"""
import json
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # load OPENAI_API_KEY / OPENAI_MODEL from a local .env (no-op on Cloud)

st.set_page_config(page_title="Churn Retention Assistant", page_icon="📞", layout="wide")


# ----------------------------------------------------------------------
# Credentials: env var (local/.env) OR Streamlit secrets (Cloud).
# Must be resolved BEFORE importing the backend, which reads the model name
# at import time.
# ----------------------------------------------------------------------
def _secret(name: str, default=None):
    if os.environ.get(name):
        return os.environ[name]
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return default


_api_key = _secret("OPENAI_API_KEY")
if _api_key:
    os.environ["OPENAI_API_KEY"] = _api_key
os.environ.setdefault("OPENAI_MODEL", _secret("OPENAI_MODEL", "gpt-4o"))

import churn_agent as ca  # noqa: E402  (imported after env is set)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def parse_trace(messages: list[dict]) -> list[dict]:
    """Turn raw chat messages into an ordered [{name, args, result}] tool trace."""
    pending, order = {}, []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except Exception:
                    args = tc["function"]["arguments"]
                pending[tc["id"]] = {"name": tc["function"]["name"], "args": args, "result": None}
                order.append(tc["id"])
        elif m.get("role") == "tool":
            tid = m.get("tool_call_id")
            if tid in pending:
                try:
                    pending[tid]["result"] = json.loads(m["content"])
                except Exception:
                    pending[tid]["result"] = m["content"]
    return [pending[i] for i in order]


def render_trace(trace: list[dict]) -> None:
    if not trace:
        st.caption("🟢 Answered directly — no tools were needed.")
        return
    names = " → ".join(f"`{s['name']}`" for s in trace)
    with st.expander(f"🔧 {len(trace)} tool call(s):  {names}", expanded=False):
        for i, step in enumerate(trace, 1):
            st.markdown(f"**Step {i} — `{step['name']}`**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Arguments")
                st.json(step["args"], expanded=False)
            with c2:
                st.caption("Returned")
                st.json(step["result"], expanded=False)
            if i < len(trace):
                st.divider()


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "turns" not in st.session_state:
    st.session_state.turns = []      # [{role, content, trace?}] for display
if "history" not in st.session_state:
    st.session_state.history = []    # raw messages (no system) passed to the backend


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("📞 Retention Assistant")
    st.caption("AI copilot for customer-retention reps.")

    if _api_key:
        st.success("OpenAI key loaded")
    else:
        st.error("No OPENAI_API_KEY — set it in Streamlit **Secrets** (or a local .env).")
    st.write(f"**Model:** `{os.environ.get('OPENAI_MODEL')}`")

    st.subheader("Connected tools")
    for name, entry in ca.TOOL_REGISTRY.items():
        desc = (entry["schema"]["description"] or "").split(".")[0]
        st.markdown(f"- **`{name}`** — {desc.strip()}.")

    st.subheader("Try asking")
    st.markdown(
        "- *What's the churn risk for TC-004711?*\n"
        "- *Assess TC-000692 and recommend an offer, then log it.*\n"
        "- *Compare TC-004711 and TC-000066 — who do we prioritize?*\n"
        "- *Customer TC-004711 demands a manager right now.*"
    )

    if st.button("🧹 New conversation", use_container_width=True):
        st.session_state.turns = []
        st.session_state.history = []
        st.rerun()


# ----------------------------------------------------------------------
# Main chat
# ----------------------------------------------------------------------
st.title("Customer Churn Retention Assistant")
st.caption("Chat with the agent. Tool calls (what was called, in what order, and what "
           "they returned) are shown inline under each answer.")

for turn in st.session_state.turns:
    with st.chat_message(turn["role"]):
        if turn["role"] == "assistant" and "trace" in turn:
            render_trace(turn["trace"])
        st.markdown(turn["content"])

prompt = st.chat_input("Ask about a customer (e.g. 'Is TC-004711 likely to churn?')")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.turns.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not _api_key:
            reply, trace = "⚠️ I can't run without an OpenAI API key. Add it in Secrets.", []
            st.markdown(reply)
        else:
            with st.spinner("Thinking & calling tools…"):
                agent = ca.Agent(model=os.environ.get("OPENAI_MODEL"), verbose=False)
                n_hist = len(st.session_state.history)
                try:
                    result = agent.run(prompt, history=st.session_state.history)
                    new_messages = result["messages"][1 + n_hist:]   # drop system + prior turns
                    trace = parse_trace(new_messages)
                    st.session_state.history = result["messages"][1:]  # persist (minus system)
                    reply = result["reply"] or "(no reply)"
                except Exception as exc:
                    trace = []
                    reply = f"❌ Backend error: `{type(exc).__name__}: {exc}`"
            render_trace(trace)
            st.markdown(reply)

    st.session_state.turns.append({"role": "assistant", "content": reply, "trace": trace})
