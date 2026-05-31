# Retention Assistant — Streamlit App

A simple chat app for customer-retention reps. A rep types a question about a
customer; the AI agent answers and, along the way, calls tools (look up the
customer, predict churn, fetch offers, log the call, escalate). The app shows
every tool call so the rep can see exactly what the agent did.


---

## What you see on screen

- A normal chat window (type at the bottom, history scrolls above).
- Under each agent answer, an expandable **"tool calls"** panel showing which
  tools ran, in what order, the inputs they were given, and what they returned.
- A sidebar with the connected tools, the model in use, the API-key status, and
  a button to start a new conversation.

---

## How it fits together

There are two main files:

| File | Role |
|---|---|
| `streamlit_app.py` | The **front end** — the chat UI and how tool calls are displayed. |
| `churn_agent.py` | The **back end** — the agent, the 5 tools, and the loop that talks to OpenAI. |

The app never talks to OpenAI directly. It just hands the user's message to the
agent and renders whatever comes back.

```
   Rep types a message
          │
          ▼
 ┌─────────────────────┐     message + chat history      ┌────────────────────┐
 │   streamlit_app.py  │  ───────────────────────────▶   │   churn_agent.py   │
 │   (chat UI)         │                                 │   Agent.run(...)   │
 │                     │  ◀───────────────────────────   │                    │
 └─────────────────────┘   final reply + full trace      └─────────┬──────────┘
          │                                                         │
          │ shows the answer                          loops, calling tools
          │ + the tool-call panel                                   │
          ▼                                                         ▼
   Rep reads the result                              ┌──────────────────────────┐
                                                     │  OpenAI model decides     │
                                                     │  which tool to call next  │
                                                     └─────────────┬─────────────┘
                                                                   │
                         ┌─────────────────────────────────────────┴───────────┐
                         ▼            ▼              ▼            ▼              ▼
                  lookup_customer  predict_churn  get_offers  log_interaction  escalate
                       │               │
                       ▼               ▼
            cleaned_data_features  churn_model.joblib
                  .csv (mock CRM)    (trained model)
```

---

## The back end (`churn_agent.py`)

**The agent loop.** `Agent.run(message)` sends the conversation to the OpenAI
model with the list of available tools. The model can either reply with text or
ask to call one or more tools. If it asks for tools, the agent runs them, feeds
the results back, and asks the model again. This repeats until the model gives a
final text answer (or a safety limit is reached).

**The tools.** There are five, and the agent picks which to use:
- `lookup_customer` — find a customer's account info (mock CRM from the CSV).
- `predict_churn` — score churn risk using the saved model.
- `get_retention_offers` — list offers for a risk level.
- `log_interaction` — record the call to a log file.
- `escalate_to_supervisor` — hand the case to a human.

**Why it's easy to extend.** Tools register themselves with a `@tool`
decorator, and the agent reads that registry. Adding a new tool is just writing
a new decorated function — the loop and the UI pick it up automatically.

---

## The front end (`streamlit_app.py`)

**Memory between turns.** Streamlit reruns the whole script on every action, so
the app keeps two things in `st.session_state`:
- `turns` — what to draw on screen (each user and agent message).
- `history` — the raw conversation it passes back to the agent, so the agent
  remembers earlier messages (e.g. "what was *that* customer's risk again?").

**Showing the tool calls.** When the agent answers, it also returns the full
list of messages from that turn. The app reads the new messages, pulls out each
tool call and its result (`parse_trace`), and draws them in the expandable panel
(`render_trace`). Order is preserved, so the rep sees the real sequence.

**API key.** The key is read from a local `.env` file when running on your
machine, or from **Streamlit Secrets** when deployed. If no key is found, the
app shows a clear warning instead of failing silently.

---


