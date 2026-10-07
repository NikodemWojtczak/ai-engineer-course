# AI Engineer Course — hands-on notebooks

A video course for the team, built on the *Senior AI Engineer Skill Roadmap* (six levels). One notebook per level.
Every concept is shown three times: **A** — with the plain OpenAI API, from scratch; **B** — step by step in LangGraph;
**C** — in a high-level framework (Agno).

| Level | Notebook | Status |
|---|---|---|
| 1 · Model foundations & the AI Gateway | [`notebooks/level_1_model_foundations_and_gateway.ipynb`](notebooks/level_1_model_foundations_and_gateway.ipynb) | **ready** — verified end to end against the real API on 2026-10-07 |
| 2 · Tool calling & structured output | — | planned |
| 3 · Context & memory | — | planned |
| 4 · MCP, A2A & Skills | — | planned |
| 5 · Evaluation and tracing | — | planned |
| 6 · Security | — | planned |

## Quick start

You need [`uv`](https://docs.astral.sh/uv/) (it installs Python 3.12 for you if needed) and an OpenAI API key.

```bash
git clone https://github.com/NikodemWojtczak/ai-engineer-course.git && cd ai-engineer-course
uv sync                                   # Python 3.12 + pinned packages from uv.lock (openai, langgraph, agno, jupyterlab …)
cp .env.example .env                      # then put your OPENAI_API_KEY in .env  (the file is git-ignored — never commit it)
uv run jupyter lab notebooks/             # open level_1_… and run the cells from the top
```

The first cell checks for the key and stops with a clear message if it is missing.
Prefer VS Code? Open the folder, pick the `.venv` kernel and run the notebook there — same thing.

### What the run needs

- **Models.** The key must have access to `gpt-4.1-mini`, `gpt-4o-mini`, `gpt-6-luna`, `gpt-6.1-sol` and `text-embedding-3-small`.
  They are used through aliases set in Part 0 (`SMALL`, `REASONING`, `LARGE`, `EMBEDDING`), so you can swap them in one place.
- **Cost.** A full run of Level 1 costs about **0.30–0.50 USD**. Most of it is the long-context experiment in Part 7;
  lower `N_LINES` or `TRIALS_LONG` there to make it cheaper.
- **Time.** About 3 minutes of API calls for the whole notebook (the author's last full run: 162 s), plus one or two minutes
  the first time the optional LiteLLM section downloads its package.
- **Ports.** The notebook runs a small toy gateway inside the kernel on `127.0.0.1:4010`, and the optional LiteLLM section
  listens on `127.0.0.1:4020`. Both must be free.
- **The optional LiteLLM section (Part 3)** starts a real LiteLLM gateway with
  `uvx --python 3.12 --from 'litellm[proxy]==1.104.0' litellm`. It needs `uvx` on your `PATH` and network access.
  If `uvx` is missing the cells skip themselves; if LiteLLM does not start, the notebook says so, points you to
  `notebooks/litellm_demo/litellm.log`, and moves on. Nothing later depends on it.

## How to read the notebook

- Run it **top to bottom**; later parts reuse functions and the gateway log from earlier parts.
  To run it again, use *Kernel → Restart Kernel and Run All Cells*.
- The notebook comes **with the outputs of the author's run** against the real API (2026-10-07), so you can read it — answers,
  tables and charts — without running it, on GitHub or in Jupyter. When you run it yourself the numbers will differ a little,
  because the model samples and prices change; the arguments never depend on an exact value.
- Three kinds of cells come back in every part:
  **🔮 Predict** — guess the result before running the next cell;
  **💥 Break it** — a working example is deliberately broken, then repaired;
  **🧠 Check yourself** — a question with the answer folded underneath.
  Each part ends with **📌 Takeaways**.
- The examples use a fictional bank, **NovaBank**, and fictional customers. Keep it that way: everything the notebook sends
  goes to the OpenAI API, so never paste real customer data, internal documents or credentials into a cell.
- "The platform" / "the gateway" in the text means the OpenAI-compatible gateway (LiteLLM) your applications call at work.
  The notebook builds a toy version of it in Part 3 so you can see what such a gateway does with a request.

## Level 1 at a glance

| Part | What you will see |
|---|---|
| 0 · Setup and the one mental model | the first call; the model as a stateless function; the four message roles |
| 1 · Tokens & context window | what a token is, how the model writes one token at a time (log-probabilities), four ways the window fills up, what a request costs — and the same support assistant priced on five models |
| 2 · Model selection | the price gap, plain vs. thinking models, three tasks measured on three models, reasoning effort, two speed numbers, how to choose |
| 3 · The AI Gateway | what "OpenAI-compatible" means; a toy gateway built in the notebook (aliases, log, guardrails before and after the call, embeddings, who spent what), then the real LiteLLM |
| 4 · Messages | system, user, assistant, tool; the model has no memory, the history is only data; roles as levels of trust; the same roles in the frameworks |
| 5 · Prompt structure | one task, six prompt versions scored on the same tickets; examples that disagree with the rules; prompts are code — version them |
| 6 · Prompt caching | a long stable prefix measured with `cached_tokens`; a timestamp at the top breaks it; two layouts of one request compared |
| 7 · Failure modes | hallucination, lost in the middle, instruction drift — each one measured, then side by side |
| 8 · Three layers | one chatbot written three times (plain API, LangGraph, Agno) behind the same gateway; one failure in three layers; a model swap without touching the applications |
| 9 · Recap | the mental model once more, glossary, quiz, what comes next |

## What is in this repository

```
notebooks/                 the notebooks (with the author's outputs) and img/ with their diagrams
pyproject.toml, uv.lock    the pinned environment the notebooks were verified with; .python-version pins Python 3.12
.env.example               template for your .env
```

The notebooks are generated from sources kept in a separate repository, so edit them only for your own experiments
and send corrections to the author. After you run a notebook, `git status` shows it as modified (your outputs replaced
the author's); `git checkout -- notebooks/` restores the published copy, and `git pull` brings new levels as they are published.
