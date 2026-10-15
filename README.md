# AI Engineer Course — hands-on notebooks

A video course for the team, built on the *Senior AI Engineer Skill Roadmap* (six levels). One notebook per level.
Every concept is shown three times: **A** — with the plain OpenAI API, from scratch; **B** — with LangChain and LangGraph;
**C** — in a high-level framework (Agno).

| Level | Notebook | Status |
|---|---|---|
| 1 · Model foundations & the AI Gateway | [`level_1_model_foundations_and_gateway.ipynb`](notebooks/level_1_model_foundations_and_gateway.ipynb) | **ready** — verified end to end against the real API on 2026-10-13 |
| 2 · Tool calling & structured output | [`level_2_tool_calling_and_structured_output.ipynb`](notebooks/level_2_tool_calling_and_structured_output.ipynb) | **ready** — verified end to end against the real API on 2026-10-14 |
| 3 · Memory and retrieval | [`level_3_memory_and_retrieval.ipynb`](notebooks/level_3_memory_and_retrieval.ipynb) | **ready** — verified end to end against the real API on 2026-10-14 |
| 4 · MCP, A2A and Skills | [`level_4_mcp_a2a_and_skills.ipynb`](notebooks/level_4_mcp_a2a_and_skills.ipynb) | **ready** — verified end to end against the real API on 2026-10-14 |
| 5 · Evals and tracing | [`level_5_evals_and_tracing.ipynb`](notebooks/level_5_evals_and_tracing.ipynb) | **ready** — verified end to end against the real API on 2026-10-14 |
| 6 · Attack and harden | [`level_6_attack_and_harden.ipynb`](notebooks/level_6_attack_and_harden.ipynb) | **ready** — verified end to end against the real API on 2026-10-14 |

Take the levels in order. Each notebook runs on its own: from Level 2 on, what it needs from the earlier levels (the toy
gateway, the NovaBank data, the agent loop) comes from its helper file `level_N_helpers.py`, next to the notebook. The text
assumes you have read the levels before it: it does not explain again what an earlier level explained, it says where, e.g.
"(Level 2, T 1.5)".

## Quick start

You need [`uv`](https://docs.astral.sh/uv/) (it installs Python 3.12 for you if needed) and an OpenAI API key.

```bash
git clone https://github.com/NikodemWojtczak/ai-engineer-course.git && cd ai-engineer-course
uv sync                                   # Python 3.12 + pinned packages from uv.lock (openai, langgraph, agno, jupyterlab …)
cp .env.example .env                      # then put your OPENAI_API_KEY in .env  (the file is git-ignored — never commit it)
uv run jupyter lab notebooks/             # open a notebook (start with level_1_…) and run the cells from the top
```

The first cell checks for the key and stops with a clear message if it is missing.
Prefer VS Code? Open the folder, pick the `.venv` kernel and run the notebook there — same thing.

### What the run needs

- **Models.** The key must have access to `gpt-4.1-mini`, `gpt-4o-mini`, `gpt-6-luna`, `gpt-6.1-sol` and `text-embedding-3-small`.
  They are used through aliases (`SMALL`, `REASONING`, `LARGE`, `EMBEDDING`), set in Part 0 of Level 1 and in
  `level_N_helpers.py` of Levels 2–6, so you can swap them in one place per level. Level 6 also calls OpenAI's moderation
  model `omni-moderation-latest`, which is free.
- **Cost, time and ports.** What one full run took when the author ran it:

  | Level | Cost | Time | Ports on `127.0.0.1` |
  |---|---|---|---|
  | 1 | 0.30–0.50 USD | about 3 min | 4010, and 4020 for the optional LiteLLM section |
  | 2 | 0.02 USD | under 2 min | 5120 |
  | 3 | 0.02 USD | about 2 min | 5130 |
  | 4 | 0.03 USD | about 1 min | 5140–5149 |
  | 5 | 0.12 USD | about 4 min | 5150 |
  | 6 | 0.01 USD | about 2 min | 5160 |

  Most of Level 1's cost is the long-context experiment in Part 7; lower `N_LINES` or `TRIALS_LONG` there to make it cheaper.
  Every notebook runs its own toy gateway inside the kernel (Level 4 also two MCP servers and three A2A agents), each level on
  its own ports, so two levels can run side by side. The last cell of every notebook stops what it started and prints what
  your run cost.
- **The optional LiteLLM section of Level 1 (Part 3)** starts a real LiteLLM gateway with
  `uvx --python 3.12 --from 'litellm[proxy]==1.104.0' litellm`. It needs `uvx` on your `PATH` and network access; the first
  time it adds a minute or two while the package downloads. If `uvx` is missing the cells skip themselves; if LiteLLM does
  not start, the notebook says so, points you to `notebooks/litellm_demo/litellm.log`, and moves on. Nothing later depends on it.
- **No other accounts.** Level 5 keeps its traces in memory, so you need no LangSmith, Langfuse or Phoenix account.
  Agno reports usage to its own servers by default; every notebook switches that off with `telemetry=False`, and Level 5
  also sets `AGNO_TELEMETRY=false`, because Agno's judge eval builds an agent of its own that ignores the flag.

## How to read the notebooks

- Run a notebook **top to bottom**; later parts reuse functions and the gateway log from earlier parts.
  To run it again, use *Kernel → Restart Kernel and Run All Cells*.
- The notebooks come **with the outputs of the author's run** against the real API (one full run per level: Level 1 on
  2026-10-13, Levels 2 and 4–6 on 2026-10-14, Level 3 on 2026-10-15), so you can read them — answers, tables and charts — without running them, on
  GitHub or in Jupyter. Every printed value says what it is: a label before a short value, a `=== title ===` header above
  a block or a table. When you run one yourself the numbers will differ a little, because the model samples and prices
  change; the text says "when we ran it" wherever it quotes a measured result, and the arguments never depend on an exact
  value.
- **Levels 2–6 put the theory next to the code.** Each part between Part 0 and the recap opens with a short
  introduction. Then come short theory sections (T 1.1, T 1.2, …), each followed right away by the examples that show it
  (Example 1.1, …). When a cell prints an object, it prints it whole, and the text under the output says what to look at.
- From Level 2 on, Part 0 lists the data and the functions the notebook uses, and in which part; the functions that the
  notebook needs but does not teach are in `level_N_helpers.py` next to it, and the first cell imports them by name.
- Two kinds of cells come back through the parts:
  **🔮 Predict** — guess the result before running the next cell;
  **💥 Break it** — a working example is deliberately broken, then repaired.
  Level 1 also has **🧠 Check yourself** — a question with the answer folded underneath. Each part ends with
  **📌 Takeaways**.
- In Level 1, after a definition — a tool, a schema, a helper, a prompt, the test data — a few lines marked `# Try it`
  often run it once, without the model, so you see what goes in and what comes out before an agent uses it. A section
  number in brackets, such as "(0.5)", tells you where a name you meet was defined.
- From Level 2 on, each idea is built by hand first and then with LangChain + LangGraph and with Agno. **Open the box** shows what each
  version actually sent through the gateway, and **New word, same idea** maps the framework names to what you built by hand.
- The examples use a fictional bank, **NovaBank**, and fictional customers. Keep it that way: everything the notebook sends
  goes to the OpenAI API, so never paste real customer data, internal documents or credentials into a cell.
- "The platform" / "the gateway" in the text means the OpenAI-compatible gateway (LiteLLM) your applications call at work.
  Level 1 builds a toy version of it in Part 3 so you can see what such a gateway does with a request; every later level
  sends all its calls through the same toy gateway.
- **Level 6 attacks only its own toy agent.** The "secret" is a made-up canary string, every address uses the `.example`
  domain, a "sent" email lands in a Python list, and the payloads only move the canary between the notebook's own fixtures.
  Do not point these attacks at a real system without permission.

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

## Level 2 at a glance

| Part | What you will see |
|---|---|
| 0 · Setup | the toy gateway and the NovaBank data from Level 1; a toy bank with four tools |
| 1 · One tool call by hand | the schema is all the model knows; the model writes a call, your code runs it, the result goes back by id; a `tool` message that answers nothing; which models call tools |
| 2 · The agent loop | the loop in eleven lines; a tool that raises; two tools in a row and several at once; a loop that never ends and the step limit; what `tool_choice` does |
| 3 · Structured output | from free text to JSON mode to a strict JSON schema; a schema strict mode rejects; Pydantic; a failed validation sent back to the model |
| 4 · Tool design | look-alike tools — the description decides; typed arguments and enums; errors the model can act on; a confirmation step before a refund; what the tool schemas cost on every call |
| 5 · Three layers | one support agent written three times (plain API, LangChain + LangGraph, Agno); what each one sent through the gateway; one failure in three layers |
| 6 · Recap | the one idea once more, glossary, clean up |

## Level 3 at a glance

| Part | What you will see |
|---|---|
| 0 · Setup | the gateway from Level 1, now keeping every request body; the handbook and twenty policy paragraphs; ten questions with known answers |
| 1 · Short-term memory | a turn that needs turn 1; the history is data you send; trimming that forgets; a summary that drops the facts and one that keeps them; LangGraph's checkpointer and LangChain's summary middleware; Agno's sessions and its summary after every turn; what reached the model on turn 6 |
| 2 · Long-term memory | a table of facts with their source; one sentence that poisons every later session; a label in the prompt vs a rule in code; LangGraph's store; Agno's user memories, the memory manager's calls as call trees, and what they do not record |
| 3 · Retrieval | the whole corpus vs three passages; what an embedding sees (king, queen, man, woman, and five sentences); embeddings, BM25 and hybrid search measured with recall@3; a miss you cannot see in the answer; a reranker; search as a tool in LangChain and Agno |
| 4 · Caching | theory only: the order of the blocks in a request, the most stable first, so that the provider's prompt cache can reuse the start |
| 5 · Recap | the one idea once more, glossary, clean up |

## Level 4 at a glance

| Part | What you will see |
|---|---|
| 0 · Setup | the gateway, the bank and the agent loop from Level 2, and a window into every request |
| 1 · Why protocols | three boundaries — another team's tools, another team's agent, procedures kept in files — and why a standard turns N × M pieces of glue into N + M |
| 2 · MCP | our tools as an MCP server; the protocol by hand over HTTP, and a tool error vs a protocol error; one changed sentence in a server's tool description that books a refund a hundred times too large, and how to catch it; resources and prompts; LangChain's `MCPAdapter` and Agno's `MCPTools`; an async tool called from sync code |
| 3 · A2A | another team's agent called by hand; a card that promises too much; the A2A SDK around a LangGraph agent; Agno's AgentOS; an Agno client that silently drops the SDK server's answer; who pays for which call |
| 4 · Skills | an agent that does not know the procedure; a catalog in the prompt and the steps on demand; a look-alike skill; the same in LangChain + LangGraph; Agno's built-in skills measured against our version |
| 5 · Putting it together | one request that uses an MCP tool, the lending agent and a skill; which one to choose when |
| 6 · Recap | the one idea once more, glossary, clean up |

## Level 5 at a glance

| Part | What you will see |
|---|---|
| 0 · Setup | the gateway, the NovaBank tickets and handbook, and the support agent from Level 2; an eval set of 30 tasks of four kinds, with references read from the handbook |
| 1 · The first evaluator, by hand | an evaluator is a function; fourteen tickets, one number and its n; a grader with a bug; a grader for "I don't know" |
| 2 · A model as the judge | a rubric of observable conditions; the judge measured on answers with a known verdict (TPR and TNR); the repair: give the judge the source; position bias; from the judge's number to the true number; the same judge in openevals and Agno |
| 3 · Trajectory evals | which tools, in which order: four match modes; 39 is not "39.0"; a step whose order is a rule; agentevals for LangChain and Agno's `ReliabilityEval`, which does not look at the order |
| 4 · Outcome evals and pass^k | a booking tool that silently loses every fourth booking, and the only grader that sees it; a grader with a bug; pass@k vs pass^k; the repair: read after write |
| 5 · Tracing | one trace id from the agent to the gateway; a span for every step; the forgotten inject; LangChain and Agno with an instrumentor; the same span in three dialects |
| 6 · The harness | three agents on the same 30 tasks, twice, in one table with cost and time per run; the model behind an alias changes; a grader that crashes; what to look at first when a number drops |
| 7 · Recap | the one idea once more, glossary, clean up |

## Level 6 at a glance

| Part | What you will see |
|---|---|
| 0 · Setup | the safety rules; a toy world (inbox, files, outbox, refunds) and its tools; a stub model that always gives in, next to the live ones; the attack success rate |
| 1 · Direct injection | resistance per attempt vs per attacker; an attack ladder measured on two models; what a leak reaches; a secret in the prompt is a secret in the answer |
| 2 · Indirect injection | an email that carries the instruction; the lethal trifecta; a leak through a markdown image; links stripped at the gateway; reading split from sending |
| 3 · Supply chain | a poisoned tool description, live; a linter for descriptions and what it misses; the rug pull; hostile skills and typosquatting |
| 4 · Least privilege | a policy on tools, arguments and role; a colleague's address that passes the domain check; a narrow verb and an output check; the same in LangChain middleware and Agno tool hooks |
| 5 · Human approval | approve the arguments, not the summary; approval fatigue; "don't ask again" scoped too loosely; LangChain's `HumanInTheLoopMiddleware` and Agno's `requires_confirmation` |
| 6 · Guardrails | a regex, an LLM classifier and moderation scored on 20 attacks and 20 benign messages; base64 walks past them; the output guard; a guard on the gateway; the same in LangChain and Agno |
| 7 · Structural defences | spotlighting; plan-then-execute; the dual-LLM pattern; a defence map |
| 8 · Harden, measure, recap | all the layers in one agent; attacks and benign tasks scored before and after; the OWASP LLM Top 10; the one idea once more, glossary, clean up |

## What is in this repository

```
notebooks/                 the notebooks (with the author's outputs), level_N_helpers.py (the functions Levels 2–6
                           import) and img/ with their diagrams
pyproject.toml, uv.lock    the pinned environment the notebooks were verified with; .python-version pins Python 3.12
.env.example               template for your .env
```

The notebooks are generated from sources kept in a separate repository, so edit them only for your own experiments
and send corrections to the author. After you run a notebook, `git status` shows it as modified (your outputs replaced
the author's); `git checkout -- notebooks/` restores the published copy, and `git pull` brings new levels as they are published.
