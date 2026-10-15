"""Level 6 helpers: everything the notebook uses but does not teach.

The notebook imports each name it needs from this file, so ⌘+click (or F12) on a name in the notebook opens it here.
Nothing in this file is new to the course: these are the helpers of Levels 1 to 5 and of Part 0 of the first edition.

    1. Models and prices             SMALL, REASONING, LARGE, EMBEDDING, PRICES, cost_usd
    2. Printing                      plain, show_json, show_error, bar_chart, BLUE, ORANGE
    3. Small tools                   count_tokens, run_many, ask
    4. The toy gateway               gateway_app, serve, stop_servers, refuse, Blocked, before_call, after_call,
                                     gateway_log, show_log, spent
    5. NovaBank data                 make_handbook, HANDBOOK, CANARY
    6. The stub's building blocks    one_tool_call, called_already
    7. Call trees                    exporter, tracer, spans_of, show_trace, and the readers it uses: UNTRUSTED_TOOLS,
                                     alerts, stops, readable, message_line, state_text, sent, describe

Importing this file loads the key from .env, creates the gateway's connection to the provider and sets up tracing. It
starts no server: the notebook does that with serve(), on its own port.
"""
import dataclasses
import json
import os
import random
import re
import textwrap
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))
assert os.getenv("OPENAI_API_KEY"), "No key found. Copy .env.example to .env and add your OpenAI key."

# ---------------------------------------------------------------------------------------------------- 1. models, prices
SMALL = "gpt-4.1-mini"               # fast and cheap: the model for tool calling in this course
REASONING = "gpt-6-luna"             # a thinking model; on Chat Completions it calls tools only with reasoning_effort="none"
LARGE = "gpt-6.1-sol"                # the strongest model; not used in this level
EMBEDDING = "text-embedding-3-small"

PRICES_CHECKED_ON = "2026-10-06"     # USD per 1M tokens
PRICES = {
    "gpt-4.1-mini": {"in": 0.40, "cached": 0.10, "out": 1.60},
    "gpt-4o-mini": {"in": 0.15, "cached": 0.075, "out": 0.60},
    "gpt-6-luna": {"in": 0.10, "cached": 0.01, "out": 0.50},
    "gpt-6.1-sol": {"in": 2.00, "cached": 0.10, "out": 10.00},
    "text-embedding-3-small": {"in": 0.02, "cached": 0.02, "out": 0.0},
}


def cost_usd(model, usage):
    """Price of one call in USD, from its usage (the SDK object or the raw dict)."""
    if not isinstance(usage, dict):
        usage = usage.model_dump()
    price = PRICES.get(model)
    if not price:
        return 0.0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    fresh = usage.get("prompt_tokens", 0) - cached
    return (fresh * price["in"] + cached * price["cached"] + usage.get("completion_tokens", 0) * price["out"]) / 1e6


# ---------------------------------------------------------------------------------------------------- 2. printing
def plain(obj):
    """SDK, pydantic and dataclass objects as plain dicts and lists, so that json.dumps can print them whole."""
    if isinstance(obj, type):
        return str(obj)
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    elif hasattr(obj, "to_dict") and not isinstance(obj, (dict, pd.DataFrame)):
        obj = obj.to_dict()
    elif dataclasses.is_dataclass(obj):
        obj = {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(key): plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [plain(value) for value in obj]
    return obj


def show_json(obj, title):
    """Print an object whole, as indented JSON, under a title."""
    print(f"=== {title} ===")
    print(json.dumps(plain(obj), indent=2, ensure_ascii=False, default=str))


def show_error(error):
    """Print an API error whole: its type and HTTP status, then its full body (or its text, if it has no body)."""
    print(f"=== Error: {type(error).__name__} · HTTP {getattr(error, 'status_code', '-')} ===")
    body = getattr(error, "body", None)
    print(json.dumps(body, indent=2, ensure_ascii=False, default=str) if body else str(error))


BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.figsize": (7, 3), "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
pd.set_option("display.max_colwidth", None)                         # tables show every cell in full, never cut with "..."


def bar_chart(values, title, fmt="{:,.2f}", color=BLUE):
    """Horizontal bars from a dict {label: number}, largest at the top."""
    labels, numbers = list(values), list(values.values())
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 1))
    ax.barh(labels[::-1], numbers[::-1], color=color)
    for i, number in enumerate(numbers[::-1]):
        ax.text(number, i, " " + fmt.format(number), va="center")
    ax.set_title(title, loc="left")
    ax.set_xlim(0, max(numbers) * 1.25 if max(numbers) > 0 else 1)
    plt.show()


# ---------------------------------------------------------------------------------------------------- 3. small tools
enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text):
    return len(enc.encode(text))


def run_many(function, items, workers=8):
    """Call function(item) for every item, a few at a time, keeping the order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))


def ask(client, prompt, system=None, model=SMALL, **options):
    """One question, one answer, through the given client. The simplest possible call."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages, **options)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------------------------------- 4. the toy gateway
UPSTREAM = str(OpenAI().base_url)                                   # the provider behind the gateway
TEAM_KEYS = {"sk-team-support": "support", "sk-team-lending": "lending"}
ALIASES = {"small": SMALL, "reasoning": REASONING, "large": LARGE, "embedding": EMBEDDING}
gateway_log = []                                                    # one entry per request
before_call, after_call = [], []                                    # guardrails: the notebook adds and removes them


class Blocked(Exception):
    """A guardrail raises this to refuse a request."""


gateway_app = FastAPI()
# The gateway's own client of the provider (OpenAI): the only client that holds the real key.
provider = httpx.Client(base_url=UPSTREAM, timeout=180, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})


def refuse(status, code, message):
    return JSONResponse({"error": {"message": message, "type": "gateway_error", "code": code}}, status_code=status)


@gateway_app.post("/v1/{endpoint:path}")
def gateway(endpoint: str, body: dict, request: Request):
    key = request.headers.get("authorization", "").removeprefix("Bearer ")
    entry = {"id": uuid.uuid4().hex[:8], "team": TEAM_KEYS.get(key), "app": request.headers.get("x-app", "-"),
             "model": ALIASES.get(body.get("model"), body.get("model")), "status": 0, "note": "", "tokens": 0, "cost": 0.0,
             "seconds": 0.0}
    gateway_log.append(entry)
    if entry["team"] is None:
        entry["status"] = 401
        return refuse(401, "invalid_api_key", "Unknown team key.")
    body["model"] = entry["model"]                                  # alias -> real model name
    try:
        for check in before_call:
            check(body, entry, request.headers)
        started = time.perf_counter()
        for attempt in range(4):                                    # the platform retries rate limits for its clients
            upstream = provider.post(endpoint, json=body)
            if upstream.status_code not in (429, 500, 502, 503) or attempt == 3:
                break
            time.sleep(2 ** attempt)
        entry["seconds"] = round(time.perf_counter() - started, 2)
        entry["status"] = upstream.status_code
        if upstream.status_code != 200:
            return JSONResponse(upstream.json(), status_code=upstream.status_code)
        answer = upstream.json()
        for check in after_call:
            check(answer, entry)
    except Blocked as reason:
        entry["status"], entry["note"] = 400, str(reason)
        return refuse(400, "blocked_by_guardrail", str(reason))
    usage = answer.get("usage") or {}
    entry["tokens"], entry["cost"] = usage.get("total_tokens", 0), cost_usd(body["model"], usage)
    return JSONResponse(answer, headers={"x-gateway-request-id": entry["id"]})


servers = {}                                                        # port -> (uvicorn server, its thread)


def serve(app, port):
    """Run a FastAPI app in a background thread of this kernel. Calling it again on the same port restarts it."""
    if port in servers:                                             # stop the copy from an earlier run
        servers[port][0].should_exit = True
        servers[port][1].join(timeout=5)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:                                       # a taken port stops uvicorn, and then us
        assert thread.is_alive(), f"Could not start on port {port}. Is another program or kernel using it?"
        time.sleep(0.05)
    servers[port] = (server, thread)
    return f"http://127.0.0.1:{port}/v1"


def stop_servers():
    """Stop every server serve() started, and say which ports were freed."""
    for port, (server, thread) in list(servers.items()):
        server.should_exit = True
        thread.join(timeout=5)
        print(f"Stopped the server on port {port}.")
    servers.clear()


def show_log(since=0, title="Gateway log"):
    """Every gateway entry from position `since` on (all of them by default), with every field, as a table under a title."""
    print(f"=== {title} ===")
    return pd.DataFrame(gateway_log[since:])


def spent():
    """What this notebook has spent so far, from the gateway log."""
    return round(sum(entry["cost"] for entry in gateway_log), 4)


# ---------------------------------------------------------------------------------------------------- 5. NovaBank data
def make_handbook():
    """A made-up product handbook: 12 products, the same text every time."""
    rng = random.Random(7)
    products = ["Student account", "Everyday account", "Premium account", "Youth account", "Business account", "Savings account",
                "Fixed deposit", "Travel card", "Credit card", "Car loan", "Personal loan", "Mortgage"]
    lines = ["NovaBank product handbook, edition 2026-10.", ""]
    for product in products:
        fee, atm, transfer = rng.choice([0, 2, 5, 9, 15]), rng.choice([250, 450, 600, 1000, 2000]), rng.choice([2000, 5000, 10000, 25000])
        lines += [
            f"## {product}",
            f"The {product} costs {fee} EUR per month. The daily ATM withdrawal limit is {atm} EUR and the daily transfer limit is {transfer} EUR.",
            f"The interest rate is {rng.choice([0.5, 1.2, 2.4, 3.9, 6.5])}% per year. A replacement card costs {rng.choice([0, 5, 10])} EUR "
            f"and arrives within {rng.choice([3, 5, 7])} working days.",
            f"Customers can open the {product} in the app or in a branch with a valid id. Closing it is free after {rng.choice([6, 12, 24])} months.",
            f"Support for the {product} is available by phone from {rng.choice([7, 8, 9])}:00 to {rng.choice([18, 20, 22])}:00 on working days, "
            f"and by chat at any time.",
            f"Payments abroad with the {product} cost {rng.choice([0, 1, 2])}% of the amount, with a minimum of {rng.choice([1, 2, 3])} EUR. "
            f"Cash withdrawals abroad cost {rng.choice([2, 3, 5])} EUR each.",
            f"Disputes about {product} transactions must be reported within {rng.choice([30, 60, 90])} days. "
            f"Refunds for confirmed fraud are paid within {rng.choice([5, 10, 14])} working days.",
            "",
        ]
    return "\n".join(lines)


HANDBOOK = make_handbook()
CANARY = "NB-CANARY-7731"                      # the "secret" to protect: a canary, not a real credential


# ---------------------------------------------------------------------------------------------------- 6. the stub's blocks
def one_tool_call(name, args):
    """An assistant message that asks for exactly one tool."""
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def called_already(messages):
    """The names of the tools the assistant messages have asked for so far, in order."""
    return [c["function"]["name"] for m in messages if m.get("role") == "assistant" for c in (m.get("tool_calls") or [])]


# ---------------------------------------------------------------------------------------------------- 7. call trees
from openinference.instrumentation.agno import AgnoInstrumentor          # noqa: E402  (after the data, as in Part 0)
from openinference.instrumentation.langchain import LangChainInstrumentor  # noqa: E402
from opentelemetry import trace                                          # noqa: E402
from opentelemetry.sdk.trace import TracerProvider                       # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor           # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

exporter = InMemorySpanExporter()                                   # keeps every finished span in a list
tracer_provider = TracerProvider()                                  # OTel keeps its first provider: this file runs once
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(tracer_provider)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)  # spans for the steps of LangChain (Parts 4–6)
AgnoInstrumentor().instrument(tracer_provider=tracer_provider)      # spans for the steps of Agno (Parts 4–6)
tracer = trace.get_tracer("novabank")                               # opens our own spans: run_agent and apply_tool

UNTRUSTED_TOOLS = ("list_emails", "read_email", "read_file", "fetch_url")   # tools that bring outside text in


def spans_of(root):
    """The finished spans of root's trace, in the order they started."""
    trace_id = root.get_span_context().trace_id
    return sorted([span for span in exporter.get_finished_spans() if span.context.trace_id == trace_id],
                  key=lambda span: span.start_time)


def alerts(name, arguments, result):
    """The alerts for one tool step: outside text came in, the canary went in or out, or the step was denied."""
    marks = ["⚠ untrusted text"] if name.endswith(UNTRUSTED_TOOLS) and "Denied" not in str(result) else []
    marks += ["⚠ canary in the arguments"] if CANARY in str(arguments) else []
    marks += ["⚠ canary in the result"] if CANARY in str(result) else []
    marks += ["⛔ denied: " + str(result).split("Denied by policy: ")[1]] if "Denied by policy: " in str(result) else []
    return "".join(" · " + mark for mark in marks)


def stops(span):
    """The marks of a LangChain middleware step that paused, blocked or changed the run: a human gate, a guard, a rejection.
    Only middleware steps (named <middleware>.<hook>): the graph around them would repeat their marks."""
    if "." not in span.name:
        return ""
    output = str(span.attributes.get("output.value"))
    pauses = [str(event.attributes.get("exception.message")) for event in span.events
              if "GraphInterrupt" in str(event.attributes.get("exception.type"))]   # LangGraph pauses by raising this
    marks = ["⏸ paused for a human: " + ", ".join(re.findall(r"'name': '(\w+)'", pause)) for pause in pauses]
    marks += ["⛔ blocked: the run jumps to the end"] if '"jump_to": "end"' in output else []
    marks += ["⛔ replaced the reply"] if "redacted by the output guardrail" in output else []
    rejected = re.search(r"User rejected the tool call[^\"]*", output)
    marks += ["⛔ denied: " + rejected.group(0)] if rejected else []
    return "".join(" · " + mark for mark in marks)


def readable(text):
    """A JSON text with any non-ASCII letters written as themselves (spans store json.dumps output); other text as it is."""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False)
    except (TypeError, ValueError):
        return str(text)


def message_line(message):
    """One message as one line: who wrote it, its whole text, and the tools it asks for with their whole arguments."""
    data = message.get("data", message)                             # LangChain stores {"type": ..., "data": {...}}
    who = data.get("role") or data.get("type")
    who += f" ({data['name']})" if who == "tool" and data.get("name") else ""
    parts = [str(data["content"])] if data.get("content") else []
    parts += [f"asks for {call['name']}({json.dumps(call['args'], ensure_ascii=False)})"
              for call in data.get("tool_calls") or []]
    return f"{who}: " + (" · ".join(parts) or "(empty)")


def state_text(value):
    """What a LangGraph step got or gave back (a JSON text): one line per message, then every other key."""
    try:
        data = json.loads(value) if value else None
    except ValueError:
        return str(value)
    if not data:
        return "nothing: the run goes on unchanged"
    if not isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False)
    lines = [message_line(message) for message in data.get("messages", [])]
    lines += [f"{key}: {json.dumps(item, ensure_ascii=False)}" for key, item in data.items()
              if key != "messages" and item not in (None, [], {})]
    return "\n".join(lines)


def sent(span):
    """The system and user messages a model call was sent, as (role, whole text). run_agent's spans keep the whole list
    sent; the instrumentors of LangChain and Agno keep one attribute per field ("llm.input_messages.0.message.content").
    The assistant and tool messages sent along are left out: the tree shows them already, as the steps before."""
    attributes = span.attributes
    if "gen_ai.input.messages" in attributes:                       # run_agent's model calls
        messages = json.loads(attributes["gen_ai.input.messages"])
    elif attributes.get("openinference.span.kind") == "LLM":        # a model call of LangChain or Agno
        fields = {}                                                 # message number -> {"role": …, "content": …}
        for key, value in attributes.items():
            found = re.fullmatch(r"llm\.input_messages\.(\d+)\.message\.(role|content)", key)
            if found:
                fields.setdefault(int(found[1]), {})[found[2]] = value
        messages = [fields[number] for number in sorted(fields)]
    else:                                                           # not a model call: LangGraph's nodes (CHAIN) record
        messages = []                                               # the messages of their state too
    return [(message["role"], str(message.get("content"))) for message in messages
            if message.get("role") in ("system", "user")]


def describe(span):
    """One step of a trace, from its span: a line that names it, and (label, whole value) pairs for the lines below it.
    A model call: the model, its tokens, then each tool call or the text. A tool: its name and alerts, then its input
    and output. An approval: the decision, then the call it judged. A LangChain step: its kind and stops, then its
    input and output."""
    attributes = span.attributes
    kind = attributes.get("openinference.span.kind")                # how the LangChain and Agno instrumentors mark a step
    if attributes.get("gen_ai.operation.name") == "chat":           # run_agent's model calls
        model, tokens_in, tokens_out = (attributes.get("gen_ai.response.model"), attributes.get("gen_ai.usage.input_tokens"),
                                        attributes.get("gen_ai.usage.output_tokens"))
        output = json.loads(attributes["gen_ai.output.messages"])
        calls = [(call["function"]["name"], call["function"]["arguments"])
                 for message in output for call in message.get("tool_calls") or []]
        reply = "\n".join(str(message["content"]) for message in output if message.get("content"))
    elif kind == "LLM":                                             # the model calls of LangChain and Agno
        model, tokens_in, tokens_out = (attributes.get("llm.model_name"), attributes.get("llm.token_count.prompt"),
                                        attributes.get("llm.token_count.completion"))
        output = {key: value for key, value in attributes.items() if key.startswith("llm.output_messages.")}  # not the input
        calls = list(zip([value for key, value in output.items() if key.endswith(".tool_call.function.name")],
                         [value for key, value in output.items() if key.endswith(".tool_call.function.arguments")]))
        reply = "\n".join(str(value) for key, value in output.items() if key.endswith(".message.content"))
    elif attributes.get("gen_ai.operation.name") == "execute_tool":
        arguments, result = attributes["gen_ai.tool.call.arguments"], attributes["gen_ai.tool.call.result"]
        return span.name + alerts(span.name, arguments, result), [("input", readable(arguments)), ("output", result)]
    elif kind == "TOOL":
        arguments, result = attributes.get("input.value"), attributes.get("output.value")
        try:                                                        # LangChain returns a whole ToolMessage: show its text
            result = json.loads(result).get("data", {}).get("content", result)
        except (TypeError, ValueError, AttributeError):
            pass
        return span.name + alerts(span.name, arguments, result), [("input", readable(arguments)), ("output", result)]
    elif "approval.decision" in attributes:
        return f"{span.name} · {attributes['approval.decision']}", [("input", attributes["approval.tool_call"])]
    elif "input.value" in attributes and (span.name == "LangGraph" or "." in span.name):  # the graph, or a middleware step
        pauses = [str(event.attributes.get("exception.message")) for event in span.events
                  if "GraphInterrupt" in str(event.attributes.get("exception.type"))]
        output = ("paused, waiting for a human: " + "\n".join(pauses) if pauses
                  else state_text(attributes.get("output.value")))
        return f"{span.name} · {kind}" + stops(span), [("input", state_text(attributes["input.value"])), ("output", output)]
    else:
        return (f"{span.name} · {kind}" if kind else span.name), []
    tokens = f"{model} · {tokens_in} → {tokens_out} tokens" if tokens_in is not None else "a stub: no model, no tokens"
    line = (f"{span.name} · {tokens} · " + (f"asks for {', '.join(name for name, _ in calls)}" if calls else "answers")
            + (" · ⚠ canary in the reply" if CANARY in reply else ""))
    details = [("call", f"{name}({readable(arguments)})") for name, arguments in calls]
    return line, details + ([("text", reply)] if reply else [])


def show_trace(root, title, width=120):
    """The call tree of root's trace: each step under the step that made it, in the order they started, with its start
    (seconds after the root began), how long it took and what it did. Under each step its details, each value whole:
    a line longer than `width` characters goes on in the next ones, two spaces further in, so nothing is cut. Under a
    model call come first the system and user messages it was sent that no step above has shown: the first call shows
    the system prompt and the request, a later one only a new message."""
    spans = spans_of(root)
    ids = {span.context.span_id for span in spans}
    children = {}                                                   # parent span id -> its child spans (None: the root)
    for span in spans:
        children.setdefault(span.parent.span_id if span.parent and span.parent.span_id in ids else None, []).append(span)
    shown = set()                                                   # the (role, text) messages the tree has shown

    def show(span, indent, branch):
        start, took = (span.start_time - spans[0].start_time) / 1e9, (span.end_time - span.start_time) / 1e9
        line, details = describe(span)
        new = [(role, text) for role, text in sent(span) if (role, text) not in shown]   # empty unless a model call
        shown.update(new)
        line += "".join(f" · ⚠ canary in the {role} message" for role, text in new if CANARY in text)
        details = new + details
        print(f"{start:+6.2f} {took:5.2f} s  {indent}{branch}{line}")
        below = children.get(span.context.span_id, [])
        indent_below = indent + {"├─ ": "│  ", "└─ ": "   "}.get(branch, "")  # the root has no branch
        margin = " " * 16 + indent_below + ("│  " if below else "   ")    # the │ leads down to this step's own children
        for label, value in details:
            room = max(40, width - len(margin) - len(label) - 2)
            pieces = [piece for text in str(value).split("\n")
                      for piece in textwrap.wrap(text, room, subsequent_indent="  ", break_on_hyphens=False) or [""]]
            for number, piece in enumerate(pieces):
                print(margin + (f"{label}: " if number == 0 else " " * (len(label) + 2)) + piece)
        for number, child in enumerate(below):
            show(child, indent_below, "└─ " if number == len(below) - 1 else "├─ ")

    print(f"=== Call tree: {title} ({len(spans)} spans) ===")
    print(" start    took  call")
    for top in children.get(None, []):
        show(top, "", "")
