"""Level 5 helpers: everything the notebook uses but does not teach.

The notebook imports each name it needs from this file, so ⌘+click (or F12) on a name in the notebook opens it here.
Nothing in this file is new to the course: these are the helpers of Level 1 and of Part 0 of the first edition, and the
printer of the call trees of Part 5.

    1. Models and prices             SMALL, REASONING, LARGE, EMBEDDING, PRICES, cost_usd
    2. Printing                      wrapped, show_json, show_messages, show_error, bar_chart
    3. Small tools                   count_tokens, run_many, pass_rate
    4. The toy gateway               gateway_app, serve, stop_servers, ask, gateway_log, before_call, ALIASES,
                                     keep_request, show_sent, show_log, spent
    5. NovaBank data                 TICKETS, HANDBOOK, section
    6. Call trees                    spans_of, reply_of, sent, tool_output, describe, details, show_trace
    7. Measuring helpers             sent_summary

Importing this file loads the key from .env and creates the gateway's connection to the provider. It starts no server:
the notebook does that with serve(), on its own port. It sets up no tracing either: Part 5 of the notebook teaches that.
"""
import json
import os
import random
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
REASONING = "gpt-6-luna"             # a thinking model; not used in this level
LARGE = "gpt-6.1-sol"                # the strongest model; used as a judge, never for tools here
EMBEDDING = "text-embedding-3-small" # not used in this level

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
def wrapped(text, width):
    """The lines of a text, each one wrapped at `width` characters. Nothing is cut: long lines go on in the next ones."""
    return [piece for line in str(text).splitlines() or [""]
            for piece in textwrap.wrap(line, width, break_on_hyphens=False) or [""]]


def show_json(title, value):
    """Print a title, then a value whole as indented JSON (an SDK object is turned into a dict first)."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", warnings=False)
    print(f"=== {title} ===")
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def show_messages(messages, title="Messages", width=118):
    """Print a conversation under a title: each message whole, its role, then its text or the tools it calls."""
    print(f"=== {title} ===")
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else message.content
        text = content if isinstance(content, str) else json.dumps(content)
        if isinstance(message, dict) and message.get("tool_calls"):
            text = "calls " + ", ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in message["tool_calls"])
        role = message["role"] if isinstance(message, dict) else message.role
        for number, piece in enumerate(wrapped(text or "", width - 12)):   # a long text goes on under the "|"
            print(f"{role if number == 0 else '':>9} | {piece}")


def show_error(error):
    """Print an API error whole: its type and HTTP status, then its full body (or its text, if it has no body)."""
    print(f"=== Error: {type(error).__name__} · HTTP {getattr(error, 'status_code', '-')} ===")
    body = getattr(error, "body", None)
    print(json.dumps(body, indent=2, ensure_ascii=False) if body else str(error))


BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
plt.rcParams.update({"figure.figsize": (7, 3), "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
pd.set_option("display.max_rows", None)                             # tables show every row ...
pd.set_option("display.max_colwidth", None)                         # ... and every text in full, never cut with "..."


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


def pass_rate(scores):
    """A list of 0/1 scores as 'k/n = rate'. Missing scores (None) are left out."""
    scores = [score for score in scores if pd.notna(score)]
    if not scores:
        return "no scores"
    k, n = int(sum(scores)), len(scores)
    return f"{k}/{n} = {k / n:.0%}"


# ---------------------------------------------------------------------------------------------------- 4. the toy gateway
UPSTREAM = str(OpenAI().base_url)                                   # the provider behind the gateway
TEAM_KEYS = {"sk-team-support": "support", "sk-team-lending": "lending"}
ALIASES = {"small": SMALL, "reasoning": REASONING, "large": LARGE, "embedding": EMBEDDING}
gateway_log = []                                                    # one entry per request
before_call, after_call = [], []                                    # hooks run on every call (Part 5 adds one)


class Blocked(Exception):
    """A guardrail raises this to refuse a request."""


gateway_app = FastAPI()
# The gateway's own client of the provider (OpenAI): the only client that holds the real key.
provider = httpx.Client(base_url=UPSTREAM, timeout=60, headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})


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
        for attempt in range(4):                                    # the platform retries rate limits and network errors
            try:
                upstream = provider.post(endpoint, json=body)
            except httpx.TransportError as error:                   # a timeout or a dropped connection counts as a 503
                upstream = httpx.Response(503, json={"error": {"message": f"provider unreachable: {type(error).__name__}"}})
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


def keep_request(body, entry, headers):
    """Gateway hook: store the request body in the log entry, under "sent". It changes nothing."""
    entry["sent"] = body


before_call.append(keep_request)                                    # from the first request on, every entry keeps its body


def show_sent(app, title=None):
    """The whole body of the last request that one app sent through the gateway, under a title; then its roles and format."""
    print(f"=== {title or 'The last request that ' + app + ' sent'}: the whole body ===")
    sent = [entry for entry in gateway_log if entry["app"] == app][-1]["sent"]
    print(json.dumps(sent, indent=2, default=str))
    print("roles of its messages:", [message["role"] for message in sent["messages"]])
    schema = (sent.get("response_format") or {}).get("json_schema") or {}
    print("response_format:", schema.get("name"), list((schema.get("schema") or {}).get("properties", {})))


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


def ask(client, prompt, system=None, model=SMALL, **options):
    """One question, one answer, through `client`. The simplest possible call."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(model=model, messages=messages, **options)
    return response.choices[0].message.content


def show_log(last=5, title="Gateway log"):
    """The last few gateway entries with every field, as a table under a title."""
    print(f"=== {title}: the last {min(last, len(gateway_log))} of {len(gateway_log)} entries, every field ===")
    return pd.DataFrame(gateway_log).tail(last)


def spent():
    """What this notebook has spent so far, from the gateway log."""
    return round(sum(entry["cost"] for entry in gateway_log), 4)


# ---------------------------------------------------------------------------------------------------- 5. NovaBank data
TICKETS = [  # (customer message, category, priority)
    ("There are two payments on my card that I did not make.", "CARD", "P1"),
    ("The cash machine kept my card and I cannot get it back.", "CARD", "P2"),
    ("How do I change the PIN of my card?", "CARD", "P3"),
    ("Contactless payments stopped working on my card.", "CARD", "P2"),
    ("I sent 900 EUR to the wrong account number by mistake.", "TRANSFER", "P1"),
    ("My standing order for the rent did not go out this month.", "TRANSFER", "P2"),
    ("How long does a transfer to the US take?", "TRANSFER", "P3"),
    ("I am locked out of online banking after three wrong passwords.", "ACCOUNT", "P2"),
    ("I moved. How do I update my address?", "ACCOUNT", "P3"),
    ("Can I open a second savings account?", "ACCOUNT", "P3"),
    ("What is the interest rate on a car loan?", "LOAN", "P3"),
    ("My loan payment was taken twice this month.", "LOAN", "P1"),
    ("The loan calculator on your website shows an error.", "LOAN", "P2"),
    ("I gave my password to someone who called and said they were from NovaBank.", "ACCOUNT", "P1"),
]


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


def section(product):
    """The handbook text of one product."""
    return HANDBOOK.split(f"## {product}\n")[1].split("\n## ")[0]


# ---------------------------------------------------------------------------------------------------- 6. call trees
# Part 5 of the notebook makes the spans (its own for Layer A, OpenInference's for B and C) and keeps them in an
# in-memory exporter; these functions only read them back and print them. Layer A's spans use the OTel GenAI names
# (gen_ai.*), B's and C's the OpenInference names (llm.*, input.value, output.value).
def spans_of(exporter, trace_id):
    """The finished spans of one trace, in the order they started."""
    spans = [span for span in exporter.get_finished_spans() if f"{span.context.trace_id:032x}" == trace_id]
    return sorted(spans, key=lambda span: span.start_time)


def reply_of(span):
    """A model span's reply: its text (None if there is none) and the tool calls it asked for, as 'name(arguments)'.
    A uses the OTel GenAI names, B and C the OpenInference names."""
    attributes = span.attributes
    if attributes.get("gen_ai.operation.name") == "chat":
        parts = [part for message in json.loads(attributes["gen_ai.output.messages"]) for part in message["parts"]]
        texts = [part["content"] for part in parts if part["type"] == "text"]
        calls = [f"{part['name']}({part['arguments']})" for part in parts if part["type"] == "tool_call"]
        return (texts[0] if texts else None), calls
    reply = "llm.output_messages.0.message."  # the reply only, not the messages of the input history
    calls, number = [], 0
    while f"{reply}tool_calls.{number}.tool_call.function.name" in attributes:
        call = f"{reply}tool_calls.{number}.tool_call.function."
        calls.append(f"{attributes[call + 'name']}({attributes.get(call + 'arguments')})")
        number += 1
    return attributes.get(reply + "content"), calls


def sent(span):
    """The system and user messages a model span was sent, as (role, whole text). A keeps every message sent in
    gen_ai.input.messages, B and C one attribute per field (llm.input_messages.0.message.content). The assistant and
    tool messages sent along are left out: the tree shows them already, as the steps before."""
    attributes = span.attributes
    if "gen_ai.input.messages" in attributes:
        return [(message["role"], "".join(part["content"] for part in message["parts"] if part["type"] == "text"))
                for message in json.loads(attributes["gen_ai.input.messages"]) if message["role"] in ("system", "user")]
    if attributes.get("openinference.span.kind") != "LLM":          # not a model call: LangGraph's nodes (CHAIN)
        return []                                                   # record the messages of their state too
    found, number = [], 0
    while f"llm.input_messages.{number}.message.role" in attributes:
        message = f"llm.input_messages.{number}.message."
        if attributes[message + "role"] in ("system", "user"):
            found.append((attributes[message + "role"], attributes.get(message + "content")))
        number += 1
    return found


def tool_output(span):
    """A tool span's result. A stores it in gen_ai.tool.call.result and Agno in output.value; LangChain stores the whole
    ToolMessage in output.value, and the result is its content."""
    attributes = span.attributes
    if "gen_ai.tool.call.result" in attributes:
        return attributes["gen_ai.tool.call.result"]
    if "output.value" not in attributes:
        return f"none: the tool raised ({span.status.description})"
    try:
        message = json.loads(attributes["output.value"])
    except ValueError:
        return attributes["output.value"]
    is_tool_message = isinstance(message, dict) and message.get("type") == "tool"
    return str(message["data"]["content"]) if is_tool_message else attributes["output.value"]


def describe(span):
    """What one step did, in one line: the model, its tokens and the tools it asked for (or that it answered), or the
    tool's name."""
    attributes = span.attributes
    kind = attributes.get("openinference.span.kind")
    if attributes.get("gen_ai.operation.name") == "chat":
        model, tokens_in, tokens_out = (attributes["gen_ai.response.model"], attributes["gen_ai.usage.input_tokens"],
                                        attributes["gen_ai.usage.output_tokens"])
    elif kind == "LLM":
        model, tokens_in, tokens_out = (attributes.get("llm.model_name"), attributes.get("llm.token_count.prompt"),
                                        attributes.get("llm.token_count.completion"))
    else:
        return f"{span.name} · {kind}" if kind else span.name  # the kind tells a LangGraph node from a model or tool call
    calls = reply_of(span)[1]
    return f"{span.name} · {model} · {tokens_in} → {tokens_out} tokens · " + (
        f"asks for {', '.join(call.split('(')[0] for call in calls)}" if calls else "answers")


def details(span):
    """The data of one step, whole, as (label, value): a tool's input and output, a model's text and its tool calls."""
    attributes = span.attributes
    if attributes.get("gen_ai.operation.name") == "execute_tool":
        return [("input", attributes["gen_ai.tool.call.arguments"]), ("output", tool_output(span))]
    if attributes.get("openinference.span.kind") == "TOOL":
        return [("input", attributes.get("input.value")), ("output", tool_output(span))]
    if attributes.get("gen_ai.operation.name") == "chat" or attributes.get("openinference.span.kind") == "LLM":
        text, calls = reply_of(span)
        return ([("text", text)] if text else []) + [("asks", call) for call in calls]
    return []


def show_trace(exporter, trace_id, title):
    """The call tree of one trace: each step under the step that made it, in the order they started, with its start
    (seconds after the run began), how long it took and what it did, and under it its data, whole (long lines are
    wrapped, never cut). Under a model call come first the system and user messages it was sent that no step above has
    shown: the first call shows the system prompt and the question, a later one only a new message. Then what the
    gateway billed under the trace id."""
    spans = spans_of(exporter, trace_id)
    ids = {span.context.span_id for span in spans}
    children = {}  # parent span id → its child spans, in the order they started (None: the root)
    for span in spans:
        children.setdefault(span.parent.span_id if span.parent and span.parent.span_id in ids else None, []).append(span)
    shown = set()  # the (role, text) messages the tree has shown

    def show(span, indent, branch):
        start, took = (span.start_time - spans[0].start_time) / 1e9, (span.end_time - span.start_time) / 1e9
        print(f"{start:+6.2f} {took:5.2f} s  {indent}{branch}{describe(span)}")
        below = children.get(span.context.span_id, [])
        indent_below = indent + {"├─ ": "│  ", "└─ ": "   "}.get(branch, "")  # the root has no branch
        margin = " " * 16 + indent_below + ("│  " if below else "   ")  # keeps the tree's lines going past the data
        new = [(role, text) for role, text in sent(span) if (role, text) not in shown]  # empty unless a model call
        shown.update(new)
        for label, value in new + details(span):
            lines = [piece for line in str(value).splitlines() or [""]
                     for piece in (textwrap.wrap(line, 100, break_on_hyphens=False) if len(line) > 100 else [line])]
            for number, line in enumerate(lines):
                print(margin + (f"{label}:" if number == 0 else "").ljust(8) + line)
        for number, child in enumerate(below):
            show(child, indent_below, "└─ " if number == len(below) - 1 else "├─ ")

    print(f"=== Call tree: {title} ({len(spans)} spans) ===")
    print(" start    took  call")
    for root in children.get(None, []):
        show(root, "", "")
    calls = [entry for entry in gateway_log if entry.get("trace_id") == trace_id]
    print(f"gateway, trace {trace_id}: {len(calls)} calls, {sum(entry['tokens'] for entry in calls)} tokens, "
          f"{sum(entry['cost'] for entry in calls):.5f} USD")


# ---------------------------------------------------------------------------------------------------- 7. measuring helpers
def sent_summary(app):
    """Calls and tokens of one app (its x-app header), and the shape of its last request, from the gateway log."""
    entries = [entry for entry in gateway_log if entry["app"] == app]
    sent = entries[-1]["sent"]
    system = [str(message["content"]) for message in sent["messages"] if message["role"] in ("system", "developer")]
    return {"app": app, "calls": len(entries), "tokens": sum(entry["tokens"] for entry in entries),
            "roles in last request": " ".join(message["role"] for message in sent["messages"]),
            "tools sent": len(sent.get("tools", [])), "system prompt tokens": count_tokens(" ".join(system))}
