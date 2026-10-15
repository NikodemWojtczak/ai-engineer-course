"""Level 4 helpers: everything the notebook uses but does not teach.

The notebook imports each name it needs from this file, so ⌘+click (or F12) on a name in the notebook opens it here.
Nothing in this file is new to the course: these are the helpers of Levels 1 to 3 and of Part 0 of the first edition.

    1. Models and prices             SMALL, REASONING, LARGE, EMBEDDING, PRICES, cost_usd
    2. Printing                      wrapped, show_json, show_messages, show_lc, show_run
    3. Small tools                   count_tokens, run_many
    4. The toy gateway               gateway_app, serve, stop_servers, gateway_log, last_request, ask, show_log, spent
    5. NovaBank data                 TICKETS, HANDBOOK, ACCOUNTS, TRANSACTIONS, REFUNDS
    6. Call trees                    tracer, spans_of, sent, model_reply, tool_input_output, describe, show_trace

Importing this file loads the key from .env, creates the gateway's connection to the provider and sets up tracing. It
starts no server: the notebook does that with serve(), on its own ports.
"""
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
import pandas as pd
import tiktoken
import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))
assert os.getenv("OPENAI_API_KEY"), "No key found. Copy .env.example to .env and add your OpenAI key."
pd.set_option("display.max_colwidth", None)                         # tables show every text in full, never cut with "..."

# ---------------------------------------------------------------------------------------------------- 1. models, prices
SMALL = "gpt-4.1-mini"               # fast and cheap: the model for tool calling in this course
REASONING = "gpt-6-luna"             # a thinking model; not used in this level
LARGE = "gpt-6.1-sol"                # the strongest model; not used in this level
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


def show_lc(messages, title="LangChain messages", width=118):
    """Print a whole LangChain conversation under a title, one entry per message: its type, then its text or the tools
    it calls."""
    print(f"=== {title} ===")
    for message in messages:
        text = message.text
        if getattr(message, "tool_calls", None):
            text = "calls " + ", ".join(f"{call['name']}({json.dumps(call['args'])})" for call in message.tool_calls)
        for number, piece in enumerate(wrapped(text, width - 12)):
            print(f"{message.type if number == 0 else '':>9} | {piece}")


def show_run(run, title="Agno run"):
    """Print an Agno run under a title: its status, the tools it called with their whole results, and the answer."""
    print(f"=== {title} ===")
    print("   status |", run.status)
    for tool in run.tools or []:
        print(f"     tool | {tool.tool_name}({json.dumps(tool.tool_args)}) -> {tool.result}")
    print("   answer |", run.content)


# ---------------------------------------------------------------------------------------------------- 3. small tools
enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text):
    return len(enc.encode(text))


def run_many(function, items, workers=8):
    """Call function(item) for every item, a few at a time, keeping the order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(function, items))


# ---------------------------------------------------------------------------------------------------- 4. the toy gateway
UPSTREAM = str(OpenAI().base_url)                                   # the provider behind the gateway
TEAM_KEYS = {"sk-team-support": "support", "sk-team-lending": "lending"}
ALIASES = {"small": SMALL, "reasoning": REASONING, "large": LARGE, "embedding": EMBEDDING}
gateway_log = []                                                    # one entry per request
before_call, after_call = [], []                                    # hooks run on every call
last_request = {}                                                   # app name -> the last request body it sent


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


def remember_request(body, entry, headers):
    """A hook that changes nothing: it keeps the last request body of each application (its x-app header)."""
    last_request[entry["app"]] = body


before_call.append(remember_request)
servers = {}                                                        # port -> (uvicorn server, its thread)


def serve(app, port):
    """Run a web app (FastAPI, or any ASGI app) in a background thread of this kernel. Calling it again on the same port
    restarts it."""
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

ACCOUNTS = {"A-1001": {"owner": "Ada Novak", "balance_eur": 120.50}, "A-1002": {"owner": "Ben Okafor", "balance_eur": 3400.00}}
TRANSACTIONS = [
    {"id": "T-5001", "account": "A-1001", "date": "2026-10-01", "merchant": "City Books", "amount_eur": 45.90},
    {"id": "T-5002", "account": "A-1001", "date": "2026-10-01", "merchant": "City Books", "amount_eur": 45.90},
    {"id": "T-5003", "account": "A-1001", "date": "2026-10-03", "merchant": "Metro Cafe", "amount_eur": 4.20},
    {"id": "T-5004", "account": "A-1002", "date": "2026-10-01", "merchant": "Rent October", "amount_eur": 1200.00},
    {"id": "T-5005", "account": "A-1002", "date": "2026-10-02", "merchant": "Electronics World", "amount_eur": 899.00},
    {"id": "T-5006", "account": "A-1002", "date": "2026-10-02", "merchant": "Electronics World", "amount_eur": 899.00},
]
REFUNDS = []                                                        # every refund the tools book lands here

# ---------------------------------------------------------------------------------------------------- 6. call trees
from openinference.instrumentation.agno import AgnoInstrumentor          # noqa: E402  (after the data, as in Part 0)
from openinference.instrumentation.langchain import LangChainInstrumentor  # noqa: E402
from opentelemetry import trace                                          # noqa: E402
from opentelemetry.sdk.trace import TracerProvider                       # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor           # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

os.environ["OTEL_INSTRUMENTATION_A2A_SDK_ENABLED"] = "false"        # the A2A SDK's own 25 spans per message (Part 3)
exporter = InMemorySpanExporter()                                   # keeps every finished span in a list
tracer_provider = TracerProvider()                                  # OTel keeps its first provider: this file runs once
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(tracer_provider)                          # for our tracer, and for MCP, FastMCP and FastAPI
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)  # spans for the steps of LangChain
AgnoInstrumentor().instrument(tracer_provider=tracer_provider)      # spans for the steps of Agno
tracer = trace.get_tracer("novabank")                               # opens our own spans


def spans_of(root):
    """The finished spans of root's trace, in the order they started."""
    trace_id = root.get_span_context().trace_id
    return sorted([span for span in exporter.get_finished_spans() if span.context.trace_id == trace_id],
                  key=lambda span: span.start_time)


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
    return [(message["role"], message.get("content")) for message in messages
            if message.get("role") in ("system", "user")]


def model_reply(span):
    """What a model call answered, from its span: the tools it asked for, as name(whole arguments), and its text."""
    attributes = span.attributes
    if attributes.get("gen_ai.operation.name") == "chat":           # run_agent's model calls
        reply = json.loads(attributes["gen_ai.output.messages"])[0]
        asked = [f"{call['function']['name']}({call['function']['arguments']})" for call in reply.get("tool_calls", [])]
        return asked, reply.get("content")
    calls = "llm.output_messages.0.message.tool_calls."              # the model calls of LangChain and Agno: the reply
    count = len([key for key in attributes if key.startswith(calls) and key.endswith(".function.name")])
    asked = [f"{attributes[f'{calls}{number}.tool_call.function.name']}"
             f"({attributes.get(f'{calls}{number}.tool_call.function.arguments', '')})" for number in range(count)]
    return asked, attributes.get("llm.output_messages.0.message.content")


def tool_input_output(span, parent):
    """A tool's whole input and output, from its span (and, for LangChain, from the span of its "tools" step)."""
    attributes = span.attributes
    if attributes.get("gen_ai.operation.name") == "execute_tool":    # run_agent's tool calls
        return attributes["gen_ai.tool.call.arguments"], attributes["gen_ai.tool.call.result"]
    tool_input, tool_output = attributes.get("input.value", ""), attributes.get("output.value", "(none)")
    if span.instrumentation_scope.name.endswith("langchain"):       # LangChain keeps only the value of a one-argument
        if parent is not None and parent.name == "tools":           # tool in input.value; its "tools" step has the call
            tool_input = next((json.dumps(call["args"], ensure_ascii=False) for call in
                               json.loads(parent.attributes["input.value"]) if call["name"] == span.name), tool_input)
        if tool_output.startswith("{"):                             # output.value is the whole ToolMessage, as JSON
            content = json.loads(tool_output)["data"]["content"]   # a text, or a list of content blocks
            tool_output = content if isinstance(content, str) else "\n".join(
                block["text"] if block.get("type") == "text" else json.dumps(block) for block in content)
    return tool_input, tool_output


def describe(span, parent=None):
    """What one step did, from its span: a headline, and (label, value) pairs for the lines under it, values whole.
    A model call: the model and its tokens; then each tool it asked for with the whole arguments, or the whole text it
    answered. A tool: its whole input and output. An A2A message: the whole text sent and answered. A request that a
    server received: [which server] and the request."""
    attributes = span.attributes
    kind = attributes.get("openinference.span.kind")                # how the LangChain and Agno instrumentors mark a step
    details = []
    if span.kind == trace.SpanKind.SERVER:
        side = "MCP server" if span.instrumentation_scope.name == "fastmcp" else "lending server"
        headline = f"[{side}] {span.name}" + (f" · port {attributes['server.port']}" if "server.port" in attributes else "")
    elif "a2a.message.text" in attributes:                          # an A2A message that our code sent (Part 3)
        headline = span.name
        details = [("message:", attributes["a2a.message.text"]), ("answer:", attributes["a2a.reply.text"] or "(empty)")]
        if "a2a.context_id.sent" in attributes:                     # the conversation id, both ways (Part 3)
            details += [("contextId sent:", attributes["a2a.context_id.sent"]),
                        ("contextId in the reply:", attributes["a2a.context_id.reply"])]
    elif attributes.get("gen_ai.operation.name") == "chat" or kind == "LLM":   # a model call
        if kind == "LLM":                                           # the model calls of LangChain and Agno
            model, tokens_in, tokens_out = (attributes.get("llm.model_name"), attributes.get("llm.token_count.prompt"),
                                            attributes.get("llm.token_count.completion"))
        else:                                                       # run_agent's model calls
            model, tokens_in, tokens_out = (attributes["gen_ai.response.model"], attributes["gen_ai.usage.input_tokens"],
                                            attributes["gen_ai.usage.output_tokens"])
        headline = f"{span.name} · {model} · {tokens_in} → {tokens_out} tokens"
        asked, text = model_reply(span)
        details = [("asks for:", call) for call in asked]
        if text or not asked:
            details.append(("text:", text or "(empty)"))
    elif attributes.get("gen_ai.operation.name") == "execute_tool" or kind == "TOOL":   # a tool call
        headline = span.name if kind is None else f"{span.name} · TOOL"
        tool_input, tool_output = tool_input_output(span, parent)
        details = [("input:", tool_input), ("output:", tool_output)]
    elif "code.function.name" in attributes:                        # FastAPI's steps inside a request (Part 3)
        headline = f"{span.name} · {attributes['code.function.name']}"
    elif span.instrumentation_scope.name in ("fastmcp", "mcp-python-sdk"):  # the MCP clients inside LangChain and Agno
        headline = f"{span.name} · recorded by {span.instrumentation_scope.name}"
    else:
        headline = f"{span.name} · {kind}" if kind else span.name
    if "tools/call " in span.name:                                  # an MCP request for a tool, client or server side:
        tool = span.name.split("tools/call ")[1].split()[0]         # the tool's step in our agent holds its data
        details.append(("input and output:", f"on our tool step {tool} above"))
    return headline, details


TREE_WIDTH = 110                                                    # longer lines go on in the next line; nothing is cut


def show_trace(root, title):
    """The call tree of root's trace: each step under the step that made it, in the order they started, with its start
    (seconds after the root began), how long it took and what it did; under it, the step's data, labelled and whole.
    Under a model call come first the system and user messages it was sent that no step above has shown: the first call
    shows the system prompt and the question, a later one only a new message."""
    time.sleep(0.2)                                                 # a server ends its span a moment after it answers
    spans = spans_of(root)
    ids = {span.context.span_id for span in spans}
    children = {}                                                   # parent span id -> its child spans (None: the root)
    for span in spans:
        children.setdefault(span.parent.span_id if span.parent and span.parent.span_id in ids else None, []).append(span)
    shown = set()                                                   # the (role, text) messages the tree has shown

    def show(span, parent, indent, branch):
        start, took = (span.start_time - spans[0].start_time) / 1e9, (span.end_time - span.start_time) / 1e9
        headline, details = describe(span, parent)
        new = [(role, text) for role, text in sent(span) if (role, text) not in shown]   # empty unless a model call
        shown.update(new)
        details = [(f"{role}:", text) for role, text in new] + details
        print(f"{start:+6.2f} {took:5.2f} s  {indent}{branch}{headline}")
        below = children.get(span.context.span_id, [])
        indent_below = indent + {"├─ ": "│  ", "└─ ": "   "}.get(branch, "")  # the root has no branch
        margin = " " * 16 + indent_below + ("│   " if below else "    ")      # the line down to its children goes on
        for label, value in details:
            width = max(TREE_WIDTH - len(margin) - len(label) - 1, 40)
            lines = [part for line in str(value).split("\n")       # keep the value's own lines, wrap the long ones
                     for part in textwrap.wrap(line, width, break_on_hyphens=False) or [""]]
            for number, line in enumerate(lines):
                print((margin + (label if number == 0 else " " * len(label)) + " " + line).rstrip())
        for number, child in enumerate(below):
            show(child, span, indent_below, "└─ " if number == len(below) - 1 else "├─ ")

    print(f"=== Call tree: {title} ({len(spans)} spans) ===")
    print(" start    took  call")
    for top in children.get(None, []):
        show(top, None, "", "")
