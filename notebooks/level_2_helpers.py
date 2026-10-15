"""Level 2 helpers: everything the notebook uses but does not teach.

The notebook imports each name it needs from this file, so ⌘+click (or F12) on a name in the notebook opens it here.
Nothing in this file is new to the course: these are the helpers of Level 1 and of Part 0 of the first edition.

    1. Models and prices             SMALL, REASONING, LARGE, EMBEDDING, PRICES, cost_usd
    2. Printing                      wrapped, show_json, show_messages, show_graph_messages, show_error, bar_chart
    3. Small tools                   count_tokens, run_many
    4. The toy gateway               gateway_app, serve, stop_servers, gateway_log, LAST_BODY, show_log, spent
    5. NovaBank data                 TICKETS, HANDBOOK, HANDBOOK_SECTIONS, TODAY, ACCOUNTS, TRANSACTIONS, REFUNDS,
                                     lookup_transaction, five extra tools that are only described (Part 4)
    6. Call trees                    tracer, chat_attributes, tool_attributes, spans_of, show_trace
    7. Measuring helpers             first_step, first_tool, forced_arguments, prompt_tokens, with_description, sent_tool

Importing this file loads the key from .env, creates the gateway's connection to the provider and sets up tracing. It
starts no server: the notebook does that with serve(), on its own port.
"""
import copy
import json
import os
import random
import re
import textwrap
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date

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
REASONING = "gpt-6-luna"             # a thinking model; on Chat Completions, tools need reasoning_effort="none"
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
def wrapped(text, width):
    """The lines of a text, each one wrapped at `width` characters. Nothing is cut: long lines go on in the next ones."""
    return [piece for line in str(text).splitlines() or [""]
            for piece in textwrap.wrap(line, width, break_on_hyphens=False) or [""]]


def show_json(title, value):
    """Print a title, then a value whole as indented JSON (an SDK object is turned into a dict first)."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", warnings=False)      # `parsed` of a parse() reply is typed loosely
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


def show_graph_messages(messages, title="Messages", width=118):
    """show_messages for LangChain message objects: each message whole, its type, then its text or the tools it calls."""
    print(f"=== {title} ===")
    for message in messages:
        text = str(message.content)
        if getattr(message, "tool_calls", None):
            text = "calls " + ", ".join(f"{call['name']}({json.dumps(call['args'])})" for call in message.tool_calls)
        for number, piece in enumerate(wrapped(text, width - 12)):
            print(f"{message.type if number == 0 else '':>9} | {piece}")


def show_error(error):
    """Print an API error whole: its type and HTTP status, then its full body (or its text, if it has no body)."""
    print(f"=== Error: {type(error).__name__} · HTTP {getattr(error, 'status_code', '-')} ===")
    body = getattr(error, "body", None)
    print(json.dumps(body, indent=2, ensure_ascii=False) if body else str(error))


BLUE = "#2a78d6"
plt.rcParams.update({"figure.figsize": (7, 3), "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
pd.set_option("display.max_colwidth", None)                         # tables show every text in full, never cut with "..."


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


# ---------------------------------------------------------------------------------------------------- 4. the toy gateway
UPSTREAM = str(OpenAI().base_url)                                   # the provider behind the gateway
TEAM_KEYS = {"sk-team-support": "support", "sk-team-lending": "lending"}
ALIASES = {"small": SMALL, "reasoning": REASONING, "large": LARGE, "embedding": EMBEDDING}
gateway_log = []                                                    # one entry per request
before_call, after_call = [], []                                    # hooks run on every call
LAST_BODY = {}                                                      # app name -> the last request body the gateway received


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


def keep_body(body, entry, headers):
    """A hook that changes nothing: it keeps the last request body of each application (its x-app header)."""
    LAST_BODY[entry["app"]] = body


before_call.append(keep_body)
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
HANDBOOK_SECTIONS = HANDBOOK.split("## ")[1:]                       # one section per product, starting with its name

TODAY = date(2026, 10, 8)                     # the bank's "today" is fixed, so every run sees the same data
ACCOUNTS = {
    "A-1001": {"product": "Everyday account", "balance_eur": 1240.50},
    "A-1002": {"product": "Student account", "balance_eur": 86.20},
    "A-1003": {"product": "Premium account", "balance_eur": 15320.00},
}
TRANSACTIONS = [
    {"id": "T-101", "account_id": "A-1001", "date": "2026-10-07", "amount_eur": -54.90, "description": "Grocery store", "status": "booked"},
    {"id": "T-102", "account_id": "A-1001", "date": "2026-10-03", "amount_eur": -12.99, "description": "Streaming service", "status": "booked"},
    {"id": "T-103", "account_id": "A-1001", "date": "2026-10-03", "amount_eur": -12.99, "description": "Streaming service", "status": "booked"},
    {"id": "T-104", "account_id": "A-1001", "date": "2026-09-30", "amount_eur": 2100.00, "description": "Salary", "status": "booked"},
    {"id": "T-105", "account_id": "A-1001", "date": "2026-09-15", "amount_eur": -800.00, "description": "Rent", "status": "booked"},
    {"id": "T-201", "account_id": "A-1002", "date": "2026-10-06", "amount_eur": -4.50, "description": "Coffee bar", "status": "booked"},
    {"id": "T-202", "account_id": "A-1002", "date": "2026-10-05", "amount_eur": -39.00, "description": "Train ticket", "status": "pending"},
    {"id": "T-203", "account_id": "A-1002", "date": "2026-09-20", "amount_eur": 300.00, "description": "Transfer from parents", "status": "booked"},
    {"id": "T-204", "account_id": "A-1002", "date": "2026-10-07", "amount_eur": -120.00, "description": "Online shop", "status": "pending"},
    {"id": "T-301", "account_id": "A-1003", "date": "2026-10-08", "amount_eur": -2500.00, "description": "Car dealer", "status": "pending"},
    {"id": "T-302", "account_id": "A-1003", "date": "2026-10-01", "amount_eur": -61.20, "description": "Restaurant", "status": "booked"},
    {"id": "T-303", "account_id": "A-1003", "date": "2026-09-10", "amount_eur": -950.00, "description": "Mortgage payment", "status": "booked"},
]
REFUNDS = []                                  # every refund the agent pays lands here: the bank's "money out"


def lookup_transaction(transaction_id):
    """One transaction by its id. Not a tool, just a helper for the tools."""
    for transaction in TRANSACTIONS:
        if transaction["id"] == transaction_id:
            return transaction
    raise KeyError(transaction_id)


# Five more bank tools for Part 4.5: never called, only described, to show what a long tool list costs.
def open_account(product: str):
    """Open a new account of one NovaBank product for the customer."""
    return {"status": "opened"}


def close_account(account_id: str):
    """Close one account. Its balance must be zero."""
    return {"status": "closed"}


def update_address(account_id: str, street: str, city: str, postcode: str):
    """Change the postal address of the account holder."""
    return {"status": "updated"}


def order_card(account_id: str, product: str):
    """Order a new card for one account."""
    return {"status": "ordered"}


def set_card_limit(account_id: str, daily_limit_eur: int):
    """Set the daily card payment limit of one account."""
    return {"status": "set"}


# ---------------------------------------------------------------------------------------------------- 6. call trees
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
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)  # spans for the steps of LangChain (Part 5)
AgnoInstrumentor().instrument(tracer_provider=tracer_provider)      # spans for the steps of Agno (Part 5)
tracer = trace.get_tracer("novabank")                               # opens our own spans


def chat_attributes(response, messages):
    """What the span of one model call records: the whole list of messages it was sent, the model, the tokens, and the
    whole reply (its text or its tool calls)."""
    reply = response.choices[0].message.model_dump(mode="json", warnings=False)   # `parsed` is typed loosely: no warning
    return {"gen_ai.operation.name": "chat", "gen_ai.response.model": response.model,
            "gen_ai.usage.input_tokens": response.usage.prompt_tokens,
            "gen_ai.usage.output_tokens": response.usage.completion_tokens,
            "gen_ai.input.messages": json.dumps(messages), "gen_ai.output.messages": json.dumps([reply])}


def tool_attributes(call):
    """What the span of one tool call records before the tool runs: its name, the call's id and the whole arguments.
    After the run, the code adds the whole result as "gen_ai.tool.call.result"."""
    return {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": call.function.name,
            "gen_ai.tool.call.id": call.id, "gen_ai.tool.call.arguments": call.function.arguments}


def spans_of(root):
    """The finished spans of root's trace, in the order they started."""
    trace_id = root.get_span_context().trace_id
    return sorted([span for span in exporter.get_finished_spans() if span.context.trace_id == trace_id],
                  key=lambda span: span.start_time)


def asked_for(span):
    """The tool calls a model span asks for, as (call id, tool name, whole arguments). Our spans keep the whole reply; the
    instrumentors of LangChain and Agno keep one attribute per field ("llm.output_messages.0.message.tool_calls.1…")."""
    if "gen_ai.output.messages" in span.attributes:                 # our model calls
        messages = json.loads(span.attributes["gen_ai.output.messages"])
        return [(call["id"], call["function"]["name"], call["function"]["arguments"])
                for message in messages for call in message.get("tool_calls") or []]
    calls = {}                                                      # (message, call) -> {"id": …, "function.name": …, …}
    for key, value in span.attributes.items():
        found = re.fullmatch(r"llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\.(.+)", key)
        if found:
            calls.setdefault((int(found[1]), int(found[2])), {})[found[3]] = value
    return [(call.get("id"), call.get("function.name"), call.get("function.arguments", ""))
            for _, call in sorted(calls.items())]


def sent(span):
    """The system and user messages a model call was sent, as (role, whole text). Our spans keep the whole list sent;
    the instrumentors of LangChain and Agno keep one attribute per field ("llm.input_messages.0.message.content"). The
    assistant and tool messages sent along are left out: the tree shows them already, as the steps before."""
    if "gen_ai.input.messages" in span.attributes:                  # our model calls
        messages = json.loads(span.attributes["gen_ai.input.messages"])
    elif span.attributes.get("openinference.span.kind") == "LLM":   # a model call of LangChain or Agno
        fields = {}                                                 # message number -> {"role": …, "content": …}
        for key, value in span.attributes.items():
            found = re.fullmatch(r"llm\.input_messages\.(\d+)\.message\.(role|content)", key)
            if found:
                fields.setdefault(int(found[1]), {})[found[2]] = value
        messages = [fields[number] for number in sorted(fields)]
    else:                                                           # not a model call: LangGraph's nodes (CHAIN) record
        messages = []                                               # the messages of their state too
    return [(message["role"], message.get("content")) for message in messages
            if message.get("role") in ("system", "user")]


def describe(span, arguments_by_id):
    """What one step did, from its span: a headline and the details under it, as (label, whole value) pairs.
    A model call: the model and its tokens, then each tool it asks for with the whole arguments, or the whole text it
    answers. A tool: its name, then its whole input and its whole output. Any other step: its name, and its kind if an
    instrumentor set one. A step that raised also shows the exception."""
    attributes = span.attributes
    kind = attributes.get("openinference.span.kind")                # how the LangChain and Agno instrumentors mark a step
    raised = [("raised:", f"{event.attributes['exception.type']}: {event.attributes['exception.message']}")
              for event in span.events if event.name == "exception"]
    if attributes.get("gen_ai.operation.name") == "chat":           # our model calls
        headline = (f"{span.name} · {attributes['gen_ai.response.model']} · {attributes['gen_ai.usage.input_tokens']} → "
                    f"{attributes['gen_ai.usage.output_tokens']} tokens")
        text = json.loads(attributes["gen_ai.output.messages"])[0].get("content")
    elif kind == "LLM":                                             # the model calls of LangChain and Agno
        headline = (f"{span.name} · {attributes.get('llm.model_name')} · {attributes.get('llm.token_count.prompt')} → "
                    f"{attributes.get('llm.token_count.completion')} tokens")
        text = attributes.get("llm.output_messages.0.message.content")   # the reply only, not the messages sent
    elif attributes.get("gen_ai.operation.name") == "execute_tool":  # our tool calls
        output = attributes.get("gen_ai.tool.call.result", "(none: the tool raised)")
        return span.name, [("input:", attributes["gen_ai.tool.call.arguments"]), ("output:", output)] + raised
    elif kind == "TOOL":                                            # the tool calls of LangChain and Agno
        try:                                                        # LangChain keeps the whole ToolMessage, as JSON
            message = json.loads(attributes.get("output.value", ""))["data"]
            # its input.value can hold the bare value only, so take the arguments the model wrote for this call id
            arguments = arguments_by_id.get(message["tool_call_id"], attributes.get("input.value"))
            output = message["content"] if isinstance(message["content"], str) else json.dumps(message["content"])
        except (ValueError, TypeError, KeyError):                   # Agno keeps the arguments and the bare result
            arguments, output = attributes.get("input.value"), attributes.get("output.value", "(none: the tool raised)")
        return f"{span.name} · {kind}", [("input:", arguments), ("output:", output)] + raised
    else:
        return (f"{span.name} · {kind}" if kind else span.name), raised
    asks = [("asks for:", f"{name}({arguments})") for _, name, arguments in asked_for(span)]
    return headline, asks + ([("text:", text)] if text else []) + raised


def show_trace(root, title, width=120):
    """The call tree of root's trace: each step under the step that made it, in the order they started, with its start
    (seconds after the root began), how long it took and what it did. Under each step its details, whole: long values go
    on in more lines, wrapped at `width` characters, never cut. Under a model call come first the system and user
    messages it was sent that no step above has shown: the first call shows the system prompt and the question, a later
    one only a new message."""
    spans = spans_of(root)
    ids = {span.context.span_id for span in spans}
    children = {}                                                   # parent span id -> its child spans (None: the root)
    for span in spans:
        children.setdefault(span.parent.span_id if span.parent and span.parent.span_id in ids else None, []).append(span)
    arguments_by_id = {call_id: arguments for span in spans for call_id, _, arguments in asked_for(span)}
    shown = set()                                                   # the (role, text) messages the tree has shown

    def show(span, indent, branch):
        start, took = (span.start_time - spans[0].start_time) / 1e9, (span.end_time - span.start_time) / 1e9
        headline, details = describe(span, arguments_by_id)
        new = [(role, text) for role, text in sent(span) if (role, text) not in shown]   # empty unless a model call
        shown.update(new)
        details = [(f"{role}:", text) for role, text in new] + details
        print(f"{start:+6.2f} {took:5.2f} s  {indent}{branch}{headline}")
        below = children.get(span.context.span_id, [])
        indent_below = indent + {"├─ ": "│  ", "└─ ": "   "}.get(branch, "")  # the root has no branch
        under = " " * 16 + indent_below + ("│  " if below else "   ")   # keeps the tree's lines going on the left
        for label, value in details:
            for number, piece in enumerate(wrapped(value, max(width - len(under) - 10, 40))):
                print(f"{under}{label if number == 0 else '':<10}{piece}")
        for number, child in enumerate(below):
            show(child, indent_below, "└─ " if number == len(below) - 1 else "├─ ")

    print(f"=== Call tree: {title} ({len(spans)} spans) ===")
    print(" start    took  call")
    for top in children.get(None, []):
        show(top, "", "")


# ---------------------------------------------------------------------------------------------------- 7. measuring
def first_step(client, question, tools, tool_choice="auto", model=SMALL):
    """Ask once (through `client`) and report what the model did first: the tool calls it wrote, or its whole text."""
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": question}],
                                              tools=tools, tool_choice=tool_choice)
    message = response.choices[0].message
    if message.tool_calls:
        return "calls " + ", ".join(f"{call.function.name}({call.function.arguments})" for call in message.tool_calls)
    return "text: " + (message.content or "")


def first_tool(client, request, tools, short=None, model=SMALL):
    """Ask once and return the name of the first tool the model chose (renamed by `short`, if given), or "(text)"."""
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": request}], tools=tools)
    calls = response.choices[0].message.tool_calls
    if not calls:
        return "(text)"
    return (short or {}).get(calls[0].function.name, calls[0].function.name)


def forced_arguments(client, request, tool, model=SMALL):
    """Force one tool (tool_choice names it) and return the arguments the model wrote for it, as a dict."""
    choice = {"type": "function", "function": {"name": tool["function"]["name"]}}
    response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": request}], tools=[tool],
                                              tool_choice=choice)
    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)


def prompt_tokens(client, tools, model=SMALL):
    """Input tokens of the same tiny request ("Say OK."), sent with a list of tools (or none)."""
    messages = [{"role": "user", "content": "Say OK."}]
    if not tools:
        return client.chat.completions.create(model=model, messages=messages).usage.prompt_tokens
    return client.chat.completions.create(model=model, messages=messages, tools=tools).usage.prompt_tokens


def with_description(schema, description):
    """A copy of a tool schema with another description; the arguments stay the same."""
    schema = copy.deepcopy(schema)
    schema["function"]["description"] = description
    return schema


def sent_tool(body, name):
    """The schema of one tool as it reached the gateway, from a request body."""
    for tool in body.get("tools", []):
        if tool["function"]["name"] == name:
            return tool["function"]
