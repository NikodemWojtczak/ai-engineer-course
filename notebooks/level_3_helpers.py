"""Level 3 helpers: everything the notebook uses but does not teach.

The notebook imports each name it needs from this file, so ⌘+click (or F12) on a name in the notebook opens it here.
Nothing in this file is new to the course: these are the helpers of Level 1 and of Part 0 of the first edition.

    1. Models and prices             SMALL, REASONING, LARGE, EMBEDDING, PRICES, cost_usd
    2. Printing                      plain, show_json, show_messages, show_error, bar_chart
    3. Small tools                   count_tokens, run_many
    4. The toy gateway               gateway_app, serve, stop_servers, gateway_log, refuse, keep_request, keep_usage,
                                     show_log, spent, show_sent, reply_text, find_call
    5. NovaBank data                 TICKETS, HANDBOOK, POLICIES

Importing this file loads the key from .env, creates the gateway's connection to the provider, adds the two hooks that
keep what reached the gateway, and sets pandas to print every row and every text whole. It starts no server: the
notebook does that with serve(), on its own port.
"""
import dataclasses
import json
import os
import random
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
def show_messages(messages, title="Messages"):
    """Print a whole conversation under a title, one entry per message."""
    print(f"=== {title} ===")
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else message.content
        text = content if isinstance(content, str) else json.dumps(content)
        if isinstance(message, dict) and message.get("tool_calls"):
            text = "calls " + ", ".join(f"{c['function']['name']}({c['function']['arguments']})" for c in message["tool_calls"])
        role = message["role"] if isinstance(message, dict) else message.role
        print(f"{role:>9} | {text or ''}")


def show_error(error):
    """Print an API error whole: its type and HTTP status, then its full body (or its text, if it has no body)."""
    print(f"=== Error: {type(error).__name__} · HTTP {getattr(error, 'status_code', '-')} ===")
    body = getattr(error, "body", None)
    print(json.dumps(body, indent=2, ensure_ascii=False, default=str) if body else str(error))


def plain(obj):
    """SDK, pydantic and dataclass objects as plain dicts and lists, so that json.dumps can print them whole."""
    if hasattr(obj, "model_dump"):                                  # pydantic: OpenAI and LangChain objects
        obj = obj.model_dump()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):    # Agno objects, with every field
        obj = {field.name: getattr(obj, field.name) for field in dataclasses.fields(obj)}
    elif hasattr(obj, "to_dict") and not isinstance(obj, (dict, pd.DataFrame)):    # Agno's stopwatch in tool metrics
        obj = obj.to_dict()
    elif hasattr(obj, "dict") and not isinstance(obj, dict):        # LangGraph's store items
        obj = obj.dict()
    if isinstance(obj, dict):
        return {str(key): plain(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [plain(value) for value in obj]
    return obj


def show_json(obj, title):
    """Print an object whole, as indented JSON, under a title."""
    print(f"=== {title} ===")
    print(json.dumps(plain(obj), indent=2, ensure_ascii=False, default=str))


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


# ---------------------------------------------------------------------------------------------------- 4. the toy gateway
UPSTREAM = str(OpenAI().base_url)                                   # the provider behind the gateway
TEAM_KEYS = {"sk-team-support": "support", "sk-team-lending": "lending"}
ALIASES = {"small": SMALL, "reasoning": REASONING, "large": LARGE, "embedding": EMBEDDING}
gateway_log = []                                                    # one entry per request
before_call, after_call = [], []                                    # hooks run on every call (keep_request, keep_usage)


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


def keep_request(body, entry, headers):
    """A hook before the call that changes nothing: it keeps the whole request body in the log entry, as "sent"."""
    entry["sent"] = body


def keep_usage(answer, entry):
    """A hook after the call: it keeps the whole answer ("received") and its input tokens, and how many were cached."""
    usage = answer.get("usage") or {}
    entry["received"] = answer
    entry["input"] = usage.get("prompt_tokens", 0)
    entry["cached"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0


before_call.append(keep_request)
after_call.append(keep_usage)
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
    """The last few gateway entries as a table, with a title printed above it: the nine fields the gateway writes. The four
    that the hooks add ("sent", "received", "input", "cached") hold whole requests and answers; show_sent and show_json print them."""
    print(f"=== {title} ===")
    columns = ["id", "team", "app", "model", "status", "note", "tokens", "cost", "seconds"]
    return pd.DataFrame(gateway_log)[columns].tail(last)


def spent():
    """What this notebook has spent so far, from the gateway log."""
    return round(sum(entry["cost"] for entry in gateway_log), 4)


def show_sent(entry, title="Messages sent"):
    """Print the messages of one logged request."""
    show_messages(entry["sent"]["messages"], title)


def reply_text(entry):
    """The text of the answer in one logged request."""
    return entry["received"]["choices"][0]["message"]["content"] or ""


def find_call(app, question):
    """The last logged chat call of an app whose last message is the question."""
    for entry in reversed(gateway_log):
        messages = entry.get("sent", {}).get("messages") or [{}]
        if entry["app"] == app and messages[-1].get("content") == question:
            return entry


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

POLICIES = """NovaBank customer policies, edition 2026-10.
## Refunds
1. A card payment to a shop is cancelled by the shop, not by NovaBank. NovaBank pays the money back itself only after a dispute has been accepted. Until then, the customer should keep every receipt and every message from the shop.
2. A refund for a confirmed duplicate charge is paid within 3 working days, always to the account that was charged. The customer does not need to fill in a form for a duplicate charge; a message in the app is enough.
3. A refund is paid in the currency of the original payment. Differences caused by the exchange rate are not refunded if they are below 2 EUR. Larger differences are refunded on request.
4. A support agent may refund fees as a goodwill gesture, up to 25 EUR per customer per calendar year. A larger goodwill refund needs the approval of a team lead, and the reason is written in the customer file.
5. Refunds are never paid in cash and never to an account of another person, even if the customer asks for it in writing. A refund to a closed account is paid to the customer's new NovaBank account or by bank transfer to an account in their own name.
## Disputes
6. A dispute is opened with form DSP-1 in the app or by phone. The customer gives the date, the amount and the merchant of each payment. One form can hold up to 10 payments from the same merchant.
7. If the investigation is not finished after 10 working days, the disputed amount is credited to the account provisionally. If the dispute is later rejected, the provisional credit is taken back and the customer is told why.
8. A payment that the customer authorised, for example a subscription they forgot to cancel, is not fraud. The customer first contacts the merchant, and opens a dispute only if the merchant does not answer within 15 days.
9. Disputes over 5,000 EUR are reviewed by the fraud team within 2 working days. The card stays blocked during the review, and the customer gets a temporary virtual card in the app.
10. A customer can open at most 3 disputes in 12 months. A fourth dispute starts a call from the fraud team before anything else happens, to check that the account is safe.
## Know your customer (KYC)
11. Every new customer shows a valid passport or national identity card. A driving licence is not accepted as the only document. The document is checked with a video call or in a branch.
12. A proof of address must be less than 3 months old: a utility bill, a statement from another bank or a rental contract. Screenshots are not accepted; the customer uploads a PDF or a photo of the paper.
13. Private customers renew their KYC check every 5 years. Business customers renew it every 2 years. NovaBank sends a reminder in the app 30 days before the check expires.
14. A customer who moves to another country must repeat the KYC check within 60 days of the move, with a proof of the new address. Until then, the old address stays in the customer file.
15. An account with an expired KYC check can only receive money. Payments out are blocked until the check has been renewed, and cards stop working after 14 days.
## Fees
16. An urgent same-day transfer costs 12 EUR. A standard transfer within the euro area is free and arrives the next working day. Transfers outside the euro area cost 0.5% of the amount, with a minimum of 6 EUR.
17. A paper statement sent by post costs 2 EUR per statement. Statements in the app are free and are kept for 10 years. A copy of an old statement costs nothing if it is downloaded in the app.
18. An unarranged overdraft costs 9.9% per year on the overdrawn amount, charged at the end of each month. NovaBank sends a message in the app on the first day of an unarranged overdraft.
19. A returned direct debit costs 4 EUR. NovaBank waives this fee once per calendar year, automatically. The customer does not need to ask for it.
20. All fees are listed on the monthly statement. A customer can contest a fee within 90 days of the statement date. Fees for a product the customer never used are refunded in full.
"""
