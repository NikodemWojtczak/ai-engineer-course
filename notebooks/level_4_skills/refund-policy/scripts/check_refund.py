#!/usr/bin/env python3
"""NovaBank's refund check: may support refund a card payment at once, or does a specialist decide?

Usage: check_refund.py AMOUNT_EUR        for example: check_refund.py 45.90
Prints the decision as JSON. It reads no files and changes nothing: the refund itself is the agent's refund tool.
"""
import json
import sys

LIMIT_EUR = 500                                     # above it, a specialist decides

if len(sys.argv) != 2:
    sys.exit("Usage: check_refund.py AMOUNT_EUR, for example: check_refund.py 45.90")
try:
    amount = float(sys.argv[1].replace("EUR", "").replace(",", "").strip())
except ValueError:
    sys.exit(f"Not an amount in EUR: {sys.argv[1]!r}")

if amount <= LIMIT_EUR:
    decision = {"decision": "refund", "refund_eur": amount, "tell_customer": "The money arrives within 5 working days."}
else:
    decision = {"decision": "specialist", "refund_eur": 0,
                "tell_customer": "A specialist will call you back within 2 working days."}
print(json.dumps({"amount_eur": amount, "limit_eur": LIMIT_EUR, **decision}))
