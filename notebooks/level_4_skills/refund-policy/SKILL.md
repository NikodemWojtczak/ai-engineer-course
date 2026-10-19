---
name: refund-policy
description: How NovaBank refunds a disputed card payment. Read it before any refund.
compatibility: Needs Python 3 to run scripts/check_refund.py.
metadata:
  owner: payments-team
  version: "1.2"
---
# Refund policy

1. Find the payment with find_transactions. Refund only a duplicate or a payment the customer did not make.
2. If the customer says they did not make the payment, read references/unauthorised-payments.md as well, and follow it.
3. Before any refund, run the script scripts/check_refund.py with the amount of that one payment in EUR as its only
   argument, for example 45.90. It prints the decision as JSON. Wait for the decision before you call refund.
4. If the decision is "refund", refund exactly refund_eur, never more. If it is "specialist", do not refund.
5. Tell the customer what tell_customer says.
