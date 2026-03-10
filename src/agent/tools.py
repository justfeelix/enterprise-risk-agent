"""Simulated enterprise tools for the agent runtime."""

from typing import Any, Dict


def lookup_customer_account(account_id: str) -> Dict[str, Any]:
    """Return a simulated customer account record."""
    return {
        "tool": "lookup_customer_account",
        "status": "executed",
        "account_id": account_id,
        "account_status": "active",
        "card_status": "active",
        "available_balance": 1842.55,
        "currency": "EUR",
    }


def refund_transaction(account_id: str, amount: float) -> Dict[str, Any]:
    """Return a simulated refund execution result."""
    return {
        "tool": "refund_transaction",
        "status": "executed",
        "account_id": account_id,
        "amount": amount,
        "currency": "EUR",
    }


def freeze_credit_card(account_id: str) -> Dict[str, Any]:
    """Return a simulated card-freeze result."""
    return {
        "tool": "freeze_credit_card",
        "status": "executed",
        "account_id": account_id,
        "card_status": "frozen",
    }


TOOL_REGISTRY = {
    "lookup_customer_account": lookup_customer_account,
    "refund_transaction": refund_transaction,
    "freeze_credit_card": freeze_credit_card,
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a named tool with structured arguments."""
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return {
            "tool": tool_name,
            "status": "denied",
            "reason": "unknown_tool",
        }
    try:
        return tool(**arguments)
    except TypeError:
        return {
            "tool": tool_name,
            "status": "denied",
            "reason": "invalid_arguments",
            "arguments": arguments,
        }
