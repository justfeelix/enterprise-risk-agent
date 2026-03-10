"""Authorization rules for agent tool usage."""

from typing import Any, Dict, Optional, TypedDict

ROLE_PERMISSIONS = {
    "customer": {"lookup_customer_account"},
    "support_agent": {"lookup_customer_account", "freeze_credit_card"},
    "admin": {
        "lookup_customer_account",
        "refund_transaction",
        "freeze_credit_card",
    },
}


class AuthorizationDecision(TypedDict):
    """Structured result of an authorization check."""

    tool: str
    status: str
    reason: Optional[str]


def authorize_tool(
    role: str,
    tool_name: str,
    arguments: Dict[str, Any],
    user_context: Optional[Dict[str, Any]] = None,
) -> AuthorizationDecision:
    """Return an authorization decision for a tool invocation."""
    allowed_tools = ROLE_PERMISSIONS.get(role, set())
    if tool_name not in allowed_tools:
        return AuthorizationDecision(
            tool=tool_name, status="denied", reason="unauthorized"
        )

    if role == "customer":
        own_account_id = str((user_context or {}).get("account_id", ""))
        requested_account_id = str(arguments.get("account_id", ""))
        if not own_account_id or requested_account_id != own_account_id:
            return AuthorizationDecision(
                tool=tool_name, status="denied", reason="unauthorized"
            )

    return AuthorizationDecision(tool=tool_name, status="authorized", reason=None)
