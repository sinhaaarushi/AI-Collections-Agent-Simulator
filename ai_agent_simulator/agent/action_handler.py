"""Simulated action handlers for the collections agent.

These functions represent the side effects a production system might perform
after the decision engine chooses an action. For this local simulator they
return clean messages instead of calling external services.
"""

from __future__ import annotations


def send_reminder(user_id: str) -> str:
    """Simulate sending a reminder to the user."""
    return send_reminder_soft(user_id)


def send_reminder_soft(user_id: str) -> str:
    """Simulate a low-pressure reminder for an early delay."""
    return f"Reminder sent to user '{user_id}' regarding the pending action."


def send_reminder_firm(user_id: str) -> str:
    """Simulate a firmer reminder after repeated delay behavior."""
    return (
        f"Firm reminder sent to user '{user_id}' with a clear follow-up timeline."
    )


def escalate_to_human(user_id: str) -> str:
    """Simulate creating a human-review handoff."""
    return f"User '{user_id}' has been escalated to a human collections specialist."


def assist_user() -> str:
    """Simulate continuing the automated support flow."""
    return "Assistance provided with next steps and payment support options."


def standard_response() -> str:
    """Simulate a default agent response."""
    return "Standard response provided with account guidance and next steps."


def execute_action(action: str, user_id: str) -> str:
    """Execute the named action and return the user-facing response message."""
    handlers = {
        "send_reminder": lambda: send_reminder(user_id),
        "send_reminder_soft": lambda: send_reminder_soft(user_id),
        "send_reminder_firm": lambda: send_reminder_firm(user_id),
        "escalate_to_human": lambda: escalate_to_human(user_id),
        "assist_user": assist_user,
        "standard_response": standard_response,
    }
    handler = handlers.get(action, standard_response)
    return handler()
