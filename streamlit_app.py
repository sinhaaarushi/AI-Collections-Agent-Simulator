"""Streamlit demo for the AI Collections Agent Simulator."""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from ai_agent_simulator.agent.agent_orchestrator import (
    get_decision_factors,
    run_agent,
)


def _init_session_state() -> None:
    """Initialize Streamlit session state used by the chat demo."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"


def _render_chat(messages: List[Dict[str, Any]]) -> None:
    """Render user messages and agent responses at the top of the page."""
    st.subheader("Chat")
    if not messages:
        st.info("Send a message to start the local agent flow.")
        return

    for turn in messages:
        with st.chat_message("user"):
            st.write(turn["user_message"])
        with st.chat_message("assistant"):
            st.write(turn["result"]["response"])


def _render_decision_details(messages: List[Dict[str, Any]]) -> None:
    """Render structured decision details for each processed turn."""
    st.subheader("Decision Details")
    if not messages:
        return

    for idx, turn in enumerate(reversed(messages), start=1):
        result = turn["result"]
        with st.expander(f"Message {len(messages) - idx + 1}: {turn['user_message']}"):
            st.markdown(f"**Intent:** `{result['intent']}`")
            st.markdown(
                f"**Compliance Probability:** `{result['compliance_probability']:.2f}`"
            )
            st.markdown(f"**Decision Confidence:** `{result['decision_confidence']:.2f}`")
            st.markdown(f"**Action:** `{result['action']}`")
            st.markdown(f"**Reason:** {result['reason']}")
            st.markdown(f"**Response:** {result['response']}")
            st.markdown(f"**Strategy:** `{result['strategy']}`")
            st.markdown(f"**Outcome:** {result['outcome']}")

            st.markdown("**Decision Factors:**")
            factors = turn.get("decision_factors", {})
            col1, col2, col3 = st.columns(3)
            col1.metric("Intent", str(factors.get("intent", "unknown")))
            col2.metric("Delays", int(factors.get("num_delays", 0)))
            col3.metric("Frustrated", int(factors.get("num_frustrated", 0)))
            col4, col5 = st.columns(2)
            col4.metric("Interactions", int(factors.get("num_interactions", 0)))
            col5.metric("Compliance", f"{float(factors.get('compliance', 0.0)):.2f}")


def _render_agent_behavior(messages: List[Dict[str, Any]]) -> None:
    """Render the current adaptive strategy and latest simulated outcome."""
    st.subheader("Agent Behavior")
    if not messages:
        st.info("Agent behavior will appear after the first message.")
        return

    latest = messages[-1]["result"]
    col1, col2 = st.columns(2)
    col1.metric("Current Strategy", latest["strategy"])
    col2.metric("Action Taken", latest["action"])
    st.write(f"**Outcome:** {latest['outcome']}")


def main() -> None:
    """Run the Streamlit chat demo."""
    st.set_page_config(
        page_title="AI Collections Agent Simulator",
        layout="centered",
    )
    _init_session_state()

    st.title("AI Collections Agent Simulator")
    st.caption("Local intent classification, memory, behavior prediction, and actions.")

    st.session_state.user_id = st.text_input("User ID", st.session_state.user_id)

    _render_chat(st.session_state.messages)
    _render_agent_behavior(st.session_state.messages)

    message = st.chat_input("Type a customer message")
    if message:
        try:
            result = run_agent(st.session_state.user_id, message)
            decision_factors = get_decision_factors(st.session_state.user_id, result)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            st.session_state.messages.append(
                {
                    "user_message": message,
                    "result": result,
                    "decision_factors": decision_factors,
                }
            )
            st.rerun()

    _render_decision_details(st.session_state.messages)


if __name__ == "__main__":
    main()
