# AI Collections Agent Simulator

A local AI agent system that models user interactions, predicts behavior, and makes real-time decisions using intent detection, memory, and machine learning.

## Problem Statement

Real-world AI systems in fintech and support cannot rely on one-off message classification alone. They need to:

* Understand what a user is asking for
* Track behavior across interactions
* Predict likely outcomes
* Choose the next action dynamically

This project models that workflow in a fully local environment. It combines intent detection, persistent memory, behavioral prediction, explainable decision rules, and simulated actions without relying on external services.

## System Overview

```text
User Input
-> Intent Classification (rule-based + sentiment)
-> Memory (SQLite interaction history)
-> Behavior Prediction (Logistic Regression)
-> Decision Engine (rule-based reasoning)
-> Action Handler (simulated system actions)
-> Response (CLI / Streamlit UI)
```

The result is a small but complete agent loop.

Each message is classified, stored, analyzed against user history, scored for compliance likelihood, routed through a decision engine, and converted into a concrete system response.

## Demo

Run the Streamlit app to interact with the AI agent and view decision-making in real time.

![AI Collections Agent Simulator demo](assets/demo-screenshot.svg)

## Architecture

| Module | Responsibility |
| ------ | -------------- |
| `intent_classifier.py` | Detects user intent and sentiment signals from local rule-based scoring. |
| `memory_manager.py` | Persists user interaction history in SQLite. |
| `behavior_model.py` | Trains a local logistic regression model and predicts compliance probability. |
| `decision_engine.py` | Selects actions using intent, behavior signals, and compliance probability. |
| `action_handler.py` | Simulates system actions such as reminders, human escalation, and user assistance. |
| `agent_orchestrator.py` | Controls the full pipeline from message input to final response. |
| `streamlit_app.py` | Provides an interactive chat UI with decision details and trace output. |

## Example Scenario

```text
Input: "I will pay later"
Intent: delaying
Compliance: 0.32
Action: send_reminder
Reason: repeated delay behavior with low compliance probability
```

## Key Features

* Intent classification with sentiment awareness
* Persistent memory using SQLite
* Behavioral prediction using machine learning
* Decision-making engine for action selection
* Explainable reasoning and trace outputs
* Interactive UI using Streamlit
* Local-first architecture with no external APIs

## Design Decisions

**Rule-based intent first:** The intent layer is deterministic, fast, and easy to inspect. For a simulator, transparent behavior is more valuable than opaque classification.

**Logistic regression for behavior prediction:** Compliance prediction uses a lightweight model that trains quickly on synthetic data, exposes meaningful probabilities, and stays easy to reason about.

**SQLite for memory:** SQLite keeps the system local, persistent, and simple to run. It also mirrors the core requirement of tracking user behavior over time without needing infrastructure.

**Modular architecture:** Intent classification, memory, prediction, decisions, actions, orchestration, and UI are separated so each part can evolve independently.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the interactive UI:

```bash
streamlit run streamlit_app.py
```

Run the CLI:

```bash
python -m ai_agent_simulator.app
```

Use a custom user or memory database:

```bash
python -m ai_agent_simulator.app --user-id alice --db ./memory.db
```
