Temporal Difference Learning Agent (TD-λ)
Overview

This module implements a reinforcement learning agent using Temporal Difference learning with eligibility traces (TD(λ)) for a turn-based strategy environment.

Rather than relying on hard-coded heuristics or fixed search depth, the agent learns a value function over abstracted game states through experience, improving its decision-making performance over repeated games.

The implementation focuses on scalable state abstraction, online learning, and efficient action selection, making it suitable for large state spaces where exhaustive search is infeasible.

Core AI Techniques

Temporal Difference Learning (TD-λ)

Eligibility traces for faster credit assignment

ε-greedy exploration with decay

Feature-based state abstraction

Persistent value learning across sessions

System Architecture
State Representation

Raw game states are converted into a normalized feature vector using:

Resource progress (food counts)

Unit composition (workers, soldiers, drones)

Relative distances to key objectives

Threat and pressure indicators

This transformation is implemented in:

encodeState(state)

Continuous features are discretized using feature bucketing to reduce the effective state space while preserving strategic structure.

Learning Mechanism

The agent maintains:

V(s) — estimated utility of each state category

E(s) — eligibility traces for recent states

Updates are applied using the TD(λ) rule implemented in:

tempDiff(s, s', reward, terminal)

This allows reward signals to propagate efficiently backward through recently visited states.

Action Selection

Actions are chosen using an ε-greedy policy:

Exploration rate decays over time

Exploitation selects the move leading to the highest predicted state utility

Move evaluation is performed by predicting the next state and consulting the learned value table.

Key logic lives in:

getMove(currentState)

evaluateMove(state, move)

Persistence & Training Continuity

Learned utilities are saved to disk after each game and automatically reloaded on startup, enabling:

Long-term learning across sessions

Training interruption without loss of progress

Implemented via:

saveWeights()

loadWeights()

Design Goals

Learn effective behavior without full game tree search

Operate efficiently under strict time constraints

Scale to large state spaces through abstraction

Improve performance through experience rather than scripting

Applications

The techniques demonstrated here are applicable to:

Reinforcement learning agents

Game AI

Autonomous decision-making systems

Control policies for complex environments

Value-function–based planning systems

Status

This agent supports continued training and refinement.
Behavior improves measurably with additional gameplay and parameter tuning.

Author

Alex Anderson
Electrical Engineering | Applied AI & Robotics
GitHub: codeSolver123
