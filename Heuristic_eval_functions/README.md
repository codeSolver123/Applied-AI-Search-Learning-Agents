Heuristic AI Agent (Rule-Based Decision Making)
Overview

This project implements a heuristic-based artificial intelligence agent for a turn-based strategy environment.
The agent makes decisions using hand-crafted evaluation rules rather than search or learning, serving as a baseline for more advanced AI techniques.

This work demonstrates how domain knowledge can be encoded into an AI system to produce reliable, efficient behavior without expensive computation.

AI Approach

The agent evaluates game states using a heuristic utility function that estimates how favorable a position is for the player.

Key characteristics:

Rule-based decision making

Deterministic behavior (no learning required)

Extremely fast execution

Stable performance across repeated games

The heuristic balances multiple objectives such as:

Resource collection efficiency

Unit survival and positioning

Avoidance of high-risk situations

Controlled offensive pressure

This design avoids deep search and instead prioritizes consistent, explainable decisions.

Heuristic Design

The utility function returns a normalized score representing how “good” the current state is for the agent.

Factors considered include:

Current and relative resource levels

Presence or absence of key units

Proximity to objectives

Risk exposure to enemy units

The heuristic is designed to:

Return values near 0.5 at the start of the game

Increase toward 1.0 as winning conditions approach

Decrease toward 0.0 in losing positions

This structure allows smooth decision gradients rather than abrupt behavior changes.

Performance Goals

This agent was designed to:

Reliably defeat baseline and random agents

Avoid crashes, deadlocks, or infinite loops

Play consistently as either player

Complete games quickly enough for batch testing

It serves as a baseline AI for comparison with more advanced approaches such as search-based and learning-based agents.

Role in the Larger Project

This heuristic agent establishes:

A performance baseline

A reference utility function

A fallback decision system

Later agents in this repository build on this work by replacing or augmenting the heuristic with:

Informed search (A*, minimax)

Neural network evaluation

Reinforcement and temporal-difference learning

Files

hw1_peacemaker_anderale26.py
Heuristic AI agent implementation

Only agent logic is included. The underlying simulation environment remains unchanged.

Skills Demonstrated

Heuristic design

AI decision modeling

Performance-constrained programming

Debugging autonomous agents

Clean separation of logic and environment

Author

Alex Anderson
Electrical Engineering — AI & Robotics Focus
GitHub: codeSolver123
