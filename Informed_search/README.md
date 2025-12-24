Informed Search & Planning AI Agents
Overview

This module implements informed search–based artificial intelligence agents that make decisions by predicting and evaluating future states of the environment before acting.

Rather than reacting only to the current state, these agents plan ahead, compare multiple possible action sequences, and select moves using heuristically guided search algorithms. This demonstrates core AI techniques used in planning, robotics, and decision-making systems.

The implementations progress from shallow informed evaluation to full multi-step planning.

Best-First Search Agent

File: best_first_search_agent.py

Description

This agent evaluates all legal actions from the current state, predicts the resulting states, and selects the move with the highest heuristic value.

Key Features

Heuristic utility function normalized to a continuous score

Node-based state representation (move, state, depth, evaluation)

Best-first selection without deep tree expansion

Fast evaluation suitable for real-time decision-making

Purpose

This agent establishes the foundation for planning by demonstrating how heuristics can guide intelligent action selection without exhaustive search.

A* Planning Agent

File: astar_planning_agent.py

Description

This agent extends informed evaluation into multi-step planning using A* search. It explores future action sequences, estimates remaining cost to the goal, and selects the action that leads to the most promising long-term outcome.

Key Features

Multi-depth planning (≥ 3 moves ahead)

Admissible and consistent heuristic estimation

Frontier and expanded-node management

Parent tracing to recover the optimal initial action

Memory-bounded optimizations to control branching factor

Purpose

This agent demonstrates goal-directed planning, balancing search depth, computational efficiency, and decision quality.

Clean separation between evaluation, search, and control logic

No modification of the underlying simulation framework

Designed for extensibility into adversarial search and learning-based methods

Emphasis on predictive decision-making, not scripted behavior

Applications

The techniques demonstrated here are applicable to:

Autonomous robotics planning

Game AI and simulation agents

Decision-making systems

Search and optimization problems

Model-based AI systems

Author

Alex Anderson
Electrical Engineering | Applied AI & Robotics
GitHub: codeSolver123
