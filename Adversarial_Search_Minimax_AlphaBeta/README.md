# Adversarial Search AI Agent (Minimax with Alpha-Beta Pruning)

## Overview

This module implements an **adversarial search–based artificial intelligence agent** using the **Minimax algorithm enhanced with alpha-beta pruning**.

Unlike heuristic or informed-search agents that assume a passive environment, this agent explicitly models an **intelligent opponent** and selects actions by anticipating and countering adversarial behavior.

This demonstrates core concepts from **game theory, multi-agent systems, and strategic decision-making**, while maintaining real-time performance constraints.

---

## Minimax with Alpha-Beta Pruning Agent

**File:** `MiniMax_Gardner.py`

### Description

The agent performs a depth-limited Minimax search to evaluate future game states under optimal play by both itself and its opponent. Alpha-beta pruning is applied to significantly reduce the number of explored nodes, enabling deeper lookahead while maintaining computational efficiency.

Unlike simplified textbook examples, maximizing and minimizing decisions are determined dynamically based on **whose turn it is in the game state**, rather than strictly alternating by tree depth.

---

## Key Features

- Depth-limited Minimax search (≥ 3 plies)
- Alpha-beta pruning for aggressive search-space reduction
- Adversarial state prediction using opponent-aware transitions
- Dynamic MAX / MIN role selection based on active player
- Heuristic evaluation refined from earlier planning agents
- Move ordering and pruning optimizations for speed

---

## Algorithmic Behavior

- Explicitly models opponent decision-making
- Selects actions that maximize worst-case outcomes
- Evaluates both offensive and defensive strategies
- Prunes branches that cannot influence final decisions
- Balances search depth with strict runtime constraints

---

## Engineering Focus

- Uses adversarial state transitions instead of single-agent prediction
- Designed for scalability under branching-factor explosion
- Clean separation between evaluation, search, and control logic
- Suitable for batch simulations and self-play evaluation

---

## Applications

The techniques demonstrated in this module are applicable to:

- Competitive game AI
- Multi-agent systems
- Strategic planning and decision-making
- Adversarial optimization problems
- Autonomous agents operating in competitive environments

---

## Author

**Alex Anderson**  
Electrical Engineering | Applied AI & Robotics  
GitHub: `codeSolver123`
