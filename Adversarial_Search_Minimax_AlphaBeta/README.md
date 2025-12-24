# Adversarial Search AI Agent (Minimax with Alpha-Beta Pruning)

## Overview

This module implements an **adversarial search–based artificial intelligence agent** using the **Minimax algorithm enhanced with alpha-beta pruning**.

Unlike heuristic or informed-search agents that assume a passive environment, this agent explicitly models an **intelligent opponent** and selects actions by anticipating and countering adversarial behavior.

The result is a decision-making system capable of **strategic planning under competition**, a core concept in game theory, multi-agent systems, and competitive AI.

---

## Minimax with Alpha-Beta Pruning Agent

**File:** `minimax_alpha_beta_agent.py`

### Description

The agent performs a depth-limited Minimax search to evaluate future game states under optimal play by both itself and its opponent. Alpha-beta pruning is applied to significantly reduce the number of explored nodes, enabling deeper lookahead while maintaining performance.

Unlike traditional textbook examples, maximizing and minimizing decisions are determined dynamically based on **whose turn it is in the game state**, not purely on tree depth.

---

## Key Features

- Depth-limited Minimax search (≥ 3 plies)
- Alpha-beta pruning for search-space reduction
- Adversarial state prediction using opponent-aware transitions
- Dynamic MAX / MIN role assignment based on active player
- Heuristic evaluation reused and refined from earlier agents
- Move ordering and pruning optimizations for speed

---

## Algorithmic Behavior

- Models the opponent as a rational decision-maker
- Evaluates both offensive and defensive outcomes
- Selects actions that maximize worst-case utility
- Prunes branches that cannot affect the final decision
- Balances search depth with real-time performance constraints

---

## Engineering Focus

- Uses adversarial state transitions rather than single-agent prediction
- Designed for scalability and performance under tight time limits
- Clean separation between evaluation, search logic, and control flow
- Compatible with batch simulations and self-play evaluation

---

## Applications

The techniques demonstrated in this module are directly applicable to:

- Competitive game AI
- Multi-agent systems
- Strategic planning under uncertainty
- Decision-making in adversarial environments
- Optimization problems involving competing objectives

---

## Author

**Alex Anderson**  
Electrical Engineering | Applied AI & Robotics  
GitHub: `codeSolver123`
