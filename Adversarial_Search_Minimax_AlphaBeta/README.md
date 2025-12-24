Adversarial Search Agent — Minimax + Alpha-Beta)
Overview

This module implements an adversarial search AI agent using depth-limited Minimax enhanced with alpha-beta pruning and practical performance optimizations. The agent explicitly models an intelligent opponent and selects actions by anticipating adversarial responses.

In addition to Minimax, the agent includes a fast tactical layer for immediate threats/opportunities and a hybrid evaluation function designed to stay stable in midgame while still reacting strongly to tactical swings.

File

MiniMax_Gardner.py

What the Agent Does
1) Tactical Layer (fast rule-based decisions)

Before running search, the agent checks for high-value “micro” actions that should happen immediately, such as:

moving the queen off the anthill if blocking production

attack-in-place if an enemy is already in range

intercepting nearby threats (e.g., drones near hill/queen)

maintaining basic economy (keeping a worker productive)

building early attackers when the hill is free

If no clear tactical move is found, the agent falls back to adversarial search.

2) Minimax Search with Alpha-Beta Pruning (depth = 3 plies)

Uses getNextStateAdversarial() to correctly simulate adversarial transitions.

Searches exactly 3 plies ahead (lookahead depth).

MAX/MIN is selected dynamically based on state.whoseTurn, not by alternating depth, which matches the game’s turn mechanics.

Evaluation Strategy (Hybrid Scoring)

This agent blends two evaluators:

A) Fast normalized utility (0..1)
A “board-light” utility that stabilizes scoring using:

food differential

hill capture progress (enemy anthill capture health)

queen survival

attacker proximity pressure

worker/carrying signals

queen safety penalties (threat / blocking hill)

B) Stronger symmetric score (swing-sensitive)
A higher-variance evaluator that emphasizes tactical advantage:

pressure on enemy queen/hill

worker harassment and defense

zone control near key structures

army strength and threat penalties

additional strategic shaping signals

Final eval used in search: a weighted blend so search is tactical but doesn’t thrash.

Performance Optimizations Implemented

This file is designed to run quickly despite branching growth:

Alpha-beta pruning (cuts branches that cannot affect the final decision)

Move ordering (scores children early so pruning happens sooner)

Killer-move heuristic (prioritizes moves that previously caused cutoffs)

END-move ordering (pushes END toward the back when other actions exist)

Top-N% child filtering (KEEP_TOP_PCT) to reduce branching factor

Hard child cap (MAX_CHILDREN) to enforce runtime bounds

Small tie jitter to reduce loops / repetitive behavior

These are practical techniques used in real adversarial search systems to keep decision-making within time constraints.

Why This Matters (Employer Context)

This module demonstrates applied AI engineering skills in:

adversarial planning / multi-agent decision-making

alpha-beta pruning + search-space reduction

evaluation function design and shaping

performance tuning under branching-factor explosion

combining fast heuristics with deeper strategic lookahead

Author

Alex Anderson (with Carter Rhoades)
Electrical Engineering | Applied AI & Robotics
GitHub: codeSolver123
