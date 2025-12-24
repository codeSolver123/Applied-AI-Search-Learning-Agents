Best-First “BFS” (your BFS_search-Random.py)
Informed Evaluation Agent (Fast Best-First Selection)

File: BFS_search-Random.py

Overview

This agent demonstrates heuristic-guided action selection using a fast evaluation function and a simple node abstraction. Each turn, it simulates candidate actions, scores the resulting states, and selects the move that looks best without deep tree expansion, enabling real-time play.

While the filename includes “BFS”, the implementation is best described as a one-step best-first evaluator (greedy selection over next-state utility), augmented with practical tactical rules.

Key Behaviors Implemented

Heuristic state utility normalized to [0, 1] (no direct board-array dependence for evaluation)

One-ply lookahead: evaluates each legal action by simulating the next state

Node structure (move, state, depth, eval, parent) for clean scoring and selection

Tactical layer before scoring:

moves queen off the anthill to avoid blocking builds

builds 1–2 SOLDIERs early (tempo)

“move-in-place” to trigger attacks when already in range

drives attackers toward enemy queen / hill

maintains basic worker food loop

Engineering Focus

Designed for fast decision-making under tight runtime constraints

Clear separation between:

evaluation (utility)

simulation (getNextState)

selection (_bestNode)

Uses lightweight heuristics that generalize to search/planning systems

Author

Alex Anderson (with Reiss Oliveros)
GitHub: codeSolver123

README for A* Planning (your Astar_Search_Ghost.py)
A* Planning Agent (Bounded Lookahead + Tactical Layer)

File: Astar_Search_Ghost.py

Overview

This agent performs bounded A* planning to select actions by simulating multi-step sequences and choosing the path that appears to reach winning objectives sooner. It combines:

a fast heuristic utility for scoring states,

a tactical “quick rule” layer for obvious moves, and

A* search planned to a short horizon (depth-limited) with strict memory/runtime caps.

This demonstrates practical planning methods used in robotics and real-time decision systems: search, heuristics, pruning, and bounded computation.

Planning Strategy

Early-game: uses one-ply evaluation until enough attackers exist (keeps turns fast)

Midgame: runs A* to a fixed horizon:

depth-limited to 3 of the agent’s moves

selects the frontier node with lowest f = g + h

reconstructs the first action via parent pointers

Heuristic Design (A*)

The heuristic h_cost(state) estimates remaining effort using blended signals:

remaining enemy hill capture progress

approximate turns to reach priority targets (enemy queen/hill) using attacker distance

remaining food needed to hit the goal (optimistic estimate)

small threat-awareness adjustment (encourages responding to nearby pressure)

Performance Safeguards

To prevent “freezing” or runaway branching, the A* loop is bounded by:

ASTAR_MAX_EXPANSIONS (max node expansions per turn)

ASTAR_FRONTIER_LIMIT (memory-bound open list size)

ASTAR_CHILD_LIMIT (per-expansion child pruning)

These are the kinds of constraints you’d add in real systems where planning must fit in a time budget.

Tactical Layer (Before Search)

The agent includes fast checks to avoid unnecessary planning:

queen-off-hill movement to unblock production

attack-in-place when already in range

early soldier production for tempo

intercept logic for immediate threats

target prioritization (carrying worker → queen → tunnel/hill → threats)

basic worker food collection loop

Engineering Focus

Demonstrates heuristic planning, bounded search, and parent-trace action recovery

Clean separation between:

evaluation (utility)

heuristic (h_cost)

node construction (makeNode_Astar)

expansion (expandNode)

control logic (getMove)

Author

Alex Anderson (with Reiss Oliveros)
GitHub: codeSolver123
