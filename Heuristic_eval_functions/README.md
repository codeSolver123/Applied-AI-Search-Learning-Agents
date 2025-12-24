Heuristic AI Agent (Rule-Based Decision Making)
Overview

This module implements a heuristic-driven artificial intelligence agent for the Antics strategy game.
The agent makes decisions using domain-specific rules and utility heuristics, without deep search or planning.

This project demonstrates how effective behavior can emerge from carefully designed heuristics, tactical rules, and state evaluation—an essential foundation for later planning, adversarial search, and learning-based agents.

Heuristic Agent

File: Peacemaker_AI.py
Agent Name: Peacemaker

Description

The Peacemaker agent relies on structured heuristics and tactical rules to manage resources, control units, and apply pressure on the opponent. Instead of predicting many moves ahead, it prioritizes robust, fast, and stable gameplay under strict runtime constraints.

The agent was designed to reliably defeat baseline opponents while remaining deterministic, safe, and self-consistent.

Key Features
Strategic Setup

Deterministic home-side layout for predictable early-game structure

Enemy food placed deep in opposing territory to delay scoring

Avoids risky midline placements

Resource Management

Maintains a minimum of two worker ants

Uses tunnels and anthill efficiently for food delivery

Routes workers using shortest-path distance estimation

Avoids unnecessary worker exposure to enemy units

Combat & Pressure

Early soldier production after economic stability

Soldiers prioritize enemy structures (anthill, tunnels)

Opportunistic attacks when adjacent to enemy targets

Pushes combat units into enemy territory instead of passive defense

Tactical Safety

Queen management avoids blocking production

Target prioritization during attacks (queen > low-health units > proximity)

Guaranteed termination without stalemates or crashes

Engineering Focus

Fully rule-based control logic (no search, no learning)

No direct access to game internals or board mutation

Deterministic, debuggable decision flow

Runs consistently as either player

Designed to scale into search-based and learning agents

Why This Matters

This agent demonstrates:

Practical heuristic design

Domain reasoning without brute-force computation

Real-time AI constraints and performance awareness

Foundations for informed search, adversarial AI, and reinforcement learning

Many production AI systems begin with rule-based controllers before introducing planning or learning layers—this project reflects that progression.

Applications

Relevant to:

Game AI foundations

Rule-based autonomous agents

Robotics behavior control

Decision systems under real-time constraints

Hybrid AI architectures (rules + learning)

Author

Alex Anderson
Electrical Engineering | Applied AI & Robotics
GitHub: codeSolver123
