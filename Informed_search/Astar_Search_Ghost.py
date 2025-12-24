# Alex Anderson
# Reiss Oliveros

# ==============================
# G H O S T   A I   (with comments for beginners)
# ==============================
#
# ------------------
#   1) Use a *fast heuristic utility* that estimates how good a GameState is (Part A).
#   2) Build simple *nodes* and pick the best one with that utility (Part A).
#   3) Run *A* search to depth 3 (three of OUR moves), which
#      chooses a path of actions that seems to reach the goal sooner (Part B).
#
# In front of that, we also use a few "tactical rules" (quick checks) so we don't waste
# time searching when there’s an obvious good move (e.g., attack in place, step queen off hill).
#
# - Lots of small helper functions to increase winning chances of agent

import random
import math
import sys
import time  
sys.path.append("..")  # allow Python to import modules that live in the parent folder

# Import ReAntics engine helpers and constants
from Player import *           # base Player class we must subclass
from Constants import *        # constants like ANTHILL, TUNNEL, etc.
from Construction import CONSTR_STATS
from Ant import UNIT_STATS     # stats per ant type (movement, range, cost, etc.)
from Move import Move          # how we represent a move to the engine
from GameState import *        # access to GameState fields (inventories, board, etc.)
from AIPlayerUtils import *    # utility helpers (listAllLegalMoves, getNextState, etc.)

# ---------------------------------------------------------------------
# Performance knobs (you can tweak these if the AI is too slow/fast)
# ---------------------------------------------------------------------
PROFILE_ASTAR = False          # set True to print timing locally
ASTAR_MAX_EXPANSIONS = 280     # max nodes we will expand during A* this turn (prevents freezing encountered in intial tests)
ASTAR_FRONTIER_LIMIT = 90      # maximum "open list" size for A* (memory-bound)
ASTAR_CHILD_LIMIT = 16         # per expansion, keep at most this many child nodes (prune)

#
class AIPlayer(Player):
    def __init__(self, inputPlayerId):
        # name of my agent
        super(AIPlayer, self).__init__(inputPlayerId, "Ghost")

    # =================================================================
    # Setup (where to place our starting pieces)
    # =================================================================
    def getPlacement(self, currentState):
        """
        Called during the game-setup phases. Keeps layout random. Once for our side and once to place
        the two enemy foods on their side.
        """
        if currentState.phase == SETUP_PHASE_1:    # placing our side’s pieces
            numToPlace = 11
            moves = []
            for _ in range(numToPlace):
                move = None
                # Try random coordinates until we find an empty spot on rows 0..3
                while move is None:
                    x = random.randint(0, 9)
                    y = random.randint(0, 3)
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        # The engine uses this "occupied" flip to prevent duplicates
                        currentState.board[x][y].constr = True
                moves.append(move)
            return moves

        elif currentState.phase == SETUP_PHASE_2:  # placing enemy food randomly (required)
            numToPlace = 2
            moves = []
            for _ in range(numToPlace):
                move = None
                while move is None:
                    x = random.randint(0, 9)
                    y = random.randint(6, 9)
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr = True
                moves.append(move)
            return moves

        # Safety fallback 
        return [(0, 0)]

    # =================================================================
    # Part A: Utility function — estimate "how good" a state is (0..1)
    # =================================================================

    @staticmethod
    def _clamp01(v):
        """Clamp any number into the [0,1] range to keep utility safe."""
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    def utility(self, state):
        """
          - ~0.5 at the start of game
          - near 1.0 when we’re almost winning
          - near 0.0 when we’re losing badly

        IMPORTANT: We do NOT use state.board[x][y] directly (per the spec),
        because later they may swap GameState with a "fast clone" lacking the board.
        Instead, we rely on high-level info in inventories
        """
        # If the engine already knows a winner, return a near-extreme value quickly.
        w = getWinner(state)
        if w == 1:  # we won!
            return 0.999
        if w == 0:  # they won :(
            return 0.001

        # Who am I? Who is the opponent?
        me  = state.whoseTurn
        opp = 1 - me
        myInv = state.inventories[me]
        enInv = state.inventories[opp]

        # Food lead: simple normalized difference (ours - theirs) / FOOD_GOAL
        # If FOOD_GOAL is small, protect against division by zero by min=1
        fg = FOOD_GOAL if FOOD_GOAL >= 1 else 1
        food_term = (myInv.foodCount - enInv.foodCount) / float(fg)

        # Anthills & queens: existence and capture progress help us evaluate
        myHill = myInv.getAnthill()
        enHill = enInv.getAnthill()
        myQ    = myInv.getQueen()
        enQ    = enInv.getQueen()

        # Enemy hill capture progress: captureHealth is 3 (new) down to 0 (captured).
        # Convert to our rating scale 0..1 progress where 1 means "captured/destroyed".
        if enHill is None:
            hill_term = 1.0
        else:
            ch = enHill.captureHealth
            if ch < 0:
                ch = 0.0
            if ch > 3.0:
                ch = 3.0
            hill_term = 1.0 - (ch / 3.0)

        # Queen status: +1 if their queen is gone, -1 if our queen is gone.
        queen_term = 0.0
        if enQ is None:
            queen_term += 1.0
        if myQ is None:
            queen_term -= 1.0

        # Proximity pressure: our attackers close to their queen/hill is good.
        attackers = getAntList(state, me, (DRONE, SOLDIER, R_SOLDIER))

        def prox_to(target_coords):
            """Return a 0..1 score: 1 if target is "handled" or no target; 0 if far."""
            if target_coords is None:
                return 1.0           # no target => treat as "done"
            if not attackers:
                return 0.0           # we have no attackers => no pressure
            # Find the nearest attacker (smallest approx distance)
            dmin = None
            for a in attackers:
                d = approxDist(a.coords, target_coords)  # engine helper
                if dmin is None or d < dmin:
                    dmin = d
            # Map "small distance" => value near 1.0; "large distance" => value near 0.0
            val = (10.0 - float(dmin)) / 10.0
            return self._clamp01(val)

        # Pressure toward enemy queen and hill (if they exist)
        fight_q_term = prox_to(enQ.coords if enQ else None)
        fight_h_term = prox_to(enHill.coords if enHill else None)

        # Small nudge for workers (if they’re carrying, we’re progressing on food)
        workers = getAntList(state, me, (WORKER,))
        work_term = 0.0
        if workers:
            carrying = 0
            for wkr in workers:
                if wkr.carrying:
                    carrying += 1
            # At most 0.5 bump from workers adjusted to beat booger etc.
            work_term = self._clamp01(0.25 * carrying + 0.25)

        # Penalties if our queen is threatened or blocking our hill
        q_threat_pen = 0.0
        q_block_pen  = 0.0
        if myQ is not None:
            enemy_attackers = getAntList(state, opp, (DRONE, SOLDIER, R_SOLDIER))
            # Any enemy attacker within distance 1? Penalize a bit.
            for a in enemy_attackers:
                if approxDist(a.coords, myQ.coords) <= 1:
                    q_threat_pen += 0.25
                    break
            # If queen is literally on top of our hill, we might be blocking builds.
            if myHill is not None and myQ.coords == myHill.coords:
                q_block_pen += 0.15

        # Weighted sum:
        # - 0.5 baseline so "start of game" utilities hover near 0.5
        # - bigger weights to queen/hill progress and proximity pressure
        score = (
            0.5
            + 0.50 * queen_term
            + 0.30 * hill_term
            + 0.20 * fight_q_term
            + 0.12 * fight_h_term
            + 0.08 * food_term
            + 0.04 * work_term
            - q_threat_pen
            - q_block_pen
        )
        return self._clamp01(score)

    # =================================================================
    # Part A: Node helpers (for single-ply evaluation)
    # =================================================================
    def makeNode(self, parent, move, childState, depth=1):
        """
        Build a node dictionary for Part A:
          - 'move'  : the move that led here (from parent)
          - 'state' : the resulting GameState after that move
          - 'depth' : depth from the *real current state*. For Part A, always set to 1.
          - 'eval'  : utility(state) + depth   (assignment asks for this)
          - 'parent': reference to the parent node (not used in Part A, yes in Part B)
        """
        return {
            "move": move,
            "state": childState,
            "depth": depth,
            "eval": self.utility(childState) + depth,
            "parent": parent,
        }

    def _bestNode(self, nodes):
        """
        Return the highest 'eval' node. If there are ties, pick randomly among them
        to avoid getting stuck in loops like what happened in earlier tests of ours.
        """
        if not nodes:
            return None
        best_val = None
        best_list = []
        for n in nodes:
            ev = n["eval"]
            if (best_val is None) or (ev > best_val):
                best_val = ev
                best_list = [n]
            elif ev == best_val:
                best_list.append(n)
        if not best_list:
            return None
        return random.choice(best_list)

    # =================================================================
    # Part B: A* helpers (cost-to-go heuristic, expansion, etc.)
    # =================================================================
    def h_cost(self, state):
        """
        Heuristic estimate for A* (lower is better).
        The assignment wants "estimated number of moves to goal", so we combine:
          • Remaining enemy hill capture ticks (0..3).
          • Steps (roughly) for our nearest attacker to reach an important target.
          • Food moves needed (2 moves per food: pick + drop) to hit FOOD_GOAL.
        We *do not* access board[ ][ ] directly (Part A)
        """
        me  = state.whoseTurn
        opp = 1 - me
        myInv = state.inventories[me]
        enInv = state.inventories[opp]

        # Food progress: optimistic 2 moves per food (1 to pick, 1 to drop).
        food_needed = FOOD_GOAL - myInv.foodCount
        if food_needed < 0:
            food_needed = 0
        food_moves  = 2 * food_needed

        # Enemy hill capture remaining (0 if already gone)
        enHill = enInv.getAnthill()
        hill_left = 0
        if enHill is not None:
            ch = int(enHill.captureHealth)
            if ch < 0:
                ch = 0
            hill_left = ch

        # Choose the best "attackable" target: enemy queen first, else their hill
        target_coords = None
        enQ = enInv.getQueen()
        if enQ is not None:
            target_coords = enQ.coords
        elif enHill is not None:
            target_coords = enHill.coords

        # Distance in "turns" to reach target with our nearest attacker.
        attackers = getAntList(state, me, (DRONE, SOLDIER, R_SOLDIER))
        dist_moves = 0
        if target_coords is not None and attackers:
            dmin = None
            for a in attackers:
                d = approxDist(a.coords, target_coords)
                if dmin is None or d < dmin:
                    dmin = d
            # Most attackers move ~2 per turn, so we divide by 2
            avg_steps = 2.0
            dist_moves = int(math.ceil(dmin / avg_steps))

        # Slight urgency bonus: if enemy drones are near our hill/queen, "pretend"
        # we are 1 move closer to encourage quicker answers (reduce h by up to 1).
        myHill = myInv.getAnthill()
        myQ    = myInv.getQueen()
        enemy_drones = getAntList(state, opp, (DRONE,))
        threat = 0
        if enemy_drones:
            for d in enemy_drones:
                if myQ is not None and approxDist(d.coords, myQ.coords) <= 3:
                    threat += 1
                    break
            if myHill is not None:
                for d in enemy_drones:
                    if approxDist(d.coords, myHill.coords) <= 3:
                        threat += 1
                        break

        # Heuristic value must be ≥ 0.
        return max(0, hill_left + dist_moves + food_moves - min(1, threat))

    def makeNode_Astar(self, parent, move, childState, depth):
        """
        Build an A* node:
          - 'f' = g + h  (g = depth in OUR atomic moves, h = h_cost(childState))
        """
        return {
            "move": move,
            "state": childState,
            "depth": depth,                         # g
            "f": depth + self.h_cost(childState),   # f = g + h
            "parent": parent,
        }

    def expandNode(self, node):
        """
        Create children by applying all OUR legal moves to node["state"].
        We intentionally prune to ASTAR_CHILD_LIMIT children to keep A* fast.
        (We do *not* use the adversarial getNextStateAdversarial() per the spec.)
        """
        moves = listAllLegalMoves(node["state"])
        kids = []
        count = 0
        for mv in moves:
            st = getNextState(node["state"], mv)  # fast simulation: apply our move to get new state
            kids.append(self.makeNode_Astar(node, mv, st, node["depth"] + 1))
            count += 1
            if count >= ASTAR_CHILD_LIMIT:
                break
        return kids

    # =================================================================
    # Small tactical helpers (quick rules so we don’t always search)
    # =================================================================
    def _safe_neighbors(self, state, coord):
        """
        Return a small list of adjacent empty squares we could step to.
        We prefer staying nearer y=0..3 (our home rows) when possible.
        """
        x, y = coord
        candidates = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        legal = []
        for c in candidates:
            cx, cy = c
            # stay on the board
            if 0 <= cx < 10 and 0 <= cy < 10:
                # free of ants?
                if state.board[cx][cy].ant is None:
                    legal.append(c)

        # Sort: prefer smaller y (closer to our home), then smaller x-distance change
        def neighbor_key(c):
            return (c[1], abs(c[0] - x))

        # Simple insertion sort (avoids introducing Python's key functions early)
        for i in range(1, len(legal)):
            key_item = legal[i]
            j = i - 1
            while j >= 0 and neighbor_key(legal[j]) > neighbor_key(key_item):
                legal[j+1] = legal[j]
                j -= 1
            legal[j+1] = key_item

        return legal

    def _move_in_place_to_attack(self, state, ant):
        """
        If this ant already has a target in range, we can "move in place"
        (i.e., give a Move that keeps the same coords). In ReAntics, that still
        lets the ant perform an attack if something is in range.
        """
        me  = state.whoseTurn
        opp = 1 - me
        rng = UNIT_STATS[ant.type][RANGE]
        if rng <= 0:
            return None
        for enemy in state.inventories[opp].ants:
            if approxDist(ant.coords, enemy.coords) <= rng:
                return Move(MOVE_ANT, [ant.coords], None)
        return None

    def _path_toward(self, state, start, dest, steps):
        """
        Build a step-by-step path from start toward dest for up to 'steps'
        using the engine helper createPathToward.
        """
        path = createPathToward(state, start, dest, steps)
        if not path:
            return None
        return Move(MOVE_ANT, path, None)

    def _closest_ant_to(self, ants, target):
        """
        Given a list of ants, return the one with the smallest approx distance to target.
        """
        best = None
        best_d = None
        for a in ants:
            d = approxDist(a.coords, target)
            if best is None or d < best_d:
                best = a
                best_d = d
        return best

    # =================================================================
    # Main decision function: choose a move each turn
    # (Tactics -> 1-ply Part A -> bounded A* Part B)
    # =================================================================
    def getMove(self, currentState):
        """
        This is called by the engine on our turn. We try, in order:
          1) Tactical rules (fast checks: attack in place, step queen off hill, build soldiers).
          2) If not enough attackers yet, do a quick 1-ply Part A choice (utility-based).
          3) Otherwise, run a small A* to depth 3 with memory/time bounds.
        """
        me  = currentState.whoseTurn
        opp = 1 - me
        myInv = currentState.inventories[me]
        enInv = currentState.inventories[opp]
        myHill = myInv.getAnthill()
        enHill = enInv.getAnthill()
        myQ    = myInv.getQueen()
        enQ    = enInv.getQueen()

        # ---- Tactical: queen management and instant attacks ----
        if myQ is not None:
            # If queen is sitting on our hill and hasn't moved yet, try to step off so we can build
            if myHill is not None and myQ.coords == myHill.coords and not myQ.hasMoved:
                neigh = self._safe_neighbors(currentState, myQ.coords)
                reach = listReachableAdjacent(currentState, myQ.coords, UNIT_STATS[QUEEN][MOVEMENT])
                for c in neigh:
                    # Prefer staying within rows 0..3 (home)
                    if c[1] <= 3:
                        for r in reach:
                            if r == c:
                                return Move(MOVE_ANT, [myQ.coords, c], None)
                # Otherwise, if queen can already attack, do so
                mip = self._move_in_place_to_attack(currentState, myQ)
                if mip:
                    return mip

            # If queen already has a target in range, attack-in-place
            mip = self._move_in_place_to_attack(currentState, myQ)
            if mip:
                return mip

        # ---- Build attackers (we favor SOLDIERs) ----
        # Make one list of our attackers, but we prefer SOLDIERs first
        attackers = getAntList(currentState, me, (SOLDIER,)) \
                  + getAntList(currentState, me, (R_SOLDIER,)) \
                  + getAntList(currentState, me, (DRONE,))
        # Re-order: SOLDIERs, then R_SOLDIERs, then DRONEs
        ordered_attackers = []
        for t in (SOLDIER, R_SOLDIER, DRONE):
            for a in getAntList(currentState, me, (t,)):
                ordered_attackers.append(a)
        attackers = ordered_attackers

        # If our anthill tile is free and we have enough food, build SOLDIERs until we have 2
        hill_free = (myHill is not None and getAntAt(currentState, myHill.coords) is None)
        if hill_free and myInv.foodCount >= UNIT_STATS[SOLDIER][COST]:
            num_soldiers = len(getAntList(currentState, me, (SOLDIER, R_SOLDIER)))
            if num_soldiers < 2:
                return Move(BUILD, [myHill.coords], SOLDIER)

        # ---- Safer drone intercept (vs Booger): only chase if it's actually nearby ----
        if myQ is not None or myHill is not None:
            enemy_drones = getAntList(currentState, opp, (DRONE,))
            danger_here = None
            for d in enemy_drones:
                nearQ = (myQ is not None and approxDist(d.coords, myQ.coords) <= 2)
                nearH = (myHill is not None and approxDist(d.coords, myHill.coords) <= 2)
                if nearQ or nearH:
                    danger_here = d.coords
                    break
            if danger_here is not None:
                # Send nearest attacker if we can actually move toward danger this turn
                movable = []
                for a in attackers:
                    if not a.hasMoved:
                        movable.append(a)
                if movable:
                    mover = self._closest_ant_to(movable, danger_here)
                    if mover is not None:
                        steps = UNIT_STATS[mover.type][MOVEMENT]
                        if approxDist(mover.coords, danger_here) > 0:
                            mv = self._path_toward(currentState, mover.coords, danger_here, steps)
                            if mv:
                                return mv
                # Also let queen attack in place if in range
                if myQ is not None:
                    mip = self._move_in_place_to_attack(currentState, myQ)
                    if mip:
                        return mip

        # ---- If any attacker already has a target in range, attack in place now ----
        for a in attackers:
            if a.hasMoved:
                continue
            mip = self._move_in_place_to_attack(currentState, a)
            if mip:
                return mip

        # ---- Target priority (general + anti-Booger) ----
        # 1) enemy worker carrying food (stop their scoring)
        # 2) enemy queen if realistically reachable soon (within ~5)
        # 3) enemy drop sites: tunnel, then hill (disrupt delivery)
        # 4) enemy worker not carrying (prefer nearer to our hill)
        # 5) nearest enemy attacker
        target = None

        # (1) carrying worker
        enemy_workers = getAntList(currentState, opp, (WORKER,))
        best_carry = None
        for ew in enemy_workers:
            if ew.carrying:
                best_carry = ew
                break
        if best_carry is not None:
            target = best_carry.coords

        # (2) enemy queen only if our attackers can get there soon
        enQ = enInv.getQueen()
        if target is None and enQ is not None:
            can_reach = False
            for a in attackers:
                if approxDist(a.coords, enQ.coords) <= 5:
                    can_reach = True
                    break
            if can_reach:
                target = enQ.coords

        # (3) enemy tunnel (first) then enemy hill
        if target is None:
            enemy_tunnels = getConstrList(currentState, opp, (TUNNEL,))
            if enemy_tunnels:
                target = enemy_tunnels[0].coords
        if target is None and enHill is not None:
            target = enHill.coords

        # (4) worker not carrying — pick the one closest to our hill if we have one
        if target is None and len(enemy_workers) > 0:
            if myHill is not None:
                closest = None
                bestd = None
                for ew in enemy_workers:
                    if ew.carrying:
                        continue
                    dd = approxDist(ew.coords, myHill.coords)
                    if closest is None or dd < bestd:
                        closest = ew
                        bestd = dd
                if closest is not None:
                    target = closest.coords
            else:
                # Otherwise, just pick the first non-carrying one we see
                for ew in enemy_workers:
                    if not ew.carrying:
                        target = ew.coords
                        break

        # (5) nothing above? move toward nearest enemy attacker to pressure them
        if target is None:
            enemy_attackers = getAntList(currentState, opp, (SOLDIER, R_SOLDIER, DRONE))
            if enemy_attackers:
                # Anchor = (priority checker) to protect: our hill first,
                #  else our queen, else just pick
                anchor = None
                if myHill is not None:
                    anchor = myHill.coords
                elif myQ is not None:
                    anchor = myQ.coords
                if anchor is None:
                    target = enemy_attackers[0].coords
                else:
                    best_e = None
                    best_d = None
                    for e in enemy_attackers:
                        dd = approxDist(e.coords, anchor)
                        if best_e is None or dd < best_d:
                            best_e = e
                            best_d = dd
                    target = best_e.coords

        # If we have a target, move the nearest available attacker toward it
        if target is not None and attackers:
            movable = []
            for aa in attackers:
                if not aa.hasMoved:
                    movable.append(aa)
            if movable:
                mover = self._closest_ant_to(movable, target)
                if mover is not None:
                    steps = UNIT_STATS[mover.type][MOVEMENT]
                    mv = self._path_toward(currentState, mover.coords, target, steps)
                    if mv:
                        return mv

        # ---- Worker behavior (simple, keeps one worker productive) ----
        workers = getAntList(currentState, me, (WORKER,))
        if workers:
            w = workers[0]
            if not w.hasMoved:
                if w.carrying:
                    # Carrying food? Go to nearest drop site (our hill or tunnel)
                    dropSites = []
                    for c in myInv.constrs:
                        if c.type in (ANTHILL, TUNNEL):
                            dropSites.append(c.coords)
                    if dropSites:
                        best = None
                        bestSteps = None
                        for d in dropSites:
                            st = stepsToReach(currentState, w.coords, d)
                            if best is None or st < bestSteps:
                                best = d
                                bestSteps = st
                        path = createPathToward(currentState, w.coords, best, UNIT_STATS[WORKER][MOVEMENT])
                        if path:
                            return Move(MOVE_ANT, path, None)
                else:
                    # Not carrying? Head to the closest food
                    foods = getConstrList(currentState, None, (FOOD,))
                    if foods:
                        bestFood = None
                        bestSteps = None
                        for f in foods:
                            st = stepsToReach(currentState, w.coords, f.coords)
                            if bestFood is None or st < bestSteps:
                                bestFood = f
                                bestSteps = st
                        path = createPathToward(currentState, w.coords, bestFood.coords, UNIT_STATS[WORKER][MOVEMENT])
                        if path:
                            return Move(MOVE_ANT, path, None)

        # =================================================================
        # Part B — A*: only when we have a couple attackers (keeps turns fast)
        # =================================================================
        MAX_DEPTH = 3  # exactly 3 of OUR atomic moves (as the spec requires)

        # Early-game speed: skip A* until we have at least 2 attackers
        num_soldiers = len(getAntList(currentState, me, (SOLDIER, R_SOLDIER)))
        if num_soldiers < 2:
            # Do a simple one-move lookahead and pick the best utility
            moves = listAllLegalMoves(currentState)
            if not moves:
                return Move(END, None, None)
            best_mv = None
            best_eval = None
            for mv in moves:
                st = getNextState(currentState, mv)
                val = self.utility(st) + 1  # Part A wants eval = utility + depth (depth=1)
                if (best_eval is None) or (val > best_eval):
                    best_eval = val
                    best_mv = mv
            if best_mv is None:
                return Move(END, None, None)
            return best_mv

        # (Optional) timing for local profiling
        if PROFILE_ASTAR:
            t0 = time.perf_counter()

        # Root of our search tree
        root = {
            "move": None,
            "state": currentState,
            "depth": 0,                        # g = 0 at the start
            "f": self.h_cost(currentState),    # f = g + h = 0 + h
            "parent": None,
        }

        # Frontier = open list for A*
        frontier = [root]
        expansions = 0

        # Main A* loop (bounded by our caps to avoid long turns)
        while frontier and expansions < ASTAR_MAX_EXPANSIONS:
            # Pick node with smallest f (best guess) — linear scan keeps it simple
            best_idx = 0
            for i in range(1, len(frontier)):
                if frontier[i]["f"] < frontier[best_idx]["f"]:
                    best_idx = i
            best_node = frontier.pop(best_idx)

            # If we already reached the search horizon (depth 3), stop and keep it
            if best_node["depth"] >= MAX_DEPTH:
                frontier.append(best_node)
                break

            # Otherwise, expand children (pruned to ASTAR_CHILD_LIMIT)
            kids = self.expandNode(best_node)
            frontier.extend(kids)
            expansions += 1

            # Memory-bound: keep only the best ASTAR_FRONTIER_LIMIT nodes by f
            if len(frontier) > ASTAR_FRONTIER_LIMIT:
                while len(frontier) > ASTAR_FRONTIER_LIMIT:
                    # remove the worst f each time
                    worst_idx = 0
                    for i in range(1, len(frontier)):
                        if frontier[i]["f"] > frontier[worst_idx]["f"]:
                            worst_idx = i
                    frontier.pop(worst_idx)

        if PROFILE_ASTAR:
            t1 = time.perf_counter()
            # print("A* time (s):", t1 - t0)  # keep prints off for grading

        # If A* failed to keep anything (unlikely), fall back to 1-ply
        if not frontier:
            moves = listAllLegalMoves(currentState)
            if not moves:
                return Move(END, None, None)
            best_mv = None
            best_eval = None
            for mv in moves:
                st = getNextState(currentState, mv)
                val = self.utility(st) + 1
                if (best_eval is None) or (val > best_eval):
                    best_eval = val
                    best_mv = mv
            if best_mv is None:
                return Move(END, None, None)
            return best_mv

        # Choose the best node left on the frontier (lowest f),
        # then walk back with parent pointers to get the very first move from the root.
        goal = frontier[0]
        for n in frontier[1:]:
            if n["f"] < goal["f"]:
                goal = n

        path = []
        cur = goal
        while cur is not None and cur["parent"] is not None:
            path.append(cur["move"])
            cur = cur["parent"]

        if not path:
            # Shouldn’t happen often; safe fallback
            return Move(END, None, None)

        # path[-1] is the depth-1 move (the first move we should make)
        return path[-1]

    # =================================================================
    # Attack resolution (engine asks us who to hit if there are multiple options)
    # =================================================================
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        """
        If we’ve moved next to multiple enemies, pick one randomly to attack.
        You can get fancy here (e.g., focus low HP), but random is acceptable.
        """
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    def registerWin(self, hasWon):
        """
        Called at the end of a game. nothing to teach agent so pass.
        """
        pass


# =====================================================================
# Minimal, silent unit tests (run when executing file directly)
# =====================================================================
if __name__ == "__main__":
    print("Unit tests starting…")

    try:
        # Some ReAntics versions provide a convenience constructor for a basic state.
        test_state = GameState.getBasicState()
    except Exception:
        # If not available, skip tests silently to avoid breaking the game loader.
        test_state = None

    if test_state is not None:
        # ---- utility() contract checks ----
        util_val = AIPlayer(0).utility(test_state)
        assert isinstance(util_val, float), "utility() must return a float"
        assert 0.0 <= util_val <= 1.0, "utility() must be clamped to [0,1]"

        # ---- Part A node construction ----
        p = AIPlayer(0)
        moves = listAllLegalMoves(test_state)
        assert isinstance(moves, list), "listAllLegalMoves should return a list"
        if moves:
            nxt = getNextState(test_state, moves[0])
            node = p.makeNode(None, moves[0], nxt, 1)
            # Make sure all required fields exist
            assert "eval" in node and "state" in node and "depth" in node and "parent" in node, \
                "Part A node missing fields"

        # ---- Part B heuristic checks ----
        h0 = AIPlayer(0).h_cost(test_state)
        assert isinstance(h0, (int, float)), "h_cost must return a number"
        assert h0 >= 0, "h_cost must be non-negative"

    # If everything passes, Unit test should print nothing else.