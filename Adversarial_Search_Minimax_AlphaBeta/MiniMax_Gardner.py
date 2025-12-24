# Alex ANderson and Carter Rhoades
# HW3_MinimaxAlphaBeta_FINAL.py
#  Minimax (depth-3) with Alpha-Beta
#
# This agent combines:
#   • A fast, board-free 0..~1 utility() (HW2 style) to stabilize choices.
#   • A stronger symmetric evaluator used by Minimax for tactical swing.
#   • Minimax search to exactly 3 plies with α-β pruning (whoseTurn decides max/min).
#   • Practical speedups: move ordering, killer-move heuristic, END-to-back ordering,
#     and Top-N% child filtering with a hard child cap to keep branching small.
#
# The flow per turn:
#   1) Quick tactical layer (fast good moves like step queen off hill, attack-in-place,
#      intercept nearby threats, maintain a productive worker).
#   2) If no quick tactic found, run Minimax + αβ (depth = 3) using getNextStateAdversarial().

import random, math, sys
sys.path.append("..")

from Player import *
from Constants import *
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

# ------------- Tunables -------------
MINIMAX_DEPTH   = 3        # spec: look 3 moves ahead (plies)
ORDER_MOVES     = True     # sort children to improve pruning
KEEP_TOP_PCT    = 0.65     # expand only top-N% after ordering (>=1 kept)
MAX_CHILDREN    = 14       # hard cap after ordering (keeps width in check)
TIE_JITTER      = 1e-6     # break ties to avoid cycles
DEBUG_ROOT      = False    # set True if you want to print root choice

class AIPlayer(Player):
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Gardener")
        self._killer = {}  # killer move per ply (for αβ ordering)

    # =========================
    # Setup (random placement)
    # =========================
    def getPlacement(self, currentState):
        used = set()
        if currentState.phase == SETUP_PHASE_1:
            out = []
            while len(out) < 11:
                x, y = random.randint(0, 9), random.randint(0, 3)
                if (x, y) not in used and currentState.board[x][y].constr is None:
                    used.add((x, y)); out.append((x, y))
            return out
        if currentState.phase == SETUP_PHASE_2:
            out = []
            while len(out) < 2:
                x, y = random.randint(0, 9), random.randint(6, 9)
                if (x, y) not in used and currentState.board[x][y].constr is None:
                    used.add((x, y)); out.append((x, y))
            return out
        return [(0, 0)]

    # =========================
    # Fast 0..1 utility (HW2-style, board-free)
    # =========================
    @staticmethod
    def _clamp01(v):
        if v < 0.0: return 0.0
        if v > 1.0: return 1.0
        return v

    def utility(self, state, me):
        """
        Return 0..~1 estimate of 'how good for me' this state is.
        - ~0.5 at start
        - near 1 when I'm close to winning
        - near 0 when I'm losing hard

        IMPORTANT: stay board-free (use inventories/ants only).
        """
        w = getWinner(state)
        if w == me:     return 0.999
        if w is not None and w != me: return 0.001

        opp  = 1 - me
        myInv = state.inventories[me]
        enInv = state.inventories[opp]

        # Food race (normalized)
        fg = FOOD_GOAL if FOOD_GOAL >= 1 else 1
        food_term = (myInv.foodCount - enInv.foodCount) / float(fg)

        # Anthills / queen presence + hill capture progress
        myHill = myInv.getAnthill()
        enHill = enInv.getAnthill()
        myQ    = myInv.getQueen()
        enQ    = enInv.getQueen()

        # enemy hill capture ⇒ convert captureHealth 3..0 into progress 0..1
        if enHill is None:
            hill_term = 1.0
        else:
            ch = enHill.captureHealth
            ch = 0.0 if ch < 0 else (3.0 if ch > 3.0 else float(ch))
            hill_term = 1.0 - (ch / 3.0)

        queen_term = 0.0
        if enQ is None: queen_term += 1.0
        if myQ is None: queen_term -= 1.0

        # Proximity pressure: nearest attacker to their queen/hill
        attackers = getAntList(state, me, (DRONE, SOLDIER, R_SOLDIER))
        def prox_to(target):
            if target is None: return 1.0
            if not attackers:  return 0.0
            dmin = min(approxDist(a.coords, target) for a in attackers)
            return self._clamp01((10.0 - float(dmin)) / 10.0)

        fight_q = prox_to(enQ.coords if enQ else None)
        fight_h = prox_to(enHill.coords if enHill else None)

        # Worker nudge
        workers = getAntList(state, me, (WORKER,))
        carrying = sum(1 for w in workers if w.carrying)
        work_term = self._clamp01(0.25 * carrying + 0.25) if workers else 0.0

        # Queen penalties: threatened or blocking hill
        q_threat_pen = 0.0
        q_block_pen  = 0.0
        if myQ is not None:
            enemy_atk = getAntList(state, opp, (DRONE, SOLDIER, R_SOLDIER))
            for a in enemy_atk:
                if approxDist(a.coords, myQ.coords) <= 1:
                    q_threat_pen += 0.25; break
            if myHill is not None and myQ.coords == myHill.coords:
                q_block_pen += 0.15

        # Weighted sum around 0.5 baseline
        score = (
            0.5
            + 0.50 * queen_term
            + 0.30 * hill_term
            + 0.20 * fight_q
            + 0.12 * fight_h
            + 0.08 * food_term
            + 0.04 * work_term
            - q_threat_pen
            - q_block_pen
        )
        return self._clamp01(score)

    # =========================
    # Stronger symmetric scorer (kept from our “winning” version)
    # =========================
    def _pressure_term(self, state, pid, target, max_bonus=6.0):
        if target is None:
            return 0.0
        attackers = getAntList(state, pid, (SOLDIER, R_SOLDIER, DRONE))
        if not attackers:
            return 0.0
        dmin = min(approxDist(a.coords, target) for a in attackers)
        return max(0.0, max_bonus - float(dmin))

    def _worker_pressure(self, state, pid):
        opp = 1 - pid
        myInv, enInv = state.inventories[pid], state.inventories[opp]
        my_workers = getAntList(state, pid, (WORKER,))
        en_workers = getAntList(state, opp, (WORKER,))
        foods      = getConstrList(state, None, (FOOD,))

        s = 0.0
        s += 1.2 * sum(1 for w in my_workers if w.carrying)
        s -= 1.6 * sum(1 for w in en_workers if w.carrying)   # heavier penalty if they're scoring
        if foods and my_workers:
            best = None
            for w in my_workers:
                if w.carrying: continue
                dmin = min(approxDist(w.coords, f.coords) for f in foods)
                if best is None or dmin < best:
                    best = dmin
            if best is not None:
                s += max(0.0, (8.0 - float(best))) * 0.15
        return s

    def _zone_control(self, state, pid):
        opp = 1 - pid
        myInv, enInv = state.inventories[pid], state.inventories[opp]
        myQ, myH = myInv.getQueen(), myInv.getAnthill()
        enQ, enH = enInv.getQueen(), enInv.getAnthill()

        my_keys = ([myQ.coords] if myQ else []) + ([myH.coords] if myH else [])
        en_keys = ([enQ.coords] if enQ else []) + ([enH.coords] if enH else [])

        my_atk = getAntList(state, pid, (SOLDIER, R_SOLDIER, DRONE))
        en_atk = getAntList(state, opp, (SOLDIER, R_SOLDIER, DRONE))

        bonus, pen = 0.0, 0.0
        for a in my_atk:
            for k in en_keys:
                if approxDist(a.coords, k) <= 3:
                    bonus += 0.8
        for a in en_atk:
            for k in my_keys:
                d = approxDist(a.coords, k)
                if d <= 1:
                    pen += 1.6
                elif d == 2:
                    pen += 0.8
        return bonus - pen

    def _side_score(self, state, pid):
        opp   = 1 - pid
        inv   = state.inventories[pid]
        oppi  = state.inventories[opp]

        fg = FOOD_GOAL if FOOD_GOAL >= 1 else 1
        food_term = 12.0 * (inv.foodCount - oppi.foodCount) / float(fg)

        enHill = oppi.getAnthill()
        if enHill is None:
            hill_term = 9.0
        else:
            ch = max(0, min(3, enHill.captureHealth))
            hill_term = (3 - ch) * 2.5

        myQ, enQ = inv.getQueen(), oppi.getQueen()
        queen_term = 0.0
        if enQ is None: queen_term += 10.0
        if myQ is None: queen_term -= 14.0

        press = 0.0
        if enQ:    press += self._pressure_term(state, pid, enQ.coords, 6.0)
        if enHill: press += self._pressure_term(state, pid, enHill.coords, 5.0)

        work  = self._worker_pressure(state, pid)

        q_block = 0.0
        myHill = inv.getAnthill()
        if myQ and myHill and myQ.coords == myHill.coords:
            q_block = 2.0

        zone = self._zone_control(state, pid)

        return food_term + hill_term + queen_term + press + work + zone - q_block

    def evalState(self, state, me):
        """
        Final evaluation used by Minimax/αβ.
        We blend:
          (A) symmetric, swing-sensitive score (kept from the "winning" version)
          (B) fast 0..1 utility() (board-free) to stabilize midgame choices

        Higher is better for 'me'.
        """
        w = getWinner(state)
        if w is not None:
            return 1e6 if w == me else -1e6

        # (A) symmetric difference
        s_me  = self._side_score(state, me)
        s_opp = self._side_score(state, 1 - me)
        sym = s_me - s_opp

        # (B) HW2-style normalized utility
        u01 = self.utility(state, me)  # 0..1

        # Blend; the weight on u01 gives steady guidance without overpowering tactical swing
        value = sym + 20.0 * u01 + TIE_JITTER * random.random()
        return value

    # =========================
    # Helper functions that support winning play (micro/tactics/paths)
    # =========================
    def _safe_neighbors(self, state, coord):
        # Return adjacent empty squares, preferring home rows and small lateral shift
        x, y = coord
        cand = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        legal = []
        for cx, cy in cand:
            if 0 <= cx < 10 and 0 <= cy < 10 and state.board[cx][cy].ant is None:
                legal.append((cx, cy))
        def key(c): return (c[1], abs(c[0]-x))
        for i in range(1, len(legal)):
            k = legal[i]; j = i-1
            while j >= 0 and key(legal[j]) > key(k):
                legal[j+1] = legal[j]; j -= 1
            legal[j+1] = k
        return legal

    def _move_in_place_to_attack(self, state, ant):
        # If an enemy is already in range, “move” to same tile to trigger an attack
        opp = 1 - state.whoseTurn
        rng = UNIT_STATS[ant.type][RANGE]
        if rng <= 0: return None
        for enemy in state.inventories[opp].ants:
            if approxDist(ant.coords, enemy.coords) <= rng:
                return Move(MOVE_ANT, [ant.coords], None)
        return None

    def _path_toward(self, state, start, dest, steps):
        # Build a path of up to 'steps' toward 'dest'
        path = createPathToward(state, start, dest, steps)
        if not path: return None
        return Move(MOVE_ANT, path, None)

    def _closest_ant_to(self, ants, target):
        # Pick the ant with smallest approximate distance to 'target'
        best, bd = None, None
        for a in ants:
            d = approxDist(a.coords, target)
            if best is None or d < bd:
                best, bd = a, d
        return best

    # =========================
    # Move ordering (αβ helper)
    # =========================
    def _orderedMoves(self, state, moves, me, maximizing, ply):
        if not moves: return moves

        # Push END to the back when there are other options
        if len(moves) > 1:
            non_end = [m for m in moves if m.moveType != END]
            ends    = [m for m in moves if m.moveType == END]
            moves   = non_end + ends

        # Killer move first (the one that caused a β/α cut at this ply previously)
        k = self._killer.get(ply)
        if k is not None:
            # enumerate gives (index, element) pairs for the 'moves' list
            for i, m in enumerate(moves):
                if m == k:
                    moves[0], moves[i] = moves[i], moves[0]
                    break

        if not ORDER_MOVES:
            return moves

        # Score children fast to sort best-first; this increases early pruning probability
        scored = []
        for mv in moves:
            st  = getNextStateAdversarial(state, mv)
            val = self.evalState(st, me)
            # tiny bias for building soldiers (tempo)
            if mv.moveType == BUILD and mv.buildType in (SOLDIER, R_SOLDIER):
                val += (0.35 if maximizing else -0.35)
            scored.append((val, mv))
        scored.sort(key=lambda t: t[0], reverse=maximizing)

        # Keep only top-N% of children after sorting by score
        # ---------------------------------------------------
        # After ranking all moves by how promising they look, retain only the
        # top fraction (e.g., 65% when KEEP_TOP_PCT = 0.65). This reduces the
        # branching factor substantially while focusing on the most relevant moves.
        # Always keep at least one move even if the list is tiny.
        if KEEP_TOP_PCT < 1.0:
            keep = int(len(scored) * KEEP_TOP_PCT)
            if keep < 1: keep = 1  # always preserve at least one
            scored = scored[:keep]

        # Hard cap — even after top-N%, limit the total children we explore
        if MAX_CHILDREN is not None and len(scored) > MAX_CHILDREN:
            scored = scored[:MAX_CHILDREN]

        return [mv for _, mv in scored]

    # =========================
    # HW3: Minimax + Alpha-Beta Pruning
    # =========================
    # - This function recursively explores the game tree up to MINIMAX_DEPTH.
    # - 'me' is the ID of our player; whoseTurn in the state determines whether we
    #   maximize (our turn) or minimize (opponent’s turn).
    # - α (alpha): the best score the MAX player can guarantee so far along this path.
    # - β (beta):  the best score the MIN player can guarantee so far along this path.
    # - If at any point α ≥ β, the remaining children can't improve the outcome → prune.
    def _minimax(self, state, depth, alpha, beta, me, ply):
        # Terminal/horizon check
        if depth == 0 or getWinner(state) is not None:
            return self.evalState(state, me), Move(END, None, None)

        # Generate legal moves for current player in this state
        moves = listAllLegalMoves(state)
        if not moves:
            return self.evalState(state, me), Move(END, None, None)

        maximizing = (state.whoseTurn == me)
        best_move = None

        # Order moves to increase chance of early αβ cuts
        moves = self._orderedMoves(state, moves, me, maximizing, ply)

        if maximizing:
            # Maximizer tries to choose the move that gives the largest value
            value = float('-inf')
            for mv in moves:
                child = getNextStateAdversarial(state, mv)
                v, _  = self._minimax(child, depth-1, alpha, beta, me, ply+1)
                if v > value:
                    value, best_move = v, mv
                # Update alpha (best known lower bound for MAX along this path)
                alpha = max(alpha, value)
                # α-β pruning: opponent would avoid branches where alpha >= beta
                if alpha >= beta:
                    self._killer[ply] = mv  # remember this cut-causing move (killer move)
                    break
            return value, (best_move or Move(END, None, None))
        else:
            # Minimizer (opponent) tries to choose the move that gives the smallest value
            value = float('+inf')
            for mv in moves:
                child = getNextStateAdversarial(state, mv)
                v, _  = self._minimax(child, depth-1, alpha, beta, me, ply+1)
                if v < value:
                    value, best_move = v, mv
                # Update beta (best known upper bound for MIN along this path)
                beta = min(beta, value)
                # α-β pruning: MAX already can do at least alpha, so if alpha >= beta,
                # MIN won't allow this branch to be reached.
                if alpha >= beta:
                    self._killer[ply] = mv
                    break
            return value, (best_move or Move(END, None, None))

    # =========================
    # Root: Tactics → Minimax
    # =========================
    def getMove(self, currentState):
        me  = currentState.whoseTurn
        opp = 1 - me
        self._killer.clear()

        # ---------- Fast tactical layer ----------
        myInv = currentState.inventories[me]
        enInv = currentState.inventories[opp]
        myHill = myInv.getAnthill()
        enHill = enInv.getAnthill()
        myQ    = myInv.getQueen()
        enQ    = enInv.getQueen()

        # Queen management: step off hill early if blocking; else attack in place
        if myQ is not None:
            if myHill is not None and myQ.coords == myHill.coords and not myQ.hasMoved:
                neigh = self._safe_neighbors(currentState, myQ.coords)
                reach = listReachableAdjacent(currentState, myQ.coords, UNIT_STATS[QUEEN][MOVEMENT])
                for c in neigh:
                    if c in reach and c[1] <= 3:  # prefer home rows
                        return Move(MOVE_ANT, [myQ.coords, c], None)
            mip = self._move_in_place_to_attack(currentState, myQ)
            if mip: return mip

        # Build SOLDIERs when hill free until we have 2
        attackers_now = getAntList(currentState, me, (SOLDIER, R_SOLDIER, DRONE))
        num_sr   = len(getAntList(currentState, me, (SOLDIER, R_SOLDIER)))
        hill_free = (myHill is not None and getAntAt(currentState, myHill.coords) is None)
        if hill_free and myInv.foodCount >= UNIT_STATS[SOLDIER][COST] and num_sr < 2:
            return Move(BUILD, [myHill.coords], SOLDIER)

        # Intercept near threats (enemy drones close to our hill/queen)
        if myQ is not None or myHill is not None:
            enemy_drones = getAntList(currentState, opp, (DRONE,))
            danger = None
            for d in enemy_drones:
                nearQ = (myQ  is not None and approxDist(d.coords, myQ.coords)  <= 2)
                nearH = (myHill is not None and approxDist(d.coords, myHill.coords) <= 2)
                if nearQ or nearH:
                    danger = d.coords; break
            if danger is not None:
                movable = [a for a in attackers_now if not a.hasMoved]
                if movable:
                    mover = self._closest_ant_to(movable, danger)
                    if mover:
                        steps = UNIT_STATS[mover.type][MOVEMENT]
                        if approxDist(mover.coords, danger) > 0:
                            mv = self._path_toward(currentState, mover.coords, danger, steps)
                            if mv: return mv
                if myQ is not None:
                    mip = self._move_in_place_to_attack(currentState, myQ)
                    if mip: return mip

        # Any attacker already has a target in range? Attack-in-place.
        for a in attackers_now:
            if a.hasMoved: continue
            mip = self._move_in_place_to_attack(currentState, a)
            if mip: return mip

        # Target priority:
        #  carry-worker > queen (if reachable) > tunnel > hill > worker > nearest attacker
        target = None
        enemy_workers = getAntList(currentState, opp, (WORKER,))
        for ew in enemy_workers:
            if ew.carrying:
                target = ew.coords; break
        if target is None and enQ is not None:
            for a in attackers_now:
                if approxDist(a.coords, enQ.coords) <= 5:
                    target = enQ.coords; break
        if target is None:
            ets = getConstrList(currentState, opp, (TUNNEL,))
            if ets: target = ets[0].coords
        if target is None and enHill is not None:
            target = enHill.coords
        if target is None and enemy_workers:
            if myHill is not None:
                closest, bd = None, None
                for ew in enemy_workers:
                    if ew.carrying: continue
                    d = approxDist(ew.coords, myHill.coords)
                    if closest is None or d < bd:
                        closest, bd = ew, d
                if closest: target = closest.coords
            else:
                for ew in enemy_workers:
                    if not ew.carrying:
                        target = ew.coords; break
        if target is None:
            enemy_atk = getAntList(currentState, opp, (SOLDIER, R_SOLDIER, DRONE))
            if enemy_atk:
                anchor = myHill.coords if myHill is not None else (myQ.coords if myQ is not None else None)
                if anchor is None:
                    target = enemy_atk[0].coords
                else:
                    best, bd = None, None
                    for e in enemy_atk:
                        d = approxDist(e.coords, anchor)
                        if best is None or d < bd:
                            best, bd = e, d
                    target = best.coords

        if target is not None and attackers_now:
            movable = [a for a in attackers_now if not a.hasMoved]
            if movable:
                mover = self._closest_ant_to(movable, target)
                if mover:
                    steps = UNIT_STATS[mover.type][MOVEMENT]
                    mv = self._path_toward(currentState, mover.coords, target, steps)
                    if mv: return mv

        # Worker simple micro (keep at least one worker productive)
        workers = getAntList(currentState, me, (WORKER,))
        if workers:
            w = workers[0]
            if not w.hasMoved:
                if w.carrying:
                    dropSites = [c.coords for c in myInv.constrs if c.type in (ANTHILL, TUNNEL)]
                    if dropSites:
                        best, bs = None, None
                        for d in dropSites:
                            stp = stepsToReach(currentState, w.coords, d)
                            if best is None or stp < bs:
                                best, bs = d, stp
                        path = createPathToward(currentState, w.coords, best, UNIT_STATS[WORKER][MOVEMENT])
                        if path: return Move(MOVE_ANT, path, None)
                else:
                    foods = getConstrList(currentState, None, (FOOD,))
                    if foods:
                        best, bs = None, None
                        for f in foods:
                            stp = stepsToReach(currentState, w.coords, f.coords)
                            if best is None or stp < bs:
                                best, bs = f, stp
                        path = createPathToward(currentState, w.coords, best.coords, UNIT_STATS[WORKER][MOVEMENT])
                        if path: return Move(MOVE_ANT, path, None)

        # ---------- If no quick tactic, run Minimax + αβ ----------
        moves = listAllLegalMoves(currentState)
        if not moves:
            return Move(END, None, None)

        value, best = self._minimax(
            currentState,
            MINIMAX_DEPTH,
            float('-inf'),
            float('+inf'),
            me,
            ply=0
        )
        if DEBUG_ROOT:
            print(f"[root] move={best.moveType if best else 'END'} val={value:.2f}")
        return best or Move(END, None, None)

    # =========================
    # Combat tie-breaker
    # =========================
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    def registerWin(self, hasWon):
        pass