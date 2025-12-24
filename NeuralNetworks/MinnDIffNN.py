# HW5B — Alex Anderson, Vu Hung-Nghi 
# Title: MinDiff+ (policy + 1-ply fallback) with NN hook + logging + SFG tilt

import random, sys, math, csv, os
sys.path.append("..")
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

# ---------------- NN switch & trained weights ----------------
# When True, the agent uses the trained neural net to evaluate states.
# When False, it uses the original hand-written utility() function.
USE_NN = True

# Turn this on only when you are collecting new training data.
# For HW5B final submission, we leave it False to avoid slowing down games.
LOGGING_ENABLED = False

def _clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

# ---------------------------------------------------------------------
# Trained neural network weights for HW5B
#
# Architecture:
#   - 5 input features from map_state_to_features(...)
#   - 8 hidden units with sigmoid activation
#   - 1 sigmoid output (overall utility in [0,1])
#
# These weights were learned offline by train_nn.py from train_log.csv,
# which logged (features, heuristic_utility) pairs during actual games.
# ---------------------------------------------------------------------

# W1: shape [5 inputs x 8 hidden]
W1 = [
    [0.042132, 0.494247, 0.649786, -0.366419, 0.167174, 0.601196, -0.575738, 0.25326],
    [0.847176, -0.186111, 0.727897, -0.355828, 0.381923, 0.961814, -1.034355, -1.250126],
    [-1.02633, 0.700299, 0.606925, 0.405036, 1.123966, 0.630778, -0.170783, 0.224983],
    [-0.744875, 0.261058, -0.885589, 0.992746, -0.099141, -0.341778, -0.29426, 0.681304],
    [-0.055535, 0.112219, -1.216045, 0.40383, 0.03877, 0.030001, 1.12293, 0.580809]
]

# b1: hidden-layer biases (length 8)
b1 = [-0.008478, -0.024093, -0.421866, -0.023024, -0.203069, -0.433824, 0.329467, -0.005985]

# W2: shape [8 hidden x 1 output]
W2 = [
    [-0.054614],
    [0.245112],
    [1.601147],
    [-0.957013],
    [0.920476],
    [1.27674],
    [-1.479522],
    [-1.069622]
]

# b2: output bias (single scalar stored as a 1-element list)
b2 = [0.023068]

# ---------------- knobs (tiny, safe nudges) ----------------
OPENING_TURNS      = 7        # a touch more early tempo
CHASE_QUEEN_RADIUS = 4
ATTACKER_CAP_SOFT  = 3        # allow situational 3rd attacker vs food-racers
MIDGAME_TURN       = 8        # only after the opening

# ---------------- logging target -------------------
LOG_FILE = "train_log.csv"

class AIPlayer(Player):
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Slick Rick")
        self.turn = 0

    # ---------------- setup (random per spec) ----------------
    def getPlacement(self, currentState):
        if currentState.phase == SETUP_PHASE_1:
            numToPlace, moves = 11, []
            for _ in range(numToPlace):
                move = None
                while move is None:
                    x, y = random.randint(0, 9), random.randint(0, 3)
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr = True
                moves.append(move)
            return moves

        elif currentState.phase == SETUP_PHASE_2:
            numToPlace, moves = 2, []
            for _ in range(numToPlace):
                move = None
                while move is None:
                    x, y = random.randint(0, 9), random.randint(6, 9)
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr = True
                moves.append(move)
            return moves

        return [(0, 0)]

    # ---------------- heuristic utility (0..1) ----------------
    @staticmethod
    def _clamp01(v):
        return _clamp01(v)

    def utility(self, state):
        w = getWinner(state)
        if w == 1: return 0.999
        if w == 0: return 0.001

        me, opp = state.whoseTurn, 1 - state.whoseTurn
        myInv, enInv = state.inventories[me], state.inventories[opp]

        fg = max(1, FOOD_GOAL)
        food_term = (myInv.foodCount - enInv.foodCount) / float(fg)

        myHill, enHill = myInv.getAnthill(), enInv.getAnthill()
        myQ,    enQ    = myInv.getQueen(),    enInv.getQueen()

        # hill progress 0..1
        if enHill is None:
            hill_term = 1.0
        else:
            hill_term = 1.0 - min(1.0, max(0.0, enHill.captureHealth / 3.0))

        # queen term {-1,0,+1}
        queen_term = (1.0 if enQ is None else 0.0) + (-1.0 if myQ is None else 0.0)

        # proximity of attackers to enemy queen/hill 0..1
        attackers = getAntList(state, me, (DRONE, SOLDIER, R_SOLDIER))
        def prox_to(target_coords):
            if target_coords is None: return 1.0
            if not attackers:         return 0.0
            dmin = min(approxDist(a.coords, target_coords) for a in attackers)
            return self._clamp01((10.0 - float(dmin)) / 10.0)

        fight_q_term = prox_to(enQ.coords if enQ else None)
        fight_h_term = prox_to(enHill.coords if enHill else None)

        # tiny worker encouragement
        workers = getAntList(state, me, (WORKER,))
        work_term = 0.0
        if workers:
            carrying = sum(1 for w in workers if w.carrying)
            work_term = self._clamp01(0.25 * carrying + 0.25)

        # NEW: worker harassment pressure (better vs Simple Food Gatherer)
        en_workers = getAntList(state, opp, (WORKER,))
        if attackers and en_workers:
            d_w = min(min(approxDist(a.coords, w.coords) for a in attackers) for w in en_workers)
            work_press = self._clamp01((10.0 - float(d_w)) / 10.0)
        else:
            work_press = 0.0

        # queen safety penalties
        q_threat_pen, q_block_pen = 0.0, 0.0
        if myQ is not None:
            enemy_attackers = getAntList(state, opp, (DRONE, SOLDIER, R_SOLDIER))
            if any(approxDist(a.coords, myQ.coords) <= 1 for a in enemy_attackers):
                q_threat_pen += 0.25
            if myHill is not None and myQ.coords == myHill.coords:
                q_block_pen += 0.15

        # Slight rebalance toward hill pressure & worker harassment
        score = (0.5
                 + 0.55 * queen_term
                 + 0.40 * hill_term           # was 0.35
                 + 0.25 * fight_q_term
                 + 0.18 * fight_h_term        # was 0.15
                 + 0.10 * food_term
                 + 0.06 * work_term
                 + 0.10 * work_press          # NEW
                 - q_threat_pen
                 - q_block_pen)
        return self._clamp01(score)

    # ---------------- features for NN (same signals) ----------------
    def map_state_to_features(self, state):
        """
        Map a GameState into 5 normalized features that the NN can use:
          f0: food lead (scaled into ~[0,1])
          f1: proximity of our attackers to enemy queen
          f2: proximity of our attackers to enemy hill
          f3: queen blocking own hill (0 or 1)
          f4: queen under threat (0 or 1)
        """
        me, opp = state.whoseTurn, 1 - state.whoseTurn
        myInv, enInv = state.inventories[me], state.inventories[opp]
        myHill, enHill = myInv.getAnthill(), enInv.getAnthill()
        myQ, enQ = myInv.getQueen(), enInv.getQueen()

        attackers = getAntList(state, me, (DRONE, SOLDIER, R_SOLDIER))

        fg = max(1, FOOD_GOAL)
        food_raw = (myInv.foodCount - enInv.foodCount) / float(fg)
        food_lead01 = _clamp01(0.5 + 0.5 * food_raw)

        def prox(target):
            if target is None or not attackers: return 0.0
            d = min(approxDist(a.coords, target.coords) for a in attackers)
            return _clamp01((10.0 - float(d)) / 10.0)

        proxQ = prox(enQ)
        proxH = prox(enHill)

        q_block  = 1.0 if (myQ and myHill and myQ.coords == myHill.coords) else 0.0
        q_threat = 0.0
        if myQ:
            enemy_atk = getAntList(state, opp, (DRONE, SOLDIER, R_SOLDIER))
            if any(approxDist(a.coords, myQ.coords) <= 1 for a in enemy_atk):
                q_threat = 1.0

        # Kept F=5 so the trainer matches; NN learns from these signals.
        return [food_lead01, proxQ, proxH, q_block, q_threat]

    # ---------------- NN forward + eval wrapper ----------------
    def _sigm(self, x):
        """Standard sigmoid activation used by the network."""
        # slightly safer sigmoid in case x gets large
        if x > 60:  return 1.0
        if x < -60: return 0.0
        return 1.0 / (1.0 + math.exp(-x))

    def nn_forward(self, x):
        """
        Feed-forward through the trained neural network.
        x: list of 5 input features from map_state_to_features(...)
        returns: scalar utility in [0,1]
        """
        # If for some reason weights aren't set, just return neutral value.
        if not W1 or not W2 or not b1 or b2 is None:
            return 0.5

        # Hidden layer: h_j = sigmoid(b1_j + sum_i W1[i][j] * x_i)
        H = []
        H_len = len(b1)
        for j in range(H_len):
            s = b1[j]
            for i in range(len(x)):
                s += W1[i][j] * x[i]
            H.append(self._sigm(s))

        # Output layer: y = sigmoid(b2 + sum_j W2[j][0] * H_j)
        s = b2[0]
        for j in range(H_len):
            s += W2[j][0] * H[j]
        return _clamp01(self._sigm(s))

    def eval_state(self, state):
        """
        Central evaluation hook:
          - If USE_NN is True, evaluate with the trained neural net.
          - Otherwise, fall back to the original utility() heuristic.
        """
        if USE_NN:
            feats = self.map_state_to_features(state)
            return self.nn_forward(feats)
        else:
            return self.utility(state)

    # ---------------- one-ply scoring ----------------
    def _queen_adjacent_after(self, state):
        me, opp = state.whoseTurn, 1 - state.whoseTurn
        enQ = state.inventories[opp].getQueen()
        if enQ is None: return False
        myAtk = getAntList(state, me, (SOLDIER, DRONE, R_SOLDIER))
        return any(approxDist(a.coords, enQ.coords) <= 1 for a in myAtk)

    def makeNode(self, parent, move, childState, depth=1):
        base = self.eval_state(childState)
        kill_bonus = 0.0

        if parent is not None:
            opp_prev = 1 - parent["state"].whoseTurn
            opp_now  = 1 - childState.whoseTurn

            enQ_prev = parent["state"].inventories[opp_prev].getQueen()
            enQ_now  = childState.inventories[opp_now].getQueen()
            if enQ_prev is not None and enQ_now is None:
                kill_bonus += 0.35

            en_prev_cnt = len(parent["state"].inventories[opp_prev].ants)
            en_now_cnt  = len(childState.inventories[opp_now].ants)
            if en_now_cnt < en_prev_cnt:
                kill_bonus += 0.10

        press_bonus = 0.08 if self._queen_adjacent_after(childState) else 0.0

        return {"move": move, "state": childState, "depth": depth,
                "eval": base + kill_bonus + press_bonus, "parent": parent}

    def _bestNode(self, nodes):
        if not nodes: return None
        best_val = max(n["eval"] for n in nodes)
        top = [n for n in nodes if abs(n["eval"] - best_val) < 1e-9]
        return random.choice(top)

    # ---------------- safe movement helpers (no zero-step) ----------------
    def _safe_neighbors(self, state, coord):
        x, y = coord
        cand = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        legal = [c for c in cand
                 if 0 <= c[0] < 10 and 0 <= c[1] < 10
                 and state.board[c[0]][c[1]].ant is None]
        return sorted(legal, key=lambda c: (c[1], abs(c[0]-x)))

    def _path_toward(self, state, start, dest, steps):
        path = createPathToward(state, start, dest, steps)
        if path:
            return Move(MOVE_ANT, path, None)
        # Greedy single-step fallback (still non-zero)
        x, y = start
        neigh = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        legal = [c for c in neigh
                 if 0 <= c[0] < 10 and 0 <= c[1] < 10
                 and state.board[c[0]][c[1]].ant is None]
        if not legal: return None
        best = min(legal, key=lambda c: approxDist(c, dest))
        return Move(MOVE_ANT, [start, best], None)

    # ---------------- main policy ----------------
    def getMove(self, currentState):
        # ---- logging for training data (features + heuristic label) ----
        if LOGGING_ENABLED:
            try:
                feats  = self.map_state_to_features(currentState)
                target = self.utility(currentState)     # teacher label for offline NN training
                row = feats + [target]
                new_file = not os.path.exists(LOG_FILE)
                with open(LOG_FILE, "a", newline="") as f:
                    w = csv.writer(f)
                    if new_file:
                        w.writerow([f"f{i}" for i in range(len(feats))] + ["y"])
                    w.writerow(row)
            except Exception:
                # If logging fails for some reason, just ignore and keep playing.
                pass
        # ---------------------------------------------------------------

        self.turn += 1

        me, opp = currentState.whoseTurn, 1 - currentState.whoseTurn
        myInv, enInv = currentState.inventories[me], currentState.inventories[opp]
        myHill, enHill = myInv.getAnthill(), enInv.getAnthill()
        myQ, enQ = myInv.getQueen(), enInv.getQueen()

        # (0) economy floor — 1 worker if we have none
        if myHill is not None:
            workers_now = getAntList(currentState, me, (WORKER,))
            hill_free = (getAntAt(currentState, myHill.coords) is None)
            if hill_free and len(workers_now) == 0 and myInv.foodCount >= UNIT_STATS[WORKER][COST]:
                return Move(BUILD, [myHill.coords], WORKER)

        # (1) queen — step off hill if blocking (no zero-step)
        if myQ is not None and myHill is not None and myQ.coords == myHill.coords and not myQ.hasMoved:
            for c in self._safe_neighbors(currentState, myQ.coords):
                if c[1] <= 3 and c in listReachableAdjacent(currentState, myQ.coords, UNIT_STATS[QUEEN][MOVEMENT]):
                    return Move(MOVE_ANT, [myQ.coords, c], None)

        attackers = getAntList(currentState, me, (SOLDIER, DRONE, R_SOLDIER))
        hill_free = (myHill is not None and getAntAt(currentState, myHill.coords) is None)

        # (1A) opening — get to 1–2 attackers quickly
        if (self.turn <= OPENING_TURNS
            and hill_free
            and myInv.foodCount >= UNIT_STATS[SOLDIER][COST]
            and len(attackers) < 2):
            return Move(BUILD, [myHill.coords], SOLDIER)

        # (2) maintain attackers (soft 3rd midgame if enemy has workers)
        en_workers_exist = bool(getAntList(currentState, opp, (WORKER,)))
        cap = 2
        if self.turn >= MIDGAME_TURN and en_workers_exist:
            cap = ATTACKER_CAP_SOFT
        if hill_free and myInv.foodCount >= UNIT_STATS[SOLDIER][COST] and len(attackers) < cap:
            return Move(BUILD, [myHill.coords], SOLDIER)

        # (3) close-range queen chase
        if enQ is not None and attackers:
            close_attackers = [a for a in attackers
                               if not a.hasMoved and approxDist(a.coords, enQ.coords) <= CHASE_QUEEN_RADIUS]
            if close_attackers:
                mover = min(close_attackers, key=lambda a: approxDist(a.coords, enQ.coords))
                steps = UNIT_STATS[mover.type][MOVEMENT]
                mv = self._path_toward(currentState, mover.coords, enQ.coords, steps)
                if mv: return mv

        # (3.5) worker harassment — deny their food race
        en_workers = getAntList(currentState, opp, (WORKER,))
        if en_workers and attackers:
            foods = getConstrList(currentState, None, (FOOD,))

            def near_food_steps(w):
                if not foods: return 99
                return min(stepsToReach(currentState, w.coords, f.coords) for f in foods)

            def worker_score(w):
                carry_bonus = -2 if w.carrying else 0
                return carry_bonus + near_food_steps(w)

            prey = min(en_workers, key=worker_score)
            if prey.carrying or near_food_steps(prey) <= 2:
                movable = [a for a in attackers if not a.hasMoved]
                if movable:
                    mover = min(movable, key=lambda a: approxDist(a.coords, prey.coords))
                    steps = UNIT_STATS[mover.type][MOVEMENT]
                    mv = self._path_toward(currentState, mover.coords, prey.coords, steps)
                    if mv: return mv

        # (4) pressure: queen > hill > worker > any enemy
        target = None
        if enQ is not None:
            target = enQ.coords
        elif enHill is not None:
            target = enHill.coords
        else:
            if en_workers and attackers:
                target = min(en_workers,
                             key=lambda w: min(approxDist(a.coords, w.coords) for a in attackers)).coords
            elif enInv.ants and attackers:
                target = min(enInv.ants,
                             key=lambda e: min(approxDist(a.coords, e.coords) for a in attackers)).coords

        if target is not None and attackers:
            movable = [a for a in attackers if not a.hasMoved]
            if movable:
                mover = min(movable, key=lambda a: approxDist(a.coords, target))
                steps = UNIT_STATS[mover.type][MOVEMENT]
                mv = self._path_toward(currentState, mover.coords, target, steps)
                if mv: return mv

        # (5) worker: bring food home, else to nearest food
        workers = getAntList(currentState, me, (WORKER,))
        if workers:
            w = workers[0]
            if not w.hasMoved:
                if w.carrying:
                    dropSites = [c.coords for c in myInv.constrs if c.type in (ANTHILL, TUNNEL)]
                    if dropSites:
                        best = min(dropSites, key=lambda d: stepsToReach(currentState, w.coords, d))
                        path = createPathToward(currentState, w.coords, best, UNIT_STATS[WORKER][MOVEMENT])
                        if path:
                            return Move(MOVE_ANT, path, None)
                else:
                    foods = getConstrList(currentState, None, (FOOD,))
                    if foods:
                        bestFood = min(foods, key=lambda f: stepsToReach(currentState, w.coords, f.coords))
                        path = createPathToward(currentState, w.coords, bestFood.coords, UNIT_STATS[WORKER][MOVEMENT])
                        if path:
                            return Move(MOVE_ANT, path, None)

        # (6) fallback: 1-ply lookahead scored by eval_state() + tiny bonuses
        moves = listAllLegalMoves(currentState)
        nodes = []
        for mv in moves:
            nxt = getNextState(currentState, mv)
            nodes.append(self.makeNode(parent={"state": currentState}, move=mv, childState=nxt, depth=1))
        best = self._bestNode(nodes)
        return best["move"] if best else Move(END, None, None)

    # ---------------- attacks: prefer queen if in range ----------------
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        opp = 1 - currentState.whoseTurn
        enQ = currentState.inventories[opp].getQueen()
        if enQ and enQ.coords in enemyLocations:
            return enQ.coords
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    def registerWin(self, hasWon): 
        pass