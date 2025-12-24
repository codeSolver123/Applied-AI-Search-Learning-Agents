# Alex Anderson
# Reiss Oliveros
import random
import sys
sys.path.append("..")  # so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *

##
# 
# 
#   * moves queen off hill early (don’t block builds / avoid free hits)
#   * builds 1–2 SOLDIERs ASAP
#   * drives attackers at enemy queen/hill
#   * “moves in place” to trigger attacks when already in range
#
class AIPlayer(Player):
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Random")

    # ------------------------------------------------------------------
    #  Random layout (unchanged per spec)
    # ------------------------------------------------------------------
    def getPlacement(self, currentState):
        if currentState.phase == SETUP_PHASE_1:    # my side
            numToPlace = 11
            moves = []
            for _ in range(numToPlace):
                move = None
                while move is None:
                    x = random.randint(0, 9)
                    y = random.randint(0, 3)
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr = True
                moves.append(move)
            return moves

        elif currentState.phase == SETUP_PHASE_2:  # enemy side
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

        else:
            return [(0, 0)]

    # ------------------------------------------------------------------
    #  Utility (no board access) + helpers for Part A grading
    # ------------------------------------------------------------------
    @staticmethod
    def _clamp01(v):
        return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

    def utility(self, state):
        """
        Fast heuristic in [0,1], NO board array access.
        Strongly rewards: enemy queen down, enemy hill capture progress, pressuring their queen/hill,
        and our queen safety. Food matters but less.
        """
        w = getWinner(state)
        if w == 1:
            return 0.999
        if w == 0:
            return 0.001

        me  = state.whoseTurn
        opp = 1 - me
        myInv = state.inventories[me]
        enInv = state.inventories[opp]

        # food lead [-1..1]
        fg = max(1, FOOD_GOAL)
        food_term = (myInv.foodCount - enInv.foodCount) / float(fg)

        myHill = myInv.getAnthill()
        enHill = enInv.getAnthill()
        myQ    = myInv.getQueen()
        enQ    = enInv.getQueen()

        # enemy hill capture progress 0..1
        if enHill is None:
            hill_term = 1.0
        else:
            hill_term = 1.0 - min(1.0, max(0.0, enHill.captureHealth / 3.0))

        # queen status {-1,0,+1}
        queen_term = (1.0 if enQ is None else 0.0) + (-1.0 if myQ is None else 0.0)

        # proximity of my attackers to enemy queen/hill 0..1
        attackers = getAntList(state, me, (DRONE, SOLDIER, R_SOLDIER))
        def prox_to(target_coords):
            if target_coords is None:
                return 1.0
            if not attackers:
                return 0.0
            dmin = min(approxDist(a.coords, target_coords) for a in attackers)
            return self._clamp01((10.0 - float(dmin)) / 10.0)

        fight_q_term = prox_to(enQ.coords if enQ else None)
        fight_h_term = prox_to(enHill.coords if enHill else None)

        # small worker productivity nudge
        workers = getAntList(state, me, (WORKER,))
        work_term = 0.0
        if workers:
            carrying = sum(1 for wkr in workers if wkr.carrying)
            work_term = self._clamp01(0.25 * carrying + 0.25)

        # queen safety penalties
        q_threat_pen = 0.0
        q_block_pen  = 0.0
        if myQ is not None:
            enemy_attackers = getAntList(state, opp, (DRONE, SOLDIER, R_SOLDIER))
            if any(approxDist(a.coords, myQ.coords) <= 1 for a in enemy_attackers):
                q_threat_pen += 0.25
            if myHill is not None and myQ.coords == myHill.coords:
                q_block_pen += 0.15

        score = (
            0.5
            + 0.55 * queen_term
            + 0.35 * hill_term
            + 0.25 * fight_q_term
            + 0.15 * fight_h_term
            + 0.10 * food_term
            + 0.05 * work_term
            - q_threat_pen
            - q_block_pen
        )
        return self._clamp01(score)

    def makeNode(self, parent, move, childState, depth=1):
        return {
            "move": move,
            "state": childState,
            "depth": depth,
            "eval": self.utility(childState) + depth,
            "parent": parent,
        }

    def _bestNode(self, nodes):
        if not nodes:
            return None
        best_val = max(n["eval"] for n in nodes)
        top = [n for n in nodes if abs(n["eval"] - best_val) < 1e-9]
        return random.choice(top)

   
    def _safe_neighbors(self, state, coord):
        """Return neighboring coords (N/E/S/W, on-board, empty) ordered by pref away from center line. 
           # adjacent, on-board, empty; prefer staying within home rows to keep queen safe"""
        x, y = coord
        cand = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
        legal = [c for c in cand if 0 <= c[0] < 10 and 0 <= c[1] < 10 and state.board[c[0]][c[1]].ant is None]
        # prefer staying in home rows (y small) to avoid queen crossing center accidentally
        return sorted(legal, key=lambda c: (c[1], abs(c[0]-x)))

    def _move_in_place_to_attack(self, state, ant):
        """If ant already has a valid target in range, return a 'move in place' to trigger attack."""
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
        path = createPathToward(state, start, dest, steps)
        if not path:
            return None
        return Move(MOVE_ANT, path, None)

    # ------------------------------------------------------------------
    #  Move selection
    # ------------------------------------------------------------------
    def getMove(self, currentState):
        me  = currentState.whoseTurn
        opp = 1 - me
        myInv = currentState.inventories[me]
        enInv = currentState.inventories[opp]
        myHill = myInv.getAnthill()
        enHill = enInv.getAnthill()
        myQ    = myInv.getQueen()
        enQ    = enInv.getQueen()

        if myQ is not None:
            # if queen sits on hill, step her off to the nearest open home tile
            if myHill is not None and myQ.coords == myHill.coords and not myQ.hasMoved:
                for c in self._safe_neighbors(currentState, myQ.coords):
                    # ensure we don't try to cross the center with queen
                    if c[1] <= 3:  # stay in home half (rows 0..3)
                        if c in listReachableAdjacent(currentState, myQ.coords, UNIT_STATS[QUEEN][MOVEMENT]):
                            return Move(MOVE_ANT, [myQ.coords, c], None)
                # fallback: move in place so she can attack if someone walked up
                mip = self._move_in_place_to_attack(currentState, myQ)
                if mip:
                    return mip

            # if enemy is in range already, attack by moving in place
            mip = self._move_in_place_to_attack(currentState, myQ)
            if mip:
                return mip

        # BUILD 1–2 SOLDIERS ASAP 
        attackers = getAntList(currentState, me, (SOLDIER, DRONE, R_SOLDIER))
        hill_free = (myHill is not None and getAntAt(currentState, myHill.coords) is None)
        if hill_free and myInv.foodCount >= UNIT_STATS[SOLDIER][COST] and len(attackers) < 2:
            return Move(BUILD, [myHill.coords], SOLDIER)

        # (a) if any attacker can already hit something important, do a move-in-place to trigger attack
        for a in attackers:
            if a.hasMoved:
                continue
            mip = self._move_in_place_to_attack(currentState, a)
            if mip:
                return mip

        # otherwise, drive first available attacker toward enemy queen, else enemy hill
        target = None
        if enQ is not None:
            target = enQ.coords
        elif enHill is not None:
            target = enHill.coords
        if target is not None:
            # pick the attacker closest to target and move it
            movable = [a for a in attackers if not a.hasMoved]
            if movable:
                mover = min(movable, key=lambda a: approxDist(a.coords, target))
                steps = UNIT_STATS[mover.type][MOVEMENT]
                mv = self._path_toward(currentState, mover.coords, target, steps)
                if mv:
                    return mv

        workers = getAntList(currentState, me, (WORKER,))
        if workers:
            w = workers[0]
            if not w.hasMoved:
                # if carrying, go home (anthill or tunnel)
                if w.carrying:
                    dropSites = [c.coords for c in myInv.constrs if c.type in (ANTHILL, TUNNEL)]
                    best = min(dropSites, key=lambda d: stepsToReach(currentState, w.coords, d))
                    path = createPathToward(currentState, w.coords, best, UNIT_STATS[WORKER][MOVEMENT])
                    if path:
                        return Move(MOVE_ANT, path, None)
                else:
                    # go to closest food
                    foods = getConstrList(currentState, None, (FOOD,))
                    if foods:
                        bestFood = min(foods, key=lambda f: stepsToReach(currentState, w.coords, f.coords))
                        path = createPathToward(currentState, w.coords, bestFood.coords, UNIT_STATS[WORKER][MOVEMENT])
                        if path:
                            return Move(MOVE_ANT, path, None)

        moves = listAllLegalMoves(currentState)
        nodes = []
        for mv in moves:
            nxt = getNextState(currentState, mv)
            nodes.append(self.makeNode(parent=None, move=mv, childState=nxt, depth=1))
        best = self._bestNode(nodes)
        return best["move"] if best else Move(END, None, None)

   
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    def registerWin(self, hasWon):
        pass