#Alex Anderson
#09/09/2025
#HW1
#Peacemaker AI
import random
import sys
sys.path.append("..")  

from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *


class AIPlayer(Player):

    # __init__
    # Description: Creates a new Player
    #
    # Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer, self).__init__(inputPlayerId, "Peacemaker") #My unique AI agent
        self.myFood = None
        self.myTunnel = None

    ##
    # getPlacement
    #
    # Booger-like: simple, deterministic home layout; enemy food prefers deep corners.
    #
    def getPlacement(self, currentState):
        self.myFood = None
        self.myTunnel = None

        # Phase 1: 11 placements on my side (hill, tunnel, 9 grass)
        if currentState.phase == SETUP_PHASE_1:
            # Put hill/tunnel separate and away from midline a bit
            hill = (5, 0)
            tunnel = (3, 1)
            layout = [hill, tunnel]

            # 9 grass in a staggered band near row 3 (like Booger's fixed pattern vibe)
            band = [(0, 3), (2, 3), (4, 3), (6, 3), (8, 3),
                    (1, 2), (3, 2), (5, 2), (7, 2)]
            # ensure we don’t duplicate hill/tunnel
            layout.extend([c for c in band if c not in layout][:9])
            return layout

        # Phase 2: place 2 foods on enemy side — favor deep corners then sweep
        elif currentState.phase == SETUP_PHASE_2:
            picks = []
            corners = [(0, 9), (9, 9), (0, 8), (9, 8)]
            for c in corners:
                x, y = c
                if legalCoord(c) and currentState.board[x][y].constr is None:
                    picks.append(c)
                    if len(picks) == 2:
                        return picks
            # fallback sweep high rows → mid
            for y in range(9, 5, -1):
                for x in range(10):
                    c = (x, y)
                    if legalCoord(c) and currentState.board[x][y].constr is None and c not in picks:
                        picks.append(c)
                        if len(picks) == 2:
                            return picks
            return picks

        return None

    ##
    # getMove
    #
    #  - Keep 2 workers first, then build SOLDIER
    #  - Workers: pick/drop if adjacent; otherwise path to nearest goal; avoid ending next to enemies if a same-length option exists
    #  - Soldiers: move into enemy territory; step onto enemy hill/tunnel if adjacent
    #
    def getMove(self, currentState):
        me = currentState.whoseTurn
        myInv = getCurrPlayerInventory(currentState)

        # Record tunnel/food like Booger (but robust if missing)
        if self.myTunnel is None:
            myTunnels = getConstrList(currentState, me, (TUNNEL,))
            if myTunnels:
                self.myTunnel = myTunnels[0]
        if self.myFood is None:
            foods = getConstrList(currentState, 2, (FOOD,))  #
            if foods:
                # choose the one closest to my tunnel if we have one, else pick first
                if self.myTunnel is not None:
                    best = None
                    bestd = 10**9
                    for f in foods:
                        d = stepsToReach(currentState, self.myTunnel.coords, f.coords)
                        if d < bestd:
                            best, bestd = f, d
                    self.myFood = best or foods[0]
                else:
                    self.myFood = foods[0]

        # Build plan
        hill = myInv.getAnthill()
        hill_empty = (hill is not None and getAntAt(currentState, hill.coords) is None)

        workers = getAntList(currentState, me, (WORKER,))
        soldiers = getAntList(currentState, me, (SOLDIER,))
        # keep two workers, then soldiers
        if hill_empty:
            if len(workers) < 2 and myInv.foodCount >= UNIT_STATS[WORKER][COST]:
                return Move(BUILD, [hill.coords], WORKER)
            if len(soldiers) < 2 and myInv.foodCount >= UNIT_STATS[SOLDIER][COST]:
                return Move(BUILD, [hill.coords], SOLDIER)

        # Workers: immediate pick/drop if adjacent
        if workers:
            for w in workers:
                if w.hasMoved:
                    continue
                adj = listReachableAdjacent(currentState, w.coords, 1)

                # drop if carrying
                if w.carrying:
                    dropSites = []
                    if hill is not None:
                        dropSites.append(hill.coords)
                    for t in myInv.getTunnels():
                        dropSites.append(t.coords)
                    for ds in dropSites:
                        if ds in adj and getAntAt(currentState, ds) is None:
                            return Move(MOVE_ANT, [w.coords, ds], None)
                else:
                    # pick up if next to food
                    foods = getConstrList(currentState, 2, (FOOD,))
                    for f in foods:
                        if f.coords in adj and getAntAt(currentState, f.coords) is None:
                            return Move(MOVE_ANT, [w.coords, f.coords], None)

        # Workers: route toward target
        if workers:
            for w in workers:
                if w.hasMoved:
                    continue
                if w.carrying:
                    targets = [hill.coords] if hill else []
                    for t in myInv.getTunnels():
                        targets.append(t.coords)
                else:
                    foods = getConstrList(currentState, 2, (FOOD,))
                    targets = [f.coords for f in foods]

                if targets:
                    bestT = min(targets, key=lambda t: stepsToReach(currentState, w.coords, t))
                    path = createPathToward(currentState, w.coords, bestT, UNIT_STATS[WORKER][MOVEMENT])
                    if path and len(path) > 1:
                        return Move(MOVE_ANT, path, None)

        # Soldiers: capture enemy buildings if adjacent, else push forward
        if soldiers:
            enemyInv = currentState.inventories[1 - me]
            enemy_bldgs = []
            if enemyInv.getAnthill() is not None:
                enemy_bldgs.append(enemyInv.getAnthill().coords)
            for t in enemyInv.getTunnels():
                enemy_bldgs.append(t.coords)

            for s in soldiers:
                if s.hasMoved:
                    continue
                adj = listReachableAdjacent(currentState, s.coords, 1)
                for bc in enemy_bldgs:
                    if bc in adj and getAntAt(currentState, bc) is None:
                        return Move(MOVE_ANT, [s.coords, bc], None)

            for s in soldiers:
                if s.hasMoved:
                    continue
                target = (5, 8) if not enemy_bldgs else min(enemy_bldgs, key=lambda t: stepsToReach(currentState, s.coords, t))
                path = createPathToward(currentState, s.coords, target, UNIT_STATS[s.type][MOVEMENT])
                if path and len(path) > 1:
                    return Move(MOVE_ANT, path, None)

        # Nothing left to do
        return Move(END, None, None)

    ##
    # getAttack
    #
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        if not enemyLocations:
            return None

        enemies = getAntList(currentState, 1 - currentState.whoseTurn)
        by_coord = {a.coords: a for a in enemies}

        def score(c):
            a = by_coord.get(c)
            if a is None:
                return (2, 10, stepsToReach(currentState, attackingAnt.coords, c))
            pri0 = 0 if a.type == QUEEN else 1
            pri1 = a.health
            pri2 = stepsToReach(currentState, attackingAnt.coords, c)
            return (pri0, pri1, pri2)

        enemyLocations.sort(key=score)
        return enemyLocations[0]

    ##
    # registerWin
    #
    def registerWin(self, hasWon):
        pass