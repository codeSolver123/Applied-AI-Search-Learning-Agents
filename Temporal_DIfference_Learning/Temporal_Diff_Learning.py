#Alex anderson and Atwood Indiana
# # TD-learning ReAntics agent using value function + eligibility traces.
# Learns state utilities over time (ε-greedy + decaying α), saves weights, and improves vs Random.

import random
import pickle
import os
import numpy as np

from Player import Player
from Constants import *
from GameState import *
from AIPlayerUtils import *

# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

ALPHA = 0.1           # learning rate
GAMMA = 0.9           # discount factor
LAMBDA = 0.7          # eligibility decay
STEP_REWARD = -0.01   # tiny negative reward every step

# epsilon schedule (explore vs exploit)
EPSILON_START = 0.10
EPSILON_DECAY = 0.00005   # per game
EPSILON_MIN = 0.02        

# where we store learned utilities 
WEIGHT_FILE = "./atwoodi26_anderale26_weights.txt"


# ---------------------------------------------------------------------------
# Helper: encode a GameState into a normalized feature vector
# ---------------------------------------------------------------------------

def bucketize(vec, bins):
    """
    Turn a float vector in [0,1] into integer buckets (0..bins-1) and return as tuple.
    """
    vec = np.clip(vec, 0.0, 1.0)
    return tuple((vec * (bins - 1)).astype(int).tolist())


def encodeState(state):
    """
    Convert a GameState into a fixed-length float feature vector in [0, 1].
"""
    player_id = state.whoseTurn

    my_inv = getCurrPlayerInventory(state)
    enemy_inv = getEnemyInv(player_id, state)

    features = []

    # 1. Food counts (normalize by 11 per assignment)
    features.append(my_inv.foodCount / 11.0)
    features.append(enemy_inv.foodCount / 11.0)

    # 2. My units
    my_workers = getAntList(state, player_id, (WORKER,))
    my_soldiers = getAntList(state, player_id, (SOLDIER,))
    my_drones = getAntList(state, player_id, (DRONE,))

    features.append(len(my_workers) / 5.0)
    features.append(len(my_soldiers) / 5.0)
    features.append(len(my_drones) / 5.0)

    # 3. Enemy units
    enemy_workers = getAntList(state, 1 - player_id, (WORKER,))
    enemy_soldiers = getAntList(state, 1 - player_id, (SOLDIER,))
    enemy_drones = getAntList(state, 1 - player_id, (DRONE,))
    enemy_queen = enemy_inv.getQueen()

    features.append(len(enemy_workers) / 5.0)

    if enemy_queen:
        features.append(enemy_queen.health / UNIT_STATS[QUEEN][HEALTH])
    else:
        features.append(0.0)

    # 4. Distances (Manhattan, normalized by board max ≈ 18)
    def norm_dist(a, b):
        return (abs(a[0] - b[0]) + abs(a[1] - b[1])) / 18.0

    tunnels = getConstrList(state, player_id, (TUNNEL,))
    foods = getConstrList(state, None, (FOOD,))

    # If something is missing (shouldn't in normal games), fall back safely
    tunnel = tunnels[0] if tunnels else None
    food = foods[0] if foods else None

    if my_workers and tunnel and food:
        d_to_home = np.mean([norm_dist(w.coords, tunnel.coords) for w in my_workers])
        d_to_food = np.mean([norm_dist(w.coords, food.coords) for w in my_workers])
    else:
        d_to_home = 1.0
        d_to_food = 1.0

    if my_soldiers:
        if enemy_workers:
            d_to_worker = np.mean(
                [norm_dist(e_w.coords, my_soldiers[0].coords) for e_w in enemy_workers]
            )
        else:
            d_to_worker = 1.0

        if enemy_queen:
            d_to_queen = np.mean(
                [norm_dist(s.coords, enemy_queen.coords) for s in my_soldiers]
            )
        else:
            d_to_queen = 1.0
    else:
        d_to_worker = 1.0
        d_to_queen = 1.0

    features.append(d_to_home)
    features.append(d_to_food)
    features.append(d_to_worker)
    features.append(d_to_queen)

    # 5. Relative army strength
    total_my_units = max(len(my_workers) + len(my_soldiers) + len(my_drones), 1)
    total_enemy_units = max(len(enemy_workers) + len(enemy_soldiers) + len(enemy_drones), 1)

    features.append(len(my_soldiers) / total_my_units)       # fraction of my units that are soldiers
    features.append(len(my_workers) / total_my_units)        # fraction workers
    features.append(total_my_units / (total_my_units + total_enemy_units))  # my units ratio

    # 6. Threat / aggression measures
    threat_radius = 2
    if my_workers:
        threatened_workers = sum(
            1
            for w in my_workers
            if any(
                abs(w.coords[0] - e.coords[0]) + abs(w.coords[1] - e.coords[1]) <= threat_radius
                for e in (enemy_soldiers + enemy_drones)
            )
        )
        features.append(threatened_workers / len(my_workers))
    else:
        features.append(0.0)

    if my_soldiers and (enemy_workers or enemy_queen):
        # Soldiers “threatening” if enemy exists
        features.append(1.0)
    else:
        features.append(0.0)

    offensive = getAntList(state, player_id, (DRONE, SOLDIER, R_SOLDIER))
    if offensive and (enemy_workers or enemy_queen):
        target = enemy_workers[0] if enemy_workers else enemy_queen
        d_target = np.mean([norm_dist(a.coords, target.coords) for a in offensive])
    else:
        d_target = 1.0

    features.append(d_target)

    return np.array(features, dtype=float)


# ---------------------------------------------------------------------------
# AI Player
# ---------------------------------------------------------------------------

class AIPlayer(Player):
    """
    Temporal-Difference (TD(λ)) learning agent
    Learns a state-category utility table and uses it to pick moves.
    """

    def __init__(self, inputPlayerId):
        # *** IMPORTANT FIX: pass author name into the Player constructor ***
        super(AIPlayer, self).__init__(inputPlayerId, "Einstein_agent")

        self.playerId = inputPlayerId

        # V: category -> utility estimate
        # E: category -> eligibility trace
        self.V = {}
        self.E = {}

        # last visited category (for TD update across actions)
        self.last_category = None

        # how many games this agent has completed (for epsilon schedule)
        self.games = 0

        # try loading previous utilities, if any
        self.loadWeights()

    # convenient property for current epsilon (exploration rate)
    @property
    def epsilon(self):
        return max(EPSILON_MIN, EPSILON_START - EPSILON_DECAY * self.games)

    # ---------- game setup (unchanged logic from starter) ----------

    def getPlacement(self, currentState):
        numToPlace = 0

        if currentState.phase == SETUP_PHASE_1:   # stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move is None:
                    x = random.randint(0, 9)
                    y = random.randint(0, 3)
                    if currentState.board[x][y].constr is None and (x, y) not in moves:
                        move = (x, y)
                        currentState.board[x][y].constr = True
                moves.append(move)
            return moves

        elif currentState.phase == SETUP_PHASE_2:  # stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
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

    # ---------- TD learning utilities ----------

    def ensureCategory(self, cat):
        """Make sure category exists in V/E."""
        if cat not in self.V:
            self.V[cat] = 0.0
            self.E[cat] = 0.0

    def loadWeights(self):
        """Load learned utilities (and game count) from disk if present."""
        if not os.path.exists(WEIGHT_FILE):
            return

        try:
            with open(WEIGHT_FILE, "rb") as f:
                data = pickle.load(f)

            # support both plain dict or (dict, games) tuple
            if isinstance(data, dict):
                self.V = data
            elif isinstance(data, tuple) and len(data) == 2:
                self.V, self.games = data

            self.E = {c: 0.0 for c in self.V}
            print(f"[TD_AGENT] Loaded {len(self.V)} categories from {WEIGHT_FILE}.")

        except Exception as e:
            print(f"[TD_AGENT] Failed to load weights ({e}) — starting fresh.")

    def saveWeights(self):
        """Save learned utilities (and game count) to disk."""
        try:
            with open(WEIGHT_FILE, "wb") as f:
                # store both V and games count
                pickle.dump((self.V, self.games), f)
            print(f"[TD_AGENT] Saved {len(self.V)} categories to {WEIGHT_FILE}.")
        except Exception as e:
            # If Windows/OneDrive is being weird, just print and keep going.
            print(f"[TD_AGENT] Error saving weights: {e}")

    def getCategory(self, state):
        """Map a GameState to a discrete category using feature bucketing."""
        vec = encodeState(state)
        return bucketize(vec, 6)  # 6 buckets per feature => many possible categories

    def tempDiff(self, s_cat, next_cat, reward, gameEnd):
        """
        Core TD(λ) update.
        s_cat: current state category
        next_cat: next state category (or None if terminal)
        reward: immediate reward
        gameEnd: True if game ended after this transition
        """
        self.ensureCategory(s_cat)
        if next_cat is not None:
            self.ensureCategory(next_cat)

        V_s = self.V[s_cat]
        V_snext = 0.0 if gameEnd or next_cat is None else self.V[next_cat]

        delta = reward + GAMMA * V_snext - V_s

        # Increment eligibility for the current category
        self.E[s_cat] += 1.0

        # Update all categories using eligibility traces
        for cat in list(self.E.keys()):
            self.V[cat] += ALPHA * delta * self.E[cat]
            self.E[cat] *= GAMMA * LAMBDA

    # ---------- move selection ----------

    def evaluateMove(self, state, move):
        """
        Apply move to get next state, then return (predicted_value, gameEnd).
        """
        nextState = getNextState(state, move)

        enemyInv = getEnemyInv(self.playerId, nextState)
        enemy_queen = enemyInv.getQueen()

        # If enemy Queen is gone, we assume we just won.
        if enemy_queen is None:
            return 1.0, True

        cat = self.getCategory(nextState)
        self.ensureCategory(cat)
        return self.V[cat], False

    def getMove(self, currentState):
        """
        Choose an action using epsilon-greedy selection over predicted state utilities.
        """
        legalMoves = listAllLegalMoves(currentState)
        if not legalMoves:
            return Move(END, None, None)  # just in case

        # Exploration
        if random.random() < self.epsilon:
            move = random.choice(legalMoves)
            nextState = getNextState(currentState, move)
            nextCat = self.getCategory(nextState)

            if self.last_category is not None:
                self.tempDiff(self.last_category, nextCat, STEP_REWARD, False)

            self.last_category = nextCat
            return move

        # Exploitation: choose move leading to highest predicted utility
        bestVal = -999999.0
        bestMoves = []

        for m in legalMoves:
            val, gameEnd = self.evaluateMove(currentState, m)
            if val > bestVal:
                bestVal = val
                bestMoves = [(m, gameEnd)]
            elif val == bestVal:
                bestMoves.append((m, gameEnd))

        chosen, gameEnd = random.choice(bestMoves)
        nextCat = None
        if not gameEnd:
            nextState = getNextState(currentState, chosen)
            nextCat = self.getCategory(nextState)

        # Intermediate TD update for this step
        if self.last_category is not None:
            reward = 1.0 if gameEnd else STEP_REWARD
            self.tempDiff(self.last_category, nextCat, reward, gameEnd)

        self.last_category = nextCat
        return chosen

    # ---------- combat & end-of-game ----------

    def getAttack(self, currentState, attackingAnt, enemyLocations):
        """
        Choose which enemy to attack; here we just pick a random target.
        """
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    def registerWin(self, hasWon):
        """
        Called by the game engine when the game finishes.
        Apply a final TD update and save weights.
        """
        if self.last_category is not None:
            finalReward = 1.0 if hasWon else -1.0
            self.tempDiff(self.last_category, None, finalReward, True)

        # reset eligibilities
        for c in list(self.E.keys()):
            self.E[c] = 0.0

        self.last_category = None

        # update game counter for epsilon schedule
        self.games += 1

        # occasional status printout
        if self.games % 20 == 0:
            print(
                f"[TD_AGENT] games={self.games}, "
                f"categories={len(self.V)}, epsilon={self.epsilon:.3f}"
            )

        # save learned utilities
        self.saveWeights()