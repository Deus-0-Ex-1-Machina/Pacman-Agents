# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # optimal behavior will be achieved by penalizing what Pacman should not do
        # penalize stopping
        if action == 'Stop':
            return -float('inf')
        # penalize death by being in the same position as a ghost
        for ghost in newGhostStates:
            if ghost.getPosition() == newPos and ghost.scaredTimer == 0:
                return -float('inf')
        # penalize not eating food (distance to closest food)
        min_distance = -float('inf')
        food = currentGameState.getFood().asList()
        for x in food:
            food_dist = -1 * (manhattanDistance(list(newPos), x))
            if food_dist > min_distance:
                min_distance = food_dist
        return min_distance


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def minimax_decision(gameState, depth, agent_counter):
            """
            Is current depth a min (ghost) or max (pacman) decision.
            """
            if agent_counter >= gameState.getNumAgents():
                depth += 1
                agent_counter = 0 #edit depth and reset to pacman max decision
            # check termination condition
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            elif agent_counter == 0: # agent is pacman
                return max_value(gameState, depth, agent_counter)
            else: # agent is ghost
                return min_value(gameState, depth, agent_counter)

        def max_value(gameState, depth, agent_counter):
            # termination condition
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            maximum = ['', -float('inf')]
            pacman_actions = gameState.getLegalActions(agent_counter)
            # other termination condition
            if not pacman_actions:
                return self.evaluationFunction(gameState)

            for action in pacman_actions:
                actionState = gameState.generateSuccessor(agent_counter, action)
                # determine whether actionState is min or max and recursively call
                current = minimax_decision(actionState, depth, agent_counter + 1)
                if type(current) is not list:
                    newMax = current
                else:
                    newMax = current[1]
                if newMax > maximum[1]:
                    maximum = [action, newMax]
            return maximum

        def min_value(gameState, depth, agent_counter):
            # termination condition
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            minimum = ['', float('inf')]
            ghost_actions = gameState.getLegalActions(agent_counter)
            # other termination condition
            if not ghost_actions:
                return self.evaluationFunction(gameState)

            for action in ghost_actions:
                actionState = gameState.generateSuccessor(agent_counter, action)
                # determine whether actionState is min or max and recursively call
                current = minimax_decision(actionState, depth, agent_counter + 1)
                if type(current) is not list:
                    newMin = current
                else:
                    newMin = current[1]
                if newMin < minimum[1]:
                    minimum = [action, newMin]
            return minimum
        chosen_action = minimax_decision(gameState, 0, 0)
        return chosen_action[0]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def minimax_decision(gameState, depth, agent_counter, alpha, beta):
            """
            Is current depth a min (ghost) or max (pacman) decision.
            """
            if agent_counter >= gameState.getNumAgents():
                depth += 1
                agent_counter = 0 #edit depth and reset to pacman max decision
            # check termination condition
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            elif agent_counter == 0: # agent is pacman
                return max_value(gameState, depth, agent_counter, alpha, beta)
            else: # agent is ghost
                return min_value(gameState, depth, agent_counter, alpha, beta)

        def max_value(gameState, depth, agent_counter, alpha, beta):
            # termination condition
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            maximum = ['', -float('inf')]
            pacman_actions = gameState.getLegalActions(agent_counter)
            # other termination condition
            if not pacman_actions:
                return self.evaluationFunction(gameState)

            for action in pacman_actions:
                actionState = gameState.generateSuccessor(agent_counter, action)
                # determine whether actionState is min or max and recursively call
                current = minimax_decision(actionState, depth, agent_counter + 1, alpha, beta)
                if type(current) is not list:
                    newMax = current
                else:
                    newMax = current[1]
                if newMax > maximum[1]:
                    maximum = [action, newMax]
                # alpha-beta addition
                if newMax > beta:
                    return newMax
                alpha = max(alpha, newMax)
            return maximum

        def min_value(gameState, depth, agent_counter, alpha, beta):
            # termination condition
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            minimum = ['', float('inf')]
            ghost_actions = gameState.getLegalActions(agent_counter)
            # other termination condition
            if not ghost_actions:
                return self.evaluationFunction(gameState)

            for action in ghost_actions:
                actionState = gameState.generateSuccessor(agent_counter, action)
                # determine whether actionState is min or max and recursively call
                current = minimax_decision(actionState, depth, agent_counter + 1, alpha, beta)
                if type(current) is not list:
                    newMin = current
                else:
                    newMin = current[1]
                if newMin < minimum[1]:
                    minimum = [action, newMin]
                # alpha-beta addition
                if newMin < alpha:
                    return newMin
                beta = min(beta, newMin)
            return minimum
        chosen_action = minimax_decision(gameState, 0, 0, -float('inf'), float('inf'))
        return chosen_action[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def expectimax_decision(gameState, depth, agent_counter):
            if agent_counter >= gameState.getNumAgents():
                depth += 1
                agent_counter = 0  # edit depth and reset to pacman max decision
                # check termination condition
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            elif agent_counter == 0:  # agent is pacman
                return max_value(gameState, depth, agent_counter)
            else:  # agent is ghost
                return expectation(gameState, depth, agent_counter)

        def expectation(gameState, depth, agent_counter):
            # termination condition
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            expect = ['', 0]
            ghost_actions = gameState.getLegalActions(agent_counter)
            probability = 1/len(ghost_actions)
            # other termination condition
            if not ghost_actions:
                return self.evaluationFunction(gameState)

            for action in ghost_actions:
                actionState = gameState.generateSuccessor(agent_counter, action)
                # recursive call to determine expectation and increment agent counter
                current = expectimax_decision(actionState, depth, agent_counter + 1)
                if type(current) is not list:
                    newExp = current
                else:
                    newExp = current[1]
                expect[0] = action
                expect[1] += newExp * probability
            return expect

        def max_value(gameState, depth, agent_counter):
            # same as minimax
            # termination condition
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            maximum = ['', -float('inf')]
            pacman_actions = gameState.getLegalActions(agent_counter)
            # other termination condition
            if not pacman_actions:
                return self.evaluationFunction(gameState)

            for action in pacman_actions:
                actionState = gameState.generateSuccessor(agent_counter, action)
                # recursively call, increment agent counter
                current = expectimax_decision(actionState, depth, agent_counter + 1)
                if type(current) is not list:
                    newMax = current
                else:
                    newMax = current[1]
                if newMax > maximum[1]:
                    maximum = [action, newMax]
            return maximum
        chosen_action = expectimax_decision(gameState, 0, 0)
        return chosen_action[0]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    On viewing the Expectimax Agent's behavior, the flaw seemed to be not penalizing
    the distance from food. This is implemented here.
    """
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    # use negative manhattan distance as penalization function
    penalize = []
    for position in food:
        penalize.append(-1 * manhattanDistance(pos, position))
    if not food:  # no food remaining
        penalize.append(0)
    return currentGameState.getScore() + max(penalize)  # closest food


# Abbreviation
better = betterEvaluationFunction
