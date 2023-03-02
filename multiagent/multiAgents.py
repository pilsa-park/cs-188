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
from pacman import GameState
import sys


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"

        foodList = newFood.asList()
        foodCount = len(foodList) + 1
        width = newFood.width
        height = newFood.height
        boardSize = width + height

        distFromGhosts = []
        for ghost in newGhostStates:
            ghostPos = ghost.getPosition()
            dist = manhattanDistance(newPos, ghostPos)
            distFromGhosts.append(dist)
            if dist > boardSize / 2 or dist > 2:
                distFromGhosts.append(-dist)
            else:
                distFromGhosts.append(dist)
        distanceFromGhostsAverage = sum(distFromGhosts) / len(distFromGhosts)

        distFromNearestFood = sys.maxsize
        for food in foodList:
            dist = manhattanDistance(newPos, food)
            distFromNearestFood = min(distFromNearestFood, dist)

        if foodCount <= 1:
            return successorGameState.getScore() + sys.maxsize
        return (
            successorGameState.getScore()
            + (distanceFromGhostsAverage - distFromNearestFood) / foodCount
        )


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, self.depth, 0)[0]

    def minimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return Directions.STOP, self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.getMax(gameState, depth, agentIndex)
        else:
            return self.getMin(gameState, depth, agentIndex)

    def getNextDepthAndAgent(self, currentDepth, currentAgent, numAgents):
        if currentAgent == numAgents - 1:
            return currentDepth - 1, 0
        else:
            return currentDepth, currentAgent + 1

    def getMax(self, gameState, depth, agentIndex):
        maxScore = -sys.maxsize
        maxAction = Directions.STOP

        nextDepth, nextAgent = self.getNextDepthAndAgent(
            depth, agentIndex, gameState.getNumAgents()
        )

        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            score = self.minimax(successorState, nextDepth, nextAgent)[1]

            if score > maxScore:
                maxScore = score
                maxAction = action

        return maxAction, maxScore

    def getMin(self, gameState, depth, agentIndex):
        minScore = sys.maxsize
        minAction = Directions.STOP

        nextDepth, nextAgent = self.getNextDepthAndAgent(
            depth, agentIndex, gameState.getNumAgents()
        )

        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            score = self.minimax(successorState, nextDepth, nextAgent)[1]
            if score < minScore:
                minScore = score
                minAction = action

        return minAction, minScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.minimaxWithPruning(
            gameState, self.depth, 0, -sys.maxsize, sys.maxsize
        )[0]

    def minimaxWithPruning(self, gameState, depth, agentIndex, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return Directions.STOP, self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.getMax(gameState, depth, agentIndex, alpha, beta)
        else:
            return self.getMin(gameState, depth, agentIndex, alpha, beta)

    def getNextDepthAndAgent(self, currentDepth, currentAgent, numAgents):
        if currentAgent == numAgents - 1:
            return currentDepth - 1, 0
        else:
            return currentDepth, currentAgent + 1

    def getMax(self, gameState, depth, agentIndex, alpha, beta):
        maxScore = -sys.maxsize
        maxAction = Directions.STOP

        nextDepth, nextAgent = self.getNextDepthAndAgent(
            depth, agentIndex, gameState.getNumAgents()
        )

        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            score = self.minimaxWithPruning(
                successorState, nextDepth, nextAgent, alpha, beta
            )[1]

            if score > beta:
                return action, score

            if score > maxScore:
                maxScore = score
                maxAction = action
                alpha = max(alpha, maxScore)

        return maxAction, maxScore

    def getMin(self, gameState, depth, agentIndex, alpha, beta):
        minScore = sys.maxsize
        minAction = Directions.STOP

        nextDepth, nextAgent = self.getNextDepthAndAgent(
            depth, agentIndex, gameState.getNumAgents()
        )

        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            score = self.minimaxWithPruning(
                successorState, nextDepth, nextAgent, alpha, beta
            )[1]

            if score < alpha:
                return action, score

            elif score < minScore:
                minScore = score
                minAction = action
                beta = min(beta, minScore)

        return minAction, minScore


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        return self.expectimax(gameState, self.depth, 0)[0]

    def expectimax(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return Directions.STOP, self.evaluationFunction(gameState)
        elif agentIndex == 0:
            return self.getMax(gameState, depth, agentIndex)
        else:
            return self.getExp(gameState, depth, agentIndex)

    def getNextDepthAndAgent(self, currentDepth, currentAgent, numAgents):
        if currentAgent == numAgents - 1:
            return currentDepth - 1, 0
        else:
            return currentDepth, currentAgent + 1

    def getMax(self, gameState, depth, agentIndex):
        maxScore = -sys.maxsize
        maxAction = Directions.STOP

        nextDepth, nextAgent = self.getNextDepthAndAgent(
            depth, agentIndex, gameState.getNumAgents()
        )

        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            score = self.expectimax(successorState, nextDepth, nextAgent)[1]

            if score > maxScore:
                maxScore = score
                maxAction = action

        return maxAction, maxScore

    def getExp(self, gameState, depth, agentIndex):
        nextDepth, nextAgent = self.getNextDepthAndAgent(
            depth, agentIndex, gameState.getNumAgents()
        )

        scores = []
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)
            score = self.expectimax(successorState, nextDepth, nextAgent)[1]
            scores.append(score)

        return Directions.STOP, sum(scores) / len(scores)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"

    foodList = newFood.asList()
    foodCount = len(foodList) + 1
    width = newFood.width
    height = newFood.height
    boardSize = width + height

    distFromGhosts = []
    for ghost in newGhostStates:
        ghostPos = ghost.getPosition()
        dist = manhattanDistance(newPos, ghostPos)
        distFromGhosts.append(dist)
        if dist > 3:
            distFromGhosts.append(-dist)
        elif ghost.scaredTimer > 0:
            distFromGhosts.append(-dist)
        else:
            distFromGhosts.append(dist)
    distanceFromGhostsAverage = sum(distFromGhosts) / len(distFromGhosts)

    distFromNearestFood = sys.maxsize
    for food in foodList:
        dist = manhattanDistance(newPos, food)
        distFromNearestFood = min(distFromNearestFood, dist)

    if foodCount <= 1:
        return currentGameState.getScore() + sys.maxsize
    return (
        currentGameState.getScore()
        + (distanceFromGhostsAverage - distFromNearestFood) / foodCount
    )


# Abbreviation
better = betterEvaluationFunction
