# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import mypy, sys

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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

  def recusivecalls(self):
      numrecursions = 0

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
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newFood = successorGameState.getFood()

    if 'foodarray' not in locals():
        foodarray = mypy.foodArray(newFood)

    pacmanx,pacmany = newPos

    fooddistance = []
    for pellet in foodarray:
        distancetofood = util.manhattanDistance(pellet,newPos)
        if distancetofood < 6:
            distancetofood = mypy.mazeDistance(pellet,newPos,successorGameState)
        fooddistance.append(distancetofood)

    ghostDistances = []

    for ghostposition in successorGameState.getGhostPositions():
        distancetoghost = util.manhattanDistance(newPos,ghostposition)
        if distancetoghost < 6:
            distancetoghost = mypy.mazeDistance(ghostposition,newPos,successorGameState)
        ghostDistances.append(distancetoghost)

    ghostDistance = min(ghostDistances)

    mindistance = 0
    if len(fooddistance) != 0:
        mindistance = min(fooddistance)

    if ghostDistance > 4 or min(newScaredTimes) > 4:
        ghostDistance = 0.5*ghostDistance

    elif min(newScaredTimes) > 7:
        ghostDistance = 0

    elif min(newScaredTimes) > 12:
        ghostDistance = -2*ghostDistance


    evaluation = successorGameState.getScore() + 2*ghostDistance - mindistance

    return evaluation

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """

    value,bestaction = self.maxvalue(gameState,self.depth,Directions.STOP)

    return bestaction

  def maxvalue(self,gameState,depth,action):
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState),action
      depth -= 1
      value = int(-1*sys.maxsize)
      bestaction = Directions.STOP

      actions = gameState.getLegalActions(0)
      if actions.count('Stop') != 0:
              actions.remove('Stop')
      for action in actions:
          childstate = gameState.generateSuccessor(0,action)
          minivalue,direction = self.minvalue(childstate,depth,action)
          if minivalue > value:
              value = minivalue
              bestaction = action

      return value,bestaction

  def minvalue(self,gameState,depth,action):
      if depth == 0 or gameState.isLose() or gameState.isWin():
          return self.evaluationFunction(gameState),action

      value = int(sys.maxsize)
      bestaction = Directions.STOP

      for agent in range(1,gameState.getNumAgents()):
          actions = gameState.getLegalActions(agent)
          if actions.count('Stop') != 0:
              actions.remove('Stop')
          for action in actions:
              childstate = gameState.generateSuccessor(agent,action)
              maxivalue,direction = self.maxvalue(childstate,depth,action)
              if maxivalue < value:
                  value = maxivalue
                  bestaction = action

      return value,bestaction

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    value,bestaction,alpha,beta = self.maxvalue(gameState,int(-sys.maxsize),int(sys.maxsize),self.depth,Directions.STOP)

    return bestaction

  def maxvalue(self,gameState,alpha,beta,depth,action):
      if depth == 0 or gameState.isWin() or gameState.isLose():
          return self.evaluationFunction(gameState),action,alpha,beta

      value = int(-1*sys.maxsize)
      bestaction = Directions.STOP

      actions = gameState.getLegalActions(0)
      if actions.count('Stop') != 0:
          actions.remove('Stop')
      for action in actions:
          childstate = gameState.generateSuccessor(0,action)
          minivalue,direction,alpha,beta = self.minvalue(childstate,alpha,beta,depth,action)
          if minivalue > value:
              value = minivalue
              bestaction = action
          if value > beta:
              break
          alpha = max(alpha,value)

      return value,bestaction,alpha,beta

  def minvalue(self,gameState,alpha,beta,depth,action):
      depth -= 1
      if depth == 0 or gameState.isLose() or gameState.isWin():
          return self.evaluationFunction(gameState),action,alpha,beta


      value = int(sys.maxsize)
      bestaction = Directions.STOP

      for agent in range(1,gameState.getNumAgents()):
          actions = gameState.getLegalActions(agent)
          if actions.count('Stop') != 0:
              actions.remove('Stop')
          for action in actions:
              childstate = gameState.generateSuccessor(agent,action)
              maxivalue,direction,alpha,beta = self.maxvalue(childstate,alpha,beta,depth,action)
              if maxivalue < value:
                  value = maxivalue
                  bestaction = action
              if value < alpha:
                  break
              beta = min(beta,value)

      return value,bestaction,alpha,beta

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
    value,bestaction = self.maxvalue(gameState,self.depth,Directions.STOP)

    return bestaction


  def maxvalue(self, gameState, depth, action):
    if depth == 0 or gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState),action
    depth -= 1
    value = int(-1*sys.maxsize)
    bestaction = Directions.STOP

    actions = gameState.getLegalActions(0)
    if actions.count('Stop') != 0:
      actions.remove('Stop')
    for action in actions:
      childstate = gameState.generateSuccessor(0,action)
      expectvalue = self.expvalue(childstate,depth)
      if expectvalue > value:
        value = expectvalue
        bestaction = action

    return value,bestaction


  def expvalue(self, gameState, depth):
    if depth == 0 or gameState.isWin() or gameState.isLose():
      return self.evaluationFunction(gameState)

    value = 0

    for agent in range(1,gameState.getNumAgents()):
      actions = gameState.getLegalActions(agent)
      if actions.count('Stop') != 0:
        actions.remove('Stop')
      for action in actions:
        childstate = gameState.generateSuccessor(agent,action)
        maxivalue,direction = self.maxvalue(childstate,depth,action)
        value += 0.25*maxivalue

    return value

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  currentPos = currentGameState.getPacmanPosition()
  currentGhostStates = currentGameState.getGhostStates()
  currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
  currentFood = currentGameState.getFood()
  currentNumFood = currentGameState.getNumFood()
  currentCapsules = currentGameState.getCapsules()

  if 'foodarray' not in locals():
      foodarray = mypy.foodArray(currentFood)

  if 'capsulesarray' not in locals():
      capsulesarray = mypy.capsulesArray(currentCapsules)

  pacmanx,pacmany = currentPos

  groupsoffood = []
  fooddistance = []
  for pellet in foodarray:
      nextdoorfood = 0
      distancetofood = util.manhattanDistance(pellet,currentPos)
      if distancetofood < 8:
          distancetofood = mypy.mazeDistance(pellet,currentPos,currentGameState)
      fooddistance.append(distancetofood)
      x,y = pellet
      if currentFood[x+1][y]:
          nextdoorfood+=1
      if currentFood[x-1][y]:
          nextdoorfood+=1
      if currentFood[x][y+1]:
          nextdoorfood+=1
      if currentFood[x][y-1]:
          nextdoorfood+=1
      groupsoffood.append(nextdoorfood)

  capsuledistance = []
  for capsule in capsulesarray:
      distancetocapsule = util.manhattanDistance(capsule,currentPos)
      if distancetocapsule < 8:
          distancetocapsule = mypy.mazeDistance(capsule,currentPos,currentGameState)
      capsuledistance.append(distancetocapsule)

  ghostDistances = []
  ghostscared = []
  scaredghosts = []

  for ghost in range(1,currentGameState.getNumAgents()):
      ghostposition = currentGameState.getGhostPosition(ghost)
      distancetoghost = util.manhattanDistance(currentPos,ghostposition)
      scaredtimer = currentScaredTimes[ghost-1]
      if distancetoghost < 8:
          distancetoghost = mypy.mazeDistance(ghostposition,currentPos,currentGameState)
      ghostDistances.append(distancetoghost)
      ghostscared.append(scaredtimer)
      if scaredtimer != 0:
          scaredghosts.append(ghostposition)

  '''
  for ghostposition in currentGameState.getGhostPositions():
      distancetoghost = util.manhattanDistance(currentPos,ghostposition)
      if distancetoghost < 8:
          distancetoghost = mypy.mazeDistance(ghostposition,currentPos,currentGameState)
      ghostDistances.append(distancetoghost)
  '''

  ghostDistance = min(ghostDistances)

  mindistance = 0
  maxdistance = 0
  minnumfoods = 0
  maxnumfoods = 0
  averagefood = 0
  averagenumfoods = 0
  numcapsules = len(capsulesarray)

  currentscore = currentGameState.getScore()

  if len(fooddistance) != 0:
      mindistance = min(fooddistance)
      maxdistance = max(fooddistance)
      minnumfoods = min(groupsoffood)
      maxnumfoods = max(groupsoffood)
      averagenumfoods = sum(groupsoffood)/len(groupsoffood)
      averagefood = sum(fooddistance)/len(fooddistance)

  if len(scaredghosts) != 0:
      if min(scaredghosts) <8 and ghostscared.index(min(scaredghosts)) > 8:
          currentscore = 0.5*currentscore
          ghostDistance = -100*min(scaredghosts)

  elif min(currentScaredTimes) > 6 or ghostDistance > 6:
      ghostDistance = 0

  elif ghostDistance > 4 or min(currentScaredTimes) > 4:
      ghostDistance = 0.5*ghostDistance
      currentscore = 2*currentscore

  foodbonus = 0
  x,y = currentPos
  if (currentFood[x][y]):
      foodbonus = 50

  #evaluation = currentGameState.getScore() + foodbonus
  evaluation = currentscore + 2*ghostDistance - 1000*numcapsules - minnumfoods - maxdistance - mindistance - averagefood -averagenumfoods

  return evaluation

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

