from game import Directions
from game import Actions
import util, sys

def foodArray(fooddoublearray):
    food = []
    width = len(fooddoublearray[:])
    height = len(fooddoublearray[0])

    for xindex in range(0,width):
        for yindex in range(0,height):
            if fooddoublearray[xindex][yindex]:
                food.append((xindex,yindex))
            yindex+=1
        xindex+=1
    return food

def capsulesArray(capsulesdoublearray):
    capsules = []
    if len(capsulesdoublearray) == 0:
        return capsules
    width = len(capsulesdoublearray[:])
    height = len(capsulesdoublearray[0])

    for xindex in range(0,width):
        for yindex in range(0,height):
            if capsulesdoublearray[xindex][yindex]:
                capsules.append((xindex,yindex))
            yindex+=1
        xindex+=1
    return capsules

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []  #No movements needed, empty array
    frontier = util.PriorityQueue()  #New queue frontier
    frontier.push((start,[],0),0)
    explored = []  #Empty array explored, we haven't been anywhere yet
    while not frontier.isEmpty():
        node = frontier.pop()  #Remove least cost element from priority queue
        if problem.isGoalState(node[0]):  #Location node[0] is the goal location
            return (node[1]) #Return list of actions node[1]
        explored.append(node[0])
        for child in problem.getSuccessors(node[0]):
            if (child[0] not in explored):
                path = list(node[1])  #Put all the actions taken to get to node
                cost = child[2]+node[2]
                path.append(child[1])  #Plus the action taken to get to child from node
                frontier.push((child[0],path,cost),cost)  #Add child to the frontier


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []  #No movements needed, empty array
    frontier = util.Queue()  #New queue frontier
    frontier.push((start,[]))
    explored = []  #Empty array explored, we haven't been anywhere yet
    while not frontier.isEmpty():
        node = frontier.pop()  #Remove last element added to frontier LIFO
        if node[0] not in explored:
            if problem.isGoalState(node[0]):  #Location node[0] is the goal location
                return node[1] #Return list of actions node[1]
            explored.append(node[0])
            for child in problem.getSuccessors(node[0]):
                path = list(node[1])  #Put all the actions taken to get to node
                path.append(child[1])  #Plus the action taken to get to child from node
                frontier.push((child[0],path))  #Add child to the frontier

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built.  The gameState can be any game state -- Pacman's position
    in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    walls = gameState.getWalls()
    #print x1,y1,x2,y2
    if walls[x1][y1] or walls[x2][y2]:
        return sys.maxint
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False)
    return len(uniformCostSearch(prob))

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test,
    successor function and cost function.  This search problem can be
    used to find paths to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self._visited, self._visitedlist, self._expanded = {}, [], 0

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal
        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

