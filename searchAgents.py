# searchAgents.py
# ---------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman
from util import manhattanDistance
from util import PriorityQueue

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
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
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

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
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
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

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState: pacman.GameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        #self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))     


        # EDIT
        self.startingPosition = (startingGameState.getPacmanPosition(),self.corners)   
        # Xây dựng state dựa trên 2 cơ sở 1: vị trí hiện tại, số lượng góc còn lại.
        # EDIT


        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded


    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"

        return self.startingPosition
        #util.raiseNotDefined()

    def isGoalState(self, state: Any):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        size = len(list(state[1]))

        return size == 0            # nếu state này ko còn góc thì ngừng
        
        #util.raiseNotDefined()

    def getSuccessors(self, state: Any):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        currentPosition,tuple_corner = state

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]
            "*** YOUR CODE HERE ***"
            x,y = currentPosition
            dx,dy = Actions.directionToVector(action)
            nextx,nexty = int(x+dx),int(y+dy)
            hitsWall = self.walls[nextx][nexty]

            if not hitsWall:
                new_tuple = tuple(i for i in tuple_corner if i != (nextx,nexty))

                new_state = ((nextx,nexty),new_tuple)
                cost = 1
                successors.append((new_state,action,cost))
            

        self._expanded += 1 # DO NOT CHANGE
        return successors


    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        pos,_= self.startingPosition                    # sửa đoạn này vì đổi cấu trúc state

        x,y = pos
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


def helper(pos,tuple_corner):           # Hàm dùng để tính heuristic từ điểm đó tới góc gần nhất, rồi tiếp tục từ góc đó tới góc gần thứ 2 và tiếp tục...
    result = 0
    tuple_corner = set(tuple_corner)
    

    while(len(list(tuple_corner)) != 0):
        min_distance_manhanttan = float('inf')
        corner_remove = None

        for i in tuple_corner:
            check_temp = abs(pos[0]-i[0]) + abs(pos[1]-i[1])

            if check_temp < min_distance_manhanttan:
                min_distance_manhanttan = check_temp
                corner_remove = i
        
        result += min_distance_manhanttan
        tuple_corner.remove(corner_remove)
        pos = corner_remove
    
    return result


def cornersHeuristic(state: Any, problem: CornersProblem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible.
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    "*** YOUR CODE HERE ***"
    result = 0
    pos,tuple_corner = state      

    return helper(pos,tuple_corner)  # Do chỉ có 4 góc nên sử dụng thẳng đường đi qua 4 góc đó dựa vào khoảng cách manhattan

    #return 0 # Default to trivial solution



class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState: pacman.GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information
        



    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost
# def manhattanHeuristic(position, position2, info={}):
#     "The Manhattan distance heuristic for a PositionSearchProblem"
#     xy1 = position
#     xy2 = position2
#     return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

# def compute_mst(points):
#     """
#     Using Prim Algorithm
#     """
#     if len(points) <= 1:
#         return 0

#     # Priority queue for Prim's algorithm
#     pq = PriorityQueue()
#     visited = set()
#     mst_cost = 0

#     # Start with the first point
#     start = points[0]
#     visited.add(start)

#     # Push initial edges to the priority queue
#     for point in points[1:]:
#         # if (start, point) not in problem.heuristicInfo:
#         #     problem.heuristicInfo[(position, next_food)] = mazeDistance(position, next_food, problem.startingGameState)
#         #     problem.heuristicInfo[(next_food, position)] = problem.heuristicInfo[(position, next_food)]
#         pq.push((start, point), manhattanDistance(start, point))

#     # Process the MST
#     while not pq.isEmpty():
#         start, end = pq.pop()  # Pop the lowest-cost edge
#         #print(cost)
#         cost = manhattanDistance(start, end)
#         if end not in visited:
#             visited.add(end)
#             mst_cost += cost

#             # Add new edges from the newly visited point
#             for other in points:
#                 if other not in visited:
#                     pq.push((end, other), manhattanDistance(end, other))

#     return mst_cost


def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your search may have a but our your heuristic is not admissible!  On the
    other hand, inadmissible heuristics may find optimal solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """

    # position, foodGrid = state
    # result = 0
    # food_list = foodGrid.asList()
    # total_distance = 0
    
    # if (len(food_list) == 0):
    #     return 0        
    # elif (len(food_list) == 1):
    #     return manhattanDistance(position,food_list[0]) # trường hợp chỉ có 1 food thì return khoảng cách manhattan của điểm xét tới food đó

    
    # max_distance=((0,0),(0,0),0)            # đầu tiên tìm khoảng cách lớn nhất giữa 2 điểm food
    
    # for current_food in food_list:
    #     for select_food in food_list:
    #         if(current_food==select_food):
    #             pass
    #         else:
    #             distance = manhattanDistance(current_food,select_food)
    #             if(max_distance[2] < distance):
    #                 max_distance = (current_food,select_food,distance)


    # d1 = manhattanDistance(position,max_distance[0]) 
    # d2 = manhattanDistance(position,max_distance[1])
    # # sau đó xét từ điểm position tới 2 điểm có khoảng cách xa nhất với nhau, tính đoạn gần hơn rồi return max_đã_tính + min tới 1 trong 2 điểm
    # result = max_distance[2] + min(d1,d2)
    
    # return result

    """
    Version 1: Manhanttan distance

    - 0 food => 0
    - 1 food => return distance pos -> food
    - >= 2 => Most distance from 2 points plus the smallest distance from that the pos to 1 point food

    - Expanded node = 7729 nodes
    - Time = 1.55s
    
    """

    """
    MST-based heuristic for the FoodSearchProblem using the provided PriorityQueue.
    """
    # pacmanPosition, foodGrid = state
    # foodList = foodGrid.asList()

    # if not foodList:
    #     return 0  # No food left, heuristic is 0 at the goal state.

    # # Compute the MST cost for the food points only
    # mst_cost = compute_mst(foodList)

    # # Add the distance from Pacman to the closest food point
    # min_dist_to_food = min(manhattanDistance(pacmanPosition, food) for food in foodList)

    # return mst_cost + min_dist_to_food
    
    """
    Version 2: MST

    - Expanded node = 7209 nodes
    - Time = 1.86s
    
    """

    """
    Code ở dưới bỏ do có sử dụng hàm mazeDistance()
    """


    # position, foodGrid = state
    # "*** YOUR CODE HERE ***"
    # # return 0
    # if foodGrid.count() == 0: return 0

    # result = 0
    # food_list = foodGrid.asList()

    # for next_food in food_list:

    #     if (position, next_food) not in problem.heuristicInfo:
    #         problem.heuristicInfo[(position, next_food)] = mazeDistance(position, next_food, problem.startingGameState)
    #         problem.heuristicInfo[(next_food, position)] = problem.heuristicInfo[(position, next_food)]
    #         # lưu 2 chiều
    #         # Cách này dùng chính xác khoảng cách từ pos tới điểm food bằng cách dùng mazeDistance
    #         # Và chỉ lấy khoảng cách lớn nhất, đồng thời lưu kết quả để lúc sau có thể truy vấn

    #     result = max(result, problem.heuristicInfo[(position, next_food)])

    # return result


    """
    Version 3: Sử dụng hàm mazeDistance để lấy khoảng cách từ pos tới 1 điểm food chính xác 
    thay vì sử dụng khoảng cách manhattan ko còn tính chính xác nhiều

    - Ta lấy khoảng cách max làm heuristic, đồng thời lưu vào set problem.heuristicInfo()

    - Expanded node = 4239
    - Time = 0.789s
    
    """

    position, foodGrid = state
    distances = []
    distances_food = [0]
    food_list = foodGrid.asList()
    for food in food_list:
        pos_to_food = 0
        if (position, food) not in problem.heuristicInfo:
            problem.heuristicInfo[(position, food)] = mazeDistance(position,food,problem.startingGameState)
            problem.heuristicInfo[(food, position)] = problem.heuristicInfo[(position, food)]
      
        pos_to_food = problem.heuristicInfo[(position, food)] 
        #Giống version 3 nhưng tạm thời lưu các kết quả vào 1 cái listtemp

        distances.append(pos_to_food)
        for tofood in food_list:
            food_to_food = 0
            if (food, tofood) not in problem.heuristicInfo:
                problem.heuristicInfo[(food, tofood)] = mazeDistance(food,tofood,problem.startingGameState)
                problem.heuristicInfo[(tofood, food)] = problem.heuristicInfo[(food, tofood)]

            food_to_food = problem.heuristicInfo[(food, tofood)]
            # Tính thêm khoảng cách từ các food với nhau

            distances_food.append(food_to_food)
    # Return khoảng cách từ pos tới food gần nhất + khoảng cách lớn nhất giữa 2 food
    return min(distances) + max(distances_food) if len(distances) else max(distances_food)


    """
    Version 4: Cải tiến hơn khi kết hợp ver2 + 3: Sử dụng tổng đường min từ pos tới 1 food + 1 max giữa 2 food

    Tính admission: Sẽ giải thích trong rp sau

    - Expanded node = 727
    - Times = 0.22 s
    
    # """

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState: pacman.GameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        from search import depthFirstSearch
        from search import breadthFirstSearch
        from search import uniformCostSearch
        from search import aStarSearch


        """

        Câu này chỉ đơn giản là gọi hàm
        - Trong 4 hàm đã implement thì chỉ có thể sử dụng BFS,UCS,Astar.

        - DFS ko thể sử dụng do => is goal đang chọn là chỉ là food thì return true nên quá trình tìm kiếm của nó khi gặp 1 
        food sẽ bắt đầu tìm tiếp. Nhưng càng tìm rộng ra thì điểm bất cập là nó ko thật sự tìm điểm gần nhất 
        => Do cơ chế chọn successor là random nên có thể hiểu là ví dụ giữa 2 bên trái và phải nếu ngay bên trái là 1 food
        và khi thuật toán tìm kiếm chọn bên phải tìm trước => Nó sẽ tìm tới khi nào bên phải ko có mới qua bên trái

        => Ko đúng ý muốn thuật toán

        - BFS thì tìm theo từng hàng nên luôn đảm bảo closest dot

        - UCS thì tìm theo path đang có giá trị nhỏ nhất nên cũng đúng

        - Astar thì thật ra lúc gọi nó pass vào hàm nullheuristic nên Astar lúc này = UCS


        """

        #return depthFirstSearch(problem)
        return breadthFirstSearch(problem)
        #return uniformCostSearch(problem)
        #return aStarSearch(problem)
        util.raiseNotDefined()

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state: Tuple[int, int]):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"

        #print(self.food)

        return self.food[x][y] # True nghia la o do co food
        util.raiseNotDefined()

def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
