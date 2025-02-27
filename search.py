# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List
from util import Stack
from util import Queue
from util import PriorityQueue
#import time

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # temp = problem.getSuccessors(problem.getStartState())

    # problem.getSuccessors(temp[0][0])                     # cái expand là khi gọi hàm getSuccesssor của thằng tại vị trí đó
    # return []
    
    fringe = Stack()
    start = problem.getStartState()
    fringe.push(start)
    parent_find = {start:None} # hashmap
    visited = set()

    while not fringe.isEmpty():
        
        currentState = fringe.pop()

        visited.add(currentState)
        #print("Curr:", currentState)
        if problem.isGoalState(currentState):
            return_path = []
            while True:
                if parent_find[currentState] != None:
                    return_path.append(parent_find[currentState][1])
                    currentState = parent_find[currentState][0]
                else:
                    break
                
            return list(reversed(return_path))
        
        temp_list = problem.getSuccessors(currentState)
        for next_state,direction,cost in temp_list:
            if next_state in visited:
                continue
            else:
                fringe.push(next_state)
                parent_find[next_state]=(currentState,direction)

    """
    Code đang gặp vấn đề bị lặp lại quá trình gán parent_find mỗi khi một node đào sâu xuống và tìm thấy 1 node đã bắt gặp được ở trên.
    Code cũ ko có visited  và trong vòng for 2 của while để điều kiện check là => gặp lỗi ko đúng khi chạy test case số 2 của bfs vs dfs

    diagram: 
        /-- B
        |   ^
        |   |
        |  *A -->[G]
        |   |     ^
        |   V     |
        \-->D ----/


        Nguyên nhân là do nó tìm A : giai đoạn này đã gán parent cho  B D G
        Sau đó tới D nó ko gán lại => khi gặp G thì nó lấy đường từ A -> G (Sai)

        Đã chỉnh cho gán lại và đúng 3/3 nhưng có vẻ chưa optimize chỗ này (dùng thêm 1 set visited)

    """
    return None
    #util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    fringe = Queue()
    start = problem.getStartState()
    fringe.push(start)
    parent_find = {start:None} # hashmap

    while not fringe.isEmpty():
        currentState = fringe.pop()

        #print("Curr:", currentState)
        if problem.isGoalState(currentState):
            return_path = []
            while True:
                if parent_find[currentState] != None:
                    return_path.append(parent_find[currentState][1])
                    currentState = parent_find[currentState][0]
                else:
                    break
                
            return list(reversed(return_path))
        
        temp_list = problem.getSuccessors(currentState)
        for next_state,direction,cost in temp_list:
            if next_state in parent_find:
                continue
            else:
                fringe.push(next_state)
                parent_find[next_state]=(currentState,direction)



    """
    Hàm này thì ko sử dụng visited nữa do tưởng tượng rằng khi 1 node phía sau 
    gặp 1 node đã có tìm parent nghĩa là cái đường phía trước chắc chắn nhỏ hơn nên giữ nguyên đường cũ
    """

    #util.raiseNotDefined()

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    fringe = PriorityQueue()
    start = problem.getStartState()
    fringe.push((start,0),0)
    parent_find = {start:None} # hashmap

    while not fringe.isEmpty():
        
        (currentState,cost_current) = fringe.pop()
        #print("Curr:", currentState)
        if problem.isGoalState(currentState):
            return_path = []
            while True:
                if parent_find[currentState] != None:
                    return_path.append(parent_find[currentState][1])
                    currentState = parent_find[currentState][0]
                else:
                    break
                
            return list(reversed(return_path))
        
        temp_list = problem.getSuccessors(currentState)
        for next_state,direction,cost in temp_list:
            #print("temp:", next_state)
            if next_state not in parent_find:
                fringe.update((next_state,cost+cost_current),cost+cost_current)
                parent_find[next_state]=(currentState,direction,cost+cost_current)
            elif parent_find[next_state] == None: # Trường họp đang xét lại thằng gốc ban đầu
                continue
            elif parent_find[next_state][2] > cost+cost_current:
                fringe.update((next_state,cost+cost_current),cost+cost_current)
                parent_find[next_state]=(currentState,direction,cost+cost_current)

        # print()
        # print()
        # print()
    
    """
    Giống như BFS, UCS tìm từ những đoạn đường nhỏ nhất trước, 
    nên khi gặp 1 node kéo ra đã có trong parent thì ko xét tiếp nx, do đường trước đó ngắn hơn

    =>> Cũng ko xài visited

    """

    
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    # Hàm này ko implement do code pass heuristic bằng terminal

    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    #start_time = time.time()
    fringe = PriorityQueue()
    start = problem.getStartState()
    fringe.push((start,0),0)
    parent_find = {start:None} # hashmap

    while not fringe.isEmpty():    
        (currentState,cost_current) = fringe.pop()
        #print("Curr:", currentState)
        if problem.isGoalState(currentState):
            return_path = []
            while True:
                if parent_find[currentState] != None:
                    return_path.append(parent_find[currentState][1])
                    currentState = parent_find[currentState][0]
                else:
                    break

            #end_time = time.time()

            #print(end_time - start_time)

            return list(reversed(return_path))
        
        temp_list = problem.getSuccessors(currentState)
        for next_state,direction,cost in temp_list:
            #print("temp:", next_state)

            heuristic_value = heuristic(next_state,problem)

            if next_state not in parent_find:
                fringe.update((next_state,cost+cost_current),cost+cost_current+heuristic_value)
                parent_find[next_state]=(currentState,direction,cost+cost_current)
            elif parent_find[next_state] == None:
                continue
            elif parent_find[next_state][2] > cost+cost_current:
                fringe.update((next_state,cost+cost_current),cost+cost_current+heuristic_value)
                parent_find[next_state]=(currentState,direction,cost+cost_current)

    """
    Tương tự UCS, ta cũng ko xài visited
    """

    #util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch