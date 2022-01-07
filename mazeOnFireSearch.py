#Author: Tegh Aiden
#NetID: tsa45
#This work is exclusively my own.


import numpy as np
from numpy.linalg import norm
import math
import queue
import matplotlib.pyplot as plt

def printPath(pathDict, start, goal):
    x = start
    while not x == goal:
        x = pathDict[x]

class Maze:
    def __init__(self, dim, prob, flam = None):
        """ dim -> int
            prob -> double between 0 and 1
            flam -> double between 0 and 1
        """
        self.dim = dim
        self.prob = prob
        self.flam = flam
        self.nearFire = dict()

        if prob < 0 or prob > 1:
            raise ValueError("Probablility must be between 0 and 1")

        if (not flam == None) and (flam < 0 or flam > 1):
            raise ValueError("Flammability must be between 0 and 1")

        #Generate dim x dim array of random floats
        self.map = np.random.uniform(size = (dim, dim))
        #Ints are unblocked (0) with probability (1 - prob) and blocked (1) with probability (prob)
        self.map = (prob > self.map).astype(int)
        #Make sure start and goal are unblocked
        self.map[0, 0] = self.map[dim - 1, dim - 1] = 0
        #Randomly select an element to be the initial pos of the fire
        self.firePos = tuple()
        self.startFire()

    def printMap(self):
        for cell in self.map:
            print(cell)

    def startFire(self):
        #Collect all indexes containing a zero
        zeros = np.argwhere(self.map == 0)
        #Select a random index within the length of zeros
        x = np.random.randint(np.size(zeros, axis = 0) - 1)
        #Set the randomly selected open block on fire
        pos = (zeros[x][0], zeros[x][1])
        self.firePos = pos
        self.map[pos[0], pos[1]] = 2
        for neighbor in self.neighbors(pos):
            self.nearFire[neighbor] = 1

    def advanceFire(self):
        ignited = set()
        for cell in self.nearFire:
            x = np.random.random()
            p = 1.0 - (np.power(1.0 - self.flam, self.nearFire[cell]))
            if x < p:
                 ignited.add(cell)
        for cell in ignited:
            del self.nearFire[cell]
            self.map[cell[0], cell[1]] = 2
            for neighbor in self.neighbors(cell):
                if self.nearFire.get(neighbor) == None:
                    self.nearFire[neighbor] = 1
                else:
                    self.nearFire[neighbor] = self.nearFire[neighbor] + 1

    def advanceLookahead(self, lookahead):
        for cell in lookahead:
            a = cell[0]
            b = cell[1]
            self.map[a][b] = 3

    def retreatLookahead(self, lookahead):
        for cell in lookahead:
            a = cell[0]
            b = cell[1]
            self.map[a][b] = 0

    def neighbors(self, pos):
        x = pos[0]
        y = pos[1]
        validNeighbors = set()
        possibleNeighbors = {(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)}
        for cell in possibleNeighbors:
            a = cell[0]
            b = cell[1]
            if (0 <= a < self.dim) and (0 <= b < self.dim):
                if self.map[a][b] == 0:
                    validNeighbors.add(cell)
        return validNeighbors

class Solve:
    def __init__(self, maze):
        self.maze = maze
        self.dim = maze.dim
        self.start = (0, 0)
        self.goal = (maze.dim - 1, maze.dim - 1)

    def dfs(self, start = None, goal = None):
        if start == None:
            start = self.start
        if goal == None:
            goal = self.goal
        stack = queue.LifoQueue()
        stack.put(start)
        visited = {start}
        while not stack.empty():
            pos = stack.get()
            if (pos == goal):
                return True
            else:
                for cell in self.maze.neighbors(pos):
                    if not cell in visited:
                        stack.put(cell)
                        visited.add(cell)
        return False

    def bfs(self, start = None, goal = None):
        if start == None:
            start = self.start
        if goal == None:
            goal = self.goal
        q = queue.Queue()
        q.put(start)
        visited = set()
        visited.add(start)
        while not q.empty():
            pos = q.get()
            if pos == goal:
                return (True, len(visited))
            else:
                for cell in self.maze.neighbors(pos):
                    if cell not in visited:
                        q.put(cell)
                        visited.add(cell)
        return (False, len(visited))

    @staticmethod
    def euclidean(a, b):
        a = np.array(a)
        b = np.array(b)
        return norm(a - b)

    def a_star_euclidean(self, start = None, goal = None):
        if start == None:
            start = self.start
        if goal == None:
            goal = self.goal
        fringe = queue.PriorityQueue()
        fringe.put((0, start))
        visited = set()
        visited.add(start)
        cost = {start : 0}
        estimate = {start : self.euclidean(start, goal)}
        while not fringe.empty():
            pos = fringe.get()
            pos = pos[1]
            if (pos == goal):
                return (True, len(visited))
            else:
                for cell in self.maze.neighbors(pos):
                    newCost = cost[pos] + 1
                    visited.add(cell)
                    #Default value must be greater than maximum possible estimate
                    if (newCost < cost.get(cell, self.dim * self.dim)):
                        cost[cell] = newCost
                        fringe.put((cost[cell] + self.euclidean(cell, goal), cell))
        return (False, len(visited))

    #Different return value as other a_star
    def a_star_euclidean_shortest_path(self, start = None, goal = None):
        if start == None:
            start = self.start
        if goal == None:
            goal = self.goal
        fringe = queue.PriorityQueue()
        fringe.put((0, start))
        visited = {start}
        parentDict = dict()
        cost = {start : 0}
        estimate = {start : self.euclidean(start, goal)}
        while not fringe.empty():
            pos = fringe.get()
            pos = pos[1]
            if (pos == goal):
                shortestPath = dict()
                cell = goal
                parent = parentDict.get(cell)
                while not parent == None:
                    shortestPath[parent] = cell
                    cell = parent
                    parent = parentDict.get(cell)
                return shortestPath
            else:
                for cell in self.maze.neighbors(pos):
                    newCost = cost[pos] + 1
                    #Default value must be greater than maximum possible estimate
                    if (newCost < cost.get(cell, self.dim * self.dim)):
                        cost[cell] = newCost
                        parentDict[cell] = pos
                        fringe.put((cost[cell] + self.euclidean(cell, goal), cell))
        return False

    def validMaze(self):
        if not self.dfs():
            return False
        #Need to temporarily set firePos to open for DFS to work
        self.maze.map[self.maze.firePos[0]][self.maze.firePos[1]] = 0
        if not self.dfs(start = None, goal = self.maze.firePos):
            return False
        self.maze.map[self.maze.firePos[0]][self.maze.firePos[1]] = 2
        return True

    def strat1(self):
        if not self.validMaze() :
            return -1
        shortestPath = self.a_star_euclidean_shortest_path()
        cell = self.start
        while not cell == self.goal:
            cell = shortestPath[cell]
            a = cell[0]
            b = cell[1]
            if self.maze.map[a][b] > 0:
                return 0
            self.maze.advanceFire()
        return 1

    def strat2(self):
        if not self.validMaze():
            return -1
        cell = self.start
        while not cell == self.goal:
            shortestPath = self.a_star_euclidean_shortest_path(start = cell, goal = self.goal)
            if not shortestPath:
                return 0
            cell = shortestPath[cell]
            self.maze.advanceFire()
        return 1

    def strat3(self):
        if not self.validMaze():
            return -1
        cell = self.start
        while not cell == self.goal:
            shortestPath = self.a_star_euclidean_shortest_path(start = cell, goal = self.goal)
            if not shortestPath:
                return 0
            lookahead = set()
            for adjascent in self.maze.nearFire:
                lookahead.add(adjascent)
            self.maze.advanceLookahead(lookahead)
            cautiousPath = self.a_star_euclidean_shortest_path(start = cell, goal = self.goal)
            self.maze.retreatLookahead(lookahead)
            if cautiousPath:
                cell = cautiousPath[cell]
            else:
                cell = shortestPath[cell]
            self.maze.advanceFire()
        return 1


maze = Maze(15, 0.1, 0.4)
Solve(maze).strat2()

"""
q = np.empty(21, dtype=object)
p = np.empty(21, dtype=object)

for i in range(21):
    q[i] = 0.05 * i
    successes = 0.0
    j = 0
    while j < 20:
        maze = Maze(100, 0.3, q[i])
        s = Solve(maze).strat1()
        print(s)
        if s == -1:
            continue
        elif s == 1:
            successes += 1.0
        j += 1
    p[i] = successes / 20.0

print(p)
print(q)

plt.plot(q, p)
plt.title("Strategy 1")
plt.ylabel("Average Success Rate")
plt.xlabel("Flammability")
x1, x2, y1, y2 = plt.axis()
plt.axis([0, 1, 0, 1])
plt.show()
"""
