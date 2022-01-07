#Author: Tegh Aiden
#NetID: tsa45
#This work is exclusively my own.

import numpy as np
from numpy.linalg import norm
import math
import queue
import matplotlib.pyplot as plt

def neighbors(dim, pos):
    x = pos[0]
    y = pos[1]
    adjascent = {(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y),
            (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)}
    neighbors = set()
    for cell in adjascent:
        a = cell[0]
        b = cell[1]
        if (0 <= a < dim) and (0 <= b < dim):
            neighbors.add(cell)
    return neighbors

class Environment:
    def __init__(self, dim, m):
        # dim -> int
        # m -> num of mines

        self.dim = dim
        self.m = m

        if dim < 2:
            raise ValueError("Invalid dimension")

        if m < 0 or m > dim * dim:
            raise ValueError("Invalid number of mines")

        # Create dim * dim array of 0s and randomly change m of them to -1s.
        # -1s represent mines

        self.map = np.zeros((dim, dim), dtype = np.short)
        self.mines = np.random.choice(dim*dim, m, replace = False)
        for x in self.mines:
            cell = (x // dim, x % dim)
            self.map[cell] = -1
            for neighbor in neighbors(dim, cell):
                if self.map[neighbor] != -1:
                    self.map[neighbor] += 1

    def printMap(self):
        for x in range(self.dim):
            for y in range(self.dim):
                if self.map[x][y] < 0:
                    print(self.map[x][y], end = " ")
                else:
                    print(end = " ")
                    print(self.map[x][y], end = " ")
            print()

class Agent:
    def __init__(self, env):
        if not isinstance(env, Environment):
            raise ValueError("Invalid environment")
        self.env = env
        self.dim = env.dim
        self.corners = {(0, 0), (0, self.dim - 1),\
         (self.dim - 1, 0), (self.dim - 1, self.dim - 1)}

        # Initialize every element in map to [-2, 0, 0, # of neighbors]
        # map[x][y][0] -
        # >= 0 -> Discovered safe cell / clue
        #  -1 -> Discovered mine
        #  -2 -> Undiscovered cell
        #  -3 -> Flag

        # map[x][y][1] - number of adjascent mines
        # map[x][y][2] - number of adjascent safe cells
        # map[x][y][3] - number of adjascent hidden cells

        self.map = np.zeros((self.dim, self.dim, 4), dtype = np.short)
        for cell in np.ndindex(self.map.shape[:2]):
            self.map[cell][0] = -2
            self.map[cell][3] = self.degree(cell)


    def printMap(self, a):
        for x in range(self.dim):
            for y in range(self.dim):
                if self.map[x][y][a] < 0:
                    print(self.map[x][y][a], end = " ")
                else:
                    print(end = " ")
                    print(self.map[x][y][a], end = " ")
            print()

    def degree(self, pos):
        if pos in self.corners:
            degree = 3
        elif pos[0] == 0 or pos[0] == (self.dim - 1) or pos[1] == 0 or pos[1] == (self.dim - 1):
            degree = 5
        else:
            degree = 8
        return degree

    def query(self, pos):
        x = self.env.map[pos]
        self.map[pos][0] = x
        if x == -1:
            for neighbor in neighbors(self.dim, pos):
                self.map[neighbor][1] += 1
                self.map[neighbor][3] -= 1
        else:
            for neighbor in neighbors(self.dim, pos):
                self.map[neighbor][2] += 1
                self.map[neighbor][3] -= 1

    def flag(self, pos):
        self.map[pos][0] = -3
        for neighbor in neighbors(self.dim, pos):
            self.map[neighbor][1] += 1
            self.map[neighbor][3] -= 1

    def basicStrat(self):
        hidden = set(np.ndindex(self.map.shape[:2]))
        fringe = queue.Queue()
        flags = 0
        #While set of hidden cells is not empty...
        while hidden:
            pos = hidden.pop()
            self.query(pos)
            fringe.put(pos)
            visited = {(pos, self.degree(pos))}
            while not fringe.empty():
                pos = fringe.get()
                if self.map[pos][0] - self.map[pos][1] == self.map[pos][3]:
                    #If clue - # of discovered mines == # of hidden, flag all hidden neighbors
                    #non = "Neighbor of Neighbor"
                    for neighbor in neighbors(self.dim, pos):
                        if self.map[neighbor][0] == -2:
                            self.flag(neighbor)
                            flags += 1
                            hidden.remove(neighbor)
                            for non in neighbors(self.dim, neighbor):
                                if self.map[non][0] >= 0 and (non, self.map[non][3]) not in visited:
                                    fringe.put(non)
                                    visited.add((non, self.map[non][3]))
                elif (self.degree(pos) - self.map[pos][0]) - self.map[pos][2] == self.map[pos][3]:
                    #If (8 - clue) - # of discovered safe cells == # of hidden, all hidden neighbors are safe
                    for neighbor in neighbors(self.dim, pos):
                        if self.map[neighbor][0] == -2:
                            self.query(neighbor)
                            hidden.remove(neighbor)
                            fringe.put(neighbor)
                            for non in neighbors(self.dim, neighbor):
                                if self.map[non][0] >= 0 and (non, self.map[non][3]) not in visited:
                                    fringe.put(non)
                                    visited.add((non, self.map[non][3]))
        return flags / self.env.m

    def one_one(self, a, b):
        a_neighbors = set()
        for neighbor in neighbors(self.dim, a):
            if self.map[neighbor][0] == -2:
                a_neighbors.add(neighbor)
        b_neighbors = set()
        for neighbor in neighbors(self.dim, b):
            if self.map[neighbor][0] == -2:
                b_neighbors.add(neighbor)
        #If a is a strict subset of b...
        if a_neighbors < b_neighbors:
            return b_neighbors.difference(a_neighbors)
        else:
            return None

    def advancedStrat(self):
        hidden = set(np.ndindex(self.map.shape[:2]))
        discovered = set()
        fringe = queue.Queue()
        flags = 0
        #While set of hidden cells is not empty...
        while hidden:
            while not fringe.empty():
                pos = fringe.get()
                if self.map[pos][0] - self.map[pos][1] == self.map[pos][3]:
                    #If clue - # of discovered mines == # of hidden, flag all hidden neighbors
                    #non = "Neighbor of Neighbor"
                    for neighbor in neighbors(self.dim, pos):
                        if self.map[neighbor][0] == -2:
                            self.flag(neighbor)
                            flags += 1
                            hidden.remove(neighbor)
                            discovered.add(neighbor)
                            for non in neighbors(self.dim, neighbor):
                                if self.map[non][0] >= 0 and (non, self.map[non][3]) not in visited:
                                    fringe.put(non)
                                    visited.add((non, self.map[non][3]))
                elif (self.degree(pos) - self.map[pos][0]) - self.map[pos][2] == self.map[pos][3]:
                    #If (8 - clue) - # of discovered safe cells == # of hidden, all hidden neighbors are safe
                    for neighbor in neighbors(self.dim, pos):
                        if self.map[neighbor][0] == -2:
                            self.query(neighbor)
                            hidden.remove(neighbor)
                            discovered.add(neighbor)
                            fringe.put(neighbor)
                            for non in neighbors(self.dim, neighbor):
                                if self.map[non][0] >= 0 and (non, self.map[non][3]) not in visited:
                                    fringe.put(non)
                                    visited.add((non, self.map[non][3]))
            ones = set()
            unsafe = set()
            for pos in discovered:
                if self.map[pos][0] == 1 and self.map[pos][1] == 0:
                    for neighbor in neighbors(self.dim, pos):
                        if self.map[neighbor][0] == 1 and self.map[neighbor][1] == 0:
                            temp = self.one_one(pos, neighbor)
                            if temp != None:
                                unsafe.update(temp)
            if unsafe:
                for cell in unsafe:
                    self.query(cell)
                    hidden.remove(cell)
                    discovered.add(cell)
                    fringe.put(cell)
                    for neighbor in neighbors(self.dim, cell):
                        if self.map[neighbor][0] >= 0 and (neighbor, self.map[neighbor][3]) not in visited:
                            fringe.put(neighbor)
                            visited.add((neighbor, self.map[neighbor][3]))
            elif hidden:
                pos = hidden.pop()
                discovered.add(pos)
                self.query(pos)
                fringe.put(pos)
                visited = {(pos, self.degree(pos))}
        return flags / self.env.m

"""
q = np.empty(21, dtype=object)
p = np.empty(21, dtype=object)

for i in range(1, 11):
    q[i] = 0.1 * i
    success = 0
    for j in range(51):
        e = Environment(10, 10*i)
        a = Agent(e)
        success += a.advancedStrat()
    p[i] = success / 50.0

print(p)
print(q)

plt.plot(q, p)
plt.title("Advanced Strategy")
plt.ylabel("Average Success Rate")
plt.xlabel("Mine Density")
x1, x2, y1, y2 = plt.axis()
plt.axis([0.1, 1, 0, 1])
plt.show()
"""
