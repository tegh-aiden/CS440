#Author: Tegh Aiden
#NetID: tsa45
#This work is exclusively my own.

import numpy as np
from numpy.linalg import norm
import math
import queue
import matplotlib.pyplot as plt

np.set_printoptions(formatter={'float_kind':"{:.2f}".format})

def manhattanDist(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def neighbors(shape, pos):
    x = pos[0]
    y = pos[1]
    adjascent = {(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)}
    neighbors = set()
    for cell in adjascent:
        a = cell[0]
        b = cell[1]
        if (0 <= a < shape[0]) and (0 <= b < shape[1]):
            neighbors.add(cell)
    return neighbors

class Ridge:
    def __init__(self, shape, ravines, goal, loss):
        self.ravines = ravines
        self.loss = loss
        self.goal = goal
        self.shape = shape

        #Initialize transitions costs to 1
        self.costArr = np.full(self.shape, 1)
        #Initialize utility values with manhattan dist as initial guess
        self.utilArr = np.empty(self.shape)
        for index in np.ndindex(self.shape):
            self.utilArr[index] = manhattanDist(index, self.goal)

        for ravine in self.ravines:
            self.costArr[ravine] = loss
            self.utilArr[ravine] = 0

        self.utilArr[self.goal] = 0

        self.action = np.full(self.shape, -1)

        self.beta = 1

        print(self.costArr)
        for x in range(1500):
            self.update()

        print(self.utilArr)
        print(self.action)


    def slideProbs(self, utils, x):
        probs = np.zeros(4)
        if x == 0 or x == 2:
            if utils[1] == math.inf or utils[3] == math.inf:
                probs[x] += 0.9
                if utils[1] == math.inf:
                    probs[3] += 0.1
                else:
                    probs[1] += 0.1
            else:
                probs[x] += 0.8
                probs[1] += 0.1
                probs[3] += 0.1
        else:
            if utils[0] == math.inf or utils[2] == math.inf:
                probs[x] += 0.9
                if utils[0] == math.inf:
                    probs[2] += 0.1
                else:
                    probs[0] += 0.1
            else:
                probs[x] += 0.8
                probs[0] += 0.1
                probs[2] += 0.1
        return probs

    def update(self):
        updatedUtils = np.empty(self.shape)
        for index in np.ndindex(self.shape):
            if index in self.ravines or index == self.goal:
                updatedUtils[index] = 0
                continue
            preUtils = np.full(4, math.inf)
            costs = np.zeros(4)
            x = index[0]
            y = index[1]
            #north = 0
            if x > 0:
                preUtils[0] = self.utilArr[(x-1,y)]
                costs[0] = self.costArr[(x-1,y)]
            #east = 1
            if y < self.shape[1] - 1:
                preUtils[1] = self.utilArr[(x,y+1)]
                costs[1] = self.costArr[(x,y+1)]
            #south = 2
            if x < self.shape[0] - 1:
                preUtils[2] = self.utilArr[(x+1,y)]
                costs[2] = self.costArr[(x+1,y)]
            #west = 3
            if y > 0:
                preUtils[3] = self.utilArr[(x,y-1)]
                costs[3] = self.costArr[(x,y-1)]
            possibleUtils = np.full(4, math.inf)
            for a in range(4):
                if preUtils[a] != math.inf:
                    probs = self.slideProbs(preUtils, a)
                    util = 0
                    for b in range(4):
                        if preUtils[b] != math.inf:
                            util += probs[b] * (costs[b] + self.beta * preUtils[b])
                    possibleUtils[a] = util
            for g in range(4):
                if possibleUtils[g] == np.amin(possibleUtils):
                    self.action[index] = g
            updatedUtils[index] = np.amin(possibleUtils)
        self.utilArr = updatedUtils

shape = (17,5)
ravines = set()

for a in range(2,7):
    ravines.add((a,0))

for a in range(10,15):
    ravines.add((a,0))

for a in range(6,11):
    ravines.add((a,4))

goal = (16,0)
loss = 15

r = Ridge(shape, ravines, goal, loss)
