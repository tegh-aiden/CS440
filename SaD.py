import numpy as np
import math
import queue

def manhattanDist(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def neighbors(pos, x, y):
    x = pos[0]
    y = pos[1]
    validNeighbors = set()
    possibleNeighbors = {(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)}
    for cell in possibleNeighbors:
        a = cell[0]
        b = cell[1]
        if (0 <= a < x) and (0 <= b < y):
            validNeighbors.add(cell)
    return validNeighbors

def nOfN(pos, x, y):
    a = pos[0]
    b = pos[1]
    if (a == 0 or a == x - 1) and (b == 0 or b == y - 1):
        return 2
    elif a == 0 or a == x - 1 or b == 0 or b == y - 1:
        return 3
    else:
        return 4

class SnD_stationarytarget:
    def __init__(self, dim1, dim2 = None):
        self.x = dim1
        self.y = dim2 if dim2 != None else dim1

        #Create an x by y map and randomly assign terrain types
        self.map = np.random.choice([0.1, 0.3, 0.7, 0.9], (self.x, self.y,), replace = True)
        self.size = np.size(self.map)
        #Randomly select target
        self.target = (np.random.randint(0, self.x), np.random.randint(0, self.y))
        #Create an array of probabilities and initialize each to 1 / num of cells
        self.probs = np.full((self.x, self.y), 1 / self.size)

    def query(self, pos):
        if pos == self.target:
            #If guess is correct, return False with probability terrainDict[pos]
            return np.random.random() > self.map[pos]
        else:
            #If guess is incorrect, return False
            return False

    def update(self, pos):
        prior = self.probs[pos]
        self.probs[pos] *= self.map[pos]
        #Since all other priors are scaled by 1, new denominator can be calculated
        #using the single updated prior
        norm = 1 / (1 - prior + self.probs[pos])
        self.probs *= norm

    def getCandidates(self, pos):
        maxProb = 0
        minDist = math.inf
        candidates = {}
        for index in np.ndindex(self.x, self.y):
            prob = self.probs[index]
            dist = manhattanDist(pos, index)
            if prob < maxProb or (prob == maxProb and dist > minDist):
                continue
            elif prob > maxProb or (prob == maxProb and dist < minDist):
                candidates = {index}
                minDist = dist
                maxProb = prob
            else:
                candidates.add(index)
        return candidates.pop()

    def basic1(self):
        pos = (np.random.randint(0, self.x), np.random.randint(0, self.y))
        count = 0
        while True:
            count += 1
            if self.query(pos):
                return count
            self.update(pos)
            maxProb = 0
            minDist = math.inf
            candidates = {}
            for index in np.ndindex(self.x, self.y):
                #Probability of containing target
                prob = self.probs[index]
                dist = manhattanDist(pos, index)
                if prob < maxProb or (prob == maxProb and dist > minDist):
                    continue
                elif prob > maxProb or (prob == maxProb and dist < minDist):
                    candidates = {index}
                    minDist = dist
                    maxProb = prob
                else:
                    candidates.add(index)
            next = candidates.pop()
            count += minDist
            pos = next

    def basic2(self):
        pos = (np.random.randint(0, self.x), np.random.randint(0, self.y))
        count = 0
        while True:
            count += 1
            if self.query(pos):
                return count
            self.update(pos)
            maxProb = 0
            minDist = math.inf
            candidates = {}
            for index in np.ndindex(self.x, self.y):
                #Probability of finding target
                prob = self.probs[index] * (1 - self.map[index])
                dist = manhattanDist(pos, index)
                if prob < maxProb or (prob == maxProb and dist > minDist):
                    continue
                elif prob > maxProb or (prob == maxProb and dist < minDist):
                    candidates = {index}
                    minDist = dist
                    maxProb = prob
                else:
                    candidates.add(index)
            next = candidates.pop()
            count += minDist
            pos = next

    def advancedStrat(self):
        pos = (np.random.randint(0, self.x), np.random.randint(0, self.y))
        count = 0
        while True:
            #After querying a position x-1 times, query it again with probability
            #self.probs[pos] ^ x
            x = 1.0
            while np.random.random() < x:
                count += 1
                if self.query(pos):
                    return count
                self.update(pos)
                x *= self.map[pos]
            maxProb = 0
            minDist = math.inf
            candidates = {}
            for index in np.ndindex(self.x, self.y):
                #Probability of finding target
                prob = self.probs[index]
                dist = manhattanDist(pos, index)
                if prob < maxProb or (prob == maxProb and dist > minDist):
                    continue
                elif prob > maxProb or (prob == maxProb and dist < minDist):
                    candidates = {index}
                    minDist = dist
                    maxProb = prob
                else:
                    candidates.add(index)
            next = candidates.pop()
            count += minDist
            pos = next

class SnD_movingtarget:
    def __init__(self, dim1, dim2 = None):
        self.x = dim1
        self.y = dim2 if dim2 != None else dim1

        #Create an x by y map and randomly assign terrain types
        self.map = np.random.choice([0.1, 0.3, 0.7, 0.9], (self.x, self.y,), replace = True)
        self.size = np.size(self.map)
        #Randomly select target
        self.target = (np.random.randint(0, self.x), np.random.randint(0, self.y))
        #Create an array of probabilities and initialize each to 1 / num of cells
        self.probs = np.full((self.x, self.y), 1 / self.size)

    def query(self, pos):
        if pos == self.target:
            #If guess is correct, return False with probability terrainDict[pos]
            if np.random.random() <= self.map[pos]:
                #Move target to a random neighboring cell
                self.target = (neighbors(self.target)).pop()
                return False
            return True
        else:
            #If guess is incorrect, return False
            return False

    def updateClose(self, pos):
        prior = self.probs[pos]
        self.probs[pos] *= self.map[pos]
        updated = np.zeros((self.x, self.y))
        a = pos[0]
        b = pos[1]
        for x in range(0, 6):
            for y in range(0, 6-x):
                if a-x > -1 and b-y > -1:
                    for neighbor in self.neighbors((a-x, b-y)):
                        updated[a-x][b-y] += self.probs[neighbor] / nOfN(neighbor, self.x, self.y)
                if a+x < self.x and b-y > -1:
                    for neighbor in self.neighbors((a+x, b-y)):
                        updated[a+x][b-y] += self.probs[neighbor] / nOfN(neighbor, self.x, self.y)
                if a-x > -1 and b+y < self.y:
                    for neighbor in self.neighbors((a-x, b+y)):
                        updated[a-x][b+y] += self.probs[neighbor] / nOfN(neighbor, self.x, self.y)
                if a+x < self.x and b+y < self.y:
                    for neighbor in self.neighbors((a+x, b+y)):
                        updated[a+x][b+y] += self.probs[neighbor] / nOfN(neighbor, self.x, self.y)
        #Calculate the pos seperately because the for loop miscalculates it
        updated[pos] = 0
        for neighbor in self.neighbors(pos):
            updated[pos] += self.probs[neighbor] / nOfN(neighbor, self.x, self.y)
        norm = 1 / np.sum(updated)
        updated *= norm
        self.probs = updated

    def updateFar(self, pos):
        pass

    def neighbors(self, pos):
        a = pos[0]
        b = pos[1]
        validNeighbors = set()
        possibleNeighbors = {(a, b + 1), (a + 1, b), (a, b - 1), (a - 1, b)}
        for cell in possibleNeighbors:
            a = cell[0]
            b = cell[1]
            if (0 <= a < self.x) and (0 <= b < self.y):
                validNeighbors.add(cell)
        return validNeighbors

    def getCandidates(self, pos):
        maxProb = 0
        minDist = math.inf
        candidates = {}
        for index in np.ndindex(self.x, self.y):
            prob = self.probs[index]
            dist = manhattanDist(pos, index)
            if prob < maxProb or (prob == maxProb and dist > minDist):
                continue
            elif prob > maxProb or (prob == maxProb and dist < minDist):
                candidates = {index}
                minDist = dist
                maxProb = prob
            else:
                candidates.add(index)
        return candidates.pop()

    def basic1(self):
        pos = (np.random.randint(0, self.x), np.random.randint(0, self.y))
        count = 0
        while True:
            count += 1
            if self.query(pos):
                return count
            self.update(pos)
            maxProb = 0
            minDist = math.inf
            candidates = {}
            for index in np.ndindex(self.x, self.y):
                #Probability of containing target
                prob = self.probs[index]
                dist = manhattanDist(pos, index)
                if prob < maxProb or (prob == maxProb and dist > minDist):
                    continue
                elif prob > maxProb or (prob == maxProb and dist < minDist):
                    candidates = {index}
                    minDist = dist
                    maxProb = prob
                else:
                    candidates.add(index)
            next = candidates.pop()
            count += minDist
            pos = next

    def basic2(self):
        pos = (np.random.randint(0, self.x), np.random.randint(0, self.y))
        count = 0
        while True:
            count += 1
            if self.query(pos):
                return count
            self.update(pos)
            maxProb = 0
            minDist = math.inf
            candidates = {}
            for index in np.ndindex(self.x, self.y):
                #Probability of finding target
                prob = self.probs[index] * (1 - self.map[index])
                dist = manhattanDist(pos, index)
                if prob < maxProb or (prob == maxProb and dist > minDist):
                    continue
                elif prob > maxProb or (prob == maxProb and dist < minDist):
                    candidates = {index}
                    minDist = dist
                    maxProb = prob
                else:
                    candidates.add(index)
            next = candidates.pop()
            count += minDist
            pos = next

    def advancedStrat(self):
        pos = (np.random.randint(0, self.x), np.random.randint(0, self.y))
        count = 0
        while True:
            #After querying a position x-1 times, query it again with probability
            #self.probs[pos] ^ x
            x = 1.0
            while np.random.random() < x:
                count += 1
                if self.query(pos):
                    return count
                self.update(pos)
                x *= self.map[pos]
            maxProb = 0
            minDist = math.inf
            candidates = {}
            for index in np.ndindex(self.x, self.y):
                #Probability of finding target
                prob = self.probs[index]
                dist = manhattanDist(pos, index)
                if prob < maxProb or (prob == maxProb and dist > minDist):
                    continue
                elif prob > maxProb or (prob == maxProb and dist < minDist):
                    candidates = {index}
                    minDist = dist
                    maxProb = prob
                else:
                    candidates.add(index)
            next = candidates.pop()
            count += minDist
            pos = next

a = 0
for i in range(500):
    s = SnD_stationarytarget(25,25)
    a += s.advancedStrat()
print(a/500)
