#Nodes
import numpy as np
class SumNode:
    def __init__(self, id):
        self.id = id;
        self.children = []
        self.parents = []
        self.weights = []
        self.rank = 0
        self.TRank = 0
        self.alive = 1
	self.flags = []
    def dead_children(self, thresh=0):
        a = []
	flags = [0]*len(self.weights)
	for f in self.flags:
		flags[f] = 1
        weights = np.array(self.weights)
        weights /= np.sum(weights)
        new_children = []
        new_weights = []
        print flags
        for c, nw, w in zip(self.children, flags, self.weights):
            if nw < 0.01:
                new_children.append(c)
                new_weights.append(w)
            else:
                a.append(c)
        self.weights = new_weights
        self.children = new_children
        return a

    def serialize_node(self):
        return self.id + ",SUM"

    def serialize_edges(self):
        return [self.id + "," + y + "," + str(x) for x, y in zip(self.weights, self.children)]

class PrdNode:
    def __init__(self, id):
        self.id = id
        self.children = []
        self.parents = []
        self.rank = 0
        self.TRank = 0
        self.alive = 1

    def dead_children(self, thresh=0):
        return []

    def serialize_node(self):
        return self.id + ",PRD"

    def serialize_edges(self):
        return [self.id + "," + x for x in self.children]

class Leaf:
    def __init__(self, id, a, b, i, t='c'):
        self.t = t
        self.id = id;
        self.inp = i;
        self.children = []
        self.parents = [];
        self.weights = [a, b];
        self.rank = 1;
        self.TRank = 0;
        self.alive = 1

    def dead_children(self, thresh=0):
        return []

    def serialize_node(self):
        return self.id + ",LEAVE," + str(self.inp) + "," + str(self.weights[1]) + "," + str(self.weights[0])
    
    def serialize_edges(self):
        return []


