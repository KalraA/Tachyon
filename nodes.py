#Nodes

class SumNode:
    def __init__(self, id):
        self.id = id;
        self.children = []
        self.parents = []
        self.weights = []
        self.rank = 0
        self.TRank = 0
        
class PrdNode:
    def __init__(self, id):
        self.id = id
        self.children = []
        self.parents = []
        self.rank = 0
        self.TRank = 0
        
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
