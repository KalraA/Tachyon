#loading a graph from a file

from tachyon.nodes import *
import random
import numpy as np
r = lambda x: 0.1 + random.random()*0.1
def generate_children(nodes, curr_node, scope, bf, depth=0, max_depth=10, ctype='c'):
 
    curr_id = int(curr_node.id) + 1
    if isinstance(curr_node, PrdNode):
        num_children = random.randint(bf[0][0], bf[0][1]) if depth > max_depth - 2 else len(scope)
        scopes = [[] for x in range(num_children)]
        weights = np.array([1.0]*num_children)/num_children
        for s in scope:
            slot = np.random.choice(num_children, 1, p=weights)[0]
            val = weights[slot]/4
            weights += (weights[slot] - val)/(len(weights)-1)
            weights[slot] = val
            scopes[slot].append(s)
        scopes = filter(lambda x: x != [], scopes)
	if depth > max_depth - 2:
		scopes = map(lambda x: [x], scope)
        for c in xrange(len(scopes)):
            if len(scopes[c]) == 1:
                a = random.random()
                child = Leaf(str(curr_id+1), a, 1-a, scopes[c][0], t=ctype)
                curr_node.children.append(str(curr_id + 1))
                child.parents.append(curr_node.id)
                curr_id += 1
                nodes.append(child)
            else:
                child = SumNode(str(curr_id + 1))
                child.parents.append(curr_node.id)
                curr_node.children.append(str(curr_id + 1))
                nodes, curr_id = generate_children(nodes, child, scopes[c], bf, depth+1, max_depth, ctype=ctype)
        nodes.append(curr_node)
        return nodes, curr_id+1
    else:
        num_children = random.randint(bf[1][0], bf[1][1])
        scopes = [scope]*num_children
        #print scopes
        for c in xrange(len(scopes)):
            child = PrdNode(str(curr_id + 1))
            curr_node.children.append(str(curr_id + 1))
            curr_node.weights.append(random.random() - 0.5)
            child.parents.append(curr_node.id)
            nodes, curr_id = generate_children(nodes, child, scopes[c], bf, depth+1, max_depth, ctype=ctype)
        norm_factor = sum(curr_node.weights)
        curr_node.weights = map(lambda x: x/norm_factor, curr_node.weights)
        nodes.append(curr_node)
        return nodes, curr_id+1
        

def split_up_file(fname):
    infile = open(fname, 'r')
    t = 'placeholder'
    lines = []
    while t != '':
        t = infile.readline()
        lines.append(t[:-1])
    n = 0
    for i in range(len(lines)):
        if "EDGES" in lines[i] or "Edges" in lines[i]:
            n = i;
            break;

    nodes = lines[1:n]
    edges = lines[n+1:]
    return nodes, edges

def build_nodes(nodes, random_weights=False, ctype="b"):
    big_dict = {}
    Leaves = []
    Prods = []
    Sums = []
    for l in nodes:
        if 'PRD' in l:
            arr = l.split(',')
            node = PrdNode(arr[0])
            big_dict[arr[0]] = node
            Prods.append(arr[0])
        elif 'SUM' in l:
            arr = l.split(',')
            node = SumNode(arr[0])
            big_dict[arr[0]] = node
            Sums.append(arr[0])
        elif 'LEAVE' in l or 'BINN' in l:
            arr = l.split(',')
            if random_weights:
                arr[3] = r(1)
                arr[4] = r(1)
            node = Leaf(arr[0], arr[4], arr[3], arr[2], t=ctype)
            big_dict[arr[0]] = node
            Leaves.append(arr[0])
    return Leaves, Prods, Sums, big_dict


def add_connections(id_node_dict, edges, random_weights=False):
    for edge in edges:
        a = edge.split(',')
        if a[0] == '' or a[1] == '':
            continue
        id_node_dict[a[0]].children.append(a[1])
        id_node_dict[a[1]].parents.append(a[0])
        if len(a) == 3:
            if random_weights:
                id_node_dict[a[0]].weights.append(r(5))
            else:
                id_node_dict[a[0]].weights.append(a[2])
    return id_node_dict

def add_ranks(id_node_dict, leaf_id):
    currs = set(leaf_id)
    rank = 1
    while len(currs) > 0:
        prev_currs = currs
        new_currs = set()
        for s in list(currs):
            for p in id_node_dict[s].parents:
                new_currs.add(p)
            id_node_dict[s].rank = rank
        currs = new_currs
        rank += 1
    orank = rank
    rank -= 1
    currs = prev_currs
    while len(currs) > 0:
        new_currs = set()
        #print rank
        #print currs
        for s in list(currs):
            for p in id_node_dict[s].children:
                new_currs.add(p)
            id_node_dict[s].TRank = rank
        currs = new_currs
        rank -= 1
    return orank, id_node_dict

def create_layers(id_node_dict, rank):
    node_list = [[] for x in range(rank)]
    for k in id_node_dict.keys():
        n = id_node_dict[k]
        #print n.TRank
        node_list[n.TRank].append(n)
    return node_list[1:]

def make_pos_dict(node_layers):
    new_dict = {}
    for i in range(len(node_layers)):
        node_layers[i].sort(lambda x, y: 1 if isinstance(x, Leaf) else -1)
        for j in range(len(node_layers[i])):
            new_dict[node_layers[i][j].id] = (i, j)
    return new_dict, node_layers

def clean_up_inputs(node_layers):
    input_layers = []
    input_order = []
    leaf_order = []
    for n_lst in node_layers:
        c = 0
        for n in n_lst:
            if isinstance(n, Leaf):
                c += 1
                input_order.append(int(n.inp))
                leaf_order.append(n.id)
        input_layers.append(c)
    return leaf_order, input_layers, input_order

def format_list_of_nodes(lst):
    id_node_dict = {}
    leafs = []
    sums = []
    prds = []
    for n in lst:
        if isinstance(n, SumNode):
            sums.append(n.id)
        elif isinstance(n, PrdNode):
            prds.append(n.id)
        else:
            leafs.append(n.id)
        assert id_node_dict.get(n.id, None) == None
        id_node_dict[n.id] = n
    return leafs, prds, sums, id_node_dict

def build_random_net(bf, inp_size, out_size):
    count = 0
    total_net = []
    for i in range(out_size):
        init_node = SumNode(str(count))# if random.random() < 0.51 else PrdNode('0')
        network, count = generate_children([], init_node, range(inp_size), bf)
        total_net += network
    network = total_net
    leaf_ids, prod_ids, sum_ids, id_node_dict = format_list_of_nodes(network)
    print len(id_node_dict)
    #determine all the ranks for each node
    rank, id_node_dict = add_ranks(id_node_dict, leaf_ids)
    #turn them all into layers
    node_layers = create_layers(id_node_dict, rank)
    # print map(lambda x: len(x), node_layers)
    #create a dict for the position of every node given the ids
    pos_dict, node_layers = make_pos_dict(node_layers)
    for n in node_layers:
        print len(n)
    print len(leaf_ids)
    #getting the ordering right
    leaf_id_order, input_layers, input_orders = clean_up_inputs(node_layers)
    print leaf_id_order

    return pos_dict, id_node_dict, node_layers, leaf_id_order, input_layers, input_orders


def load_file(fname, random_weights=False):
    #get the node and edge strings from a file
    file_nodes, file_edges = split_up_file(fname)
    #get all the different nodes and a dict that matches id to node
    leaf_ids, prod_ids, sum_ids, id_node_dict = build_nodes(file_nodes)
    #add all the edges to the nodes
    id_node_dict = add_connections(id_node_dict, file_edges, random_weights)
    if random_weightsi and False:
        for id in sum_ids:
            summ = sum(id_node_dict[id].weights)
            id_node_dict[id].weights = map(lambda x: x/summ, id_node_dict[id].weights)
    #determine all the ranks for each node
    rank, id_node_dict = add_ranks(id_node_dict, leaf_ids)
    #turn them all into layers
    node_layers = create_layers(id_node_dict, rank)
    # print map(lambda x: len(x), node_layers)
    #create a dict for the position of every node given the ids
    pos_dict, node_layers = make_pos_dict(node_layers)
    #getting the ordering right
    leaf_id_order, input_layers, input_orders = clean_up_inputs(node_layers)

    return pos_dict, id_node_dict, node_layers, leaf_id_order, input_layers, input_orders
