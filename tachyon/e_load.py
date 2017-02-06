from tachyon.load_file import *
import copy


def e_add_ranks(id_node_dict, leaf_id):
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
    top = None;
    currs = prev_currs
    top = list(currs);
    while len(currs) > 0:
        new_currs = set()
        for s in list(currs):
            for p in id_node_dict[s].children:
                new_currs.add(p)
            id_node_dict[s].TRank = rank
        currs = new_currs
        rank -= 1
    return top, orank, id_node_dict


def e_create_layers(id_node_dict, rank, curr):
    node_list = [[] for x in range(rank)]
    for k in id_node_dict.keys():
        n = id_node_dict[k]
        # print n.TRank
        node_list[n.TRank].append(n)
    node_proper = [[] for x in range(rank)]
    inds_proper = [[] for x in range(rank)]
    inds = range(len(curr))
    curr = [id_node_dict[c] for c in curr]
    while(curr and inds):
        next_curr = []
        next_inds = []
        done = set()
        i = 0;
        for n, ind in zip(curr, inds):
            node_proper[n.TRank].append(n)
            inds_proper[n.TRank].append(ind)
            for c in n.children:
                if c not in done:
                    done.add(c)
                    next_curr.append(id_node_dict[c])
                    next_inds.append(i)
            if (n.children):
            	i += 1;
        curr = next_curr#list(set(next_curr));
        inds = next_inds;

    return node_proper[1:], inds_proper[1:]

def e_finish_layers(layers):
	id_map = lambda z: {y.id: i for i, y in enumerate(z)}
	new_layers = [sorted(copy.deepcopy(lst), (lambda x, y: -1 if isinstance(x, SumNode) and isinstance(y, Leaf) else 1)) for lst in layers]
	layer_ids = [id_map(x) for x in new_layers];
	# print reduce(lambda r, f: r+f.children, new_layers[4], [])
	switches = [[layer_ids[x[0]].get(y.id) for y in x[1]] for x in zip(range(len(layers)), layers)]
	return new_layers, switches

def e_make_pos_dict(node_layers):
    new_dict = {}
    for i in range(len(node_layers)):
        for j in range(len(node_layers[i])):
            new_dict[node_layers[i][j].id] = (i, j)
    return new_dict, node_layers

def e_load(fname, random_weights=False, ctype="b"):
    #get the node and edge strings from a file
    file_nodes, file_edges = split_up_file(fname)
    #get all the different nodes and a dict that matches id to node
    leaf_ids, prod_ids, sum_ids, id_node_dict = build_nodes(file_nodes, random_weights, ctype)
    #add all the edges to the nodes
    id_node_dict = add_connections(id_node_dict, file_edges, random_weights)
    if random_weights:
        for id in sum_ids:
            summ = sum(id_node_dict[id].weights)
            id_node_dict[id].weights = map(lambda x: x/summ, id_node_dict[id].weights)
    #determine all the ranks for each node
    top ,rank, id_node_dict = e_add_ranks(id_node_dict, leaf_ids)
    #turn them all into layers
    shuffle_layers, inds = e_create_layers(id_node_dict, rank, top)
    node_layers, shuffle_layers = e_finish_layers(shuffle_layers)
    #create a dict for the position of every node given the ids
    pos_dict, node_layers = e_make_pos_dict(node_layers)
    #getting the ordering right
    leaf_id_order, input_layers, input_orders = clean_up_inputs(node_layers)

    return pos_dict, id_node_dict, node_layers, leaf_id_order, input_layers, input_orders, shuffle_layers, inds

def e_build_random_net(bf, inp_size, out, ctype='b', depth=6):
    count = 0
    total_net = []
    init_node = SumNode(str(count + 1))
    network, count = generate_children([], init_node, range(inp_size), bf, ctype=ctype, max_depth=depth)
    extras = []
    for i in range(1, out):
        node = SumNode(str(count + i))
        node.children = copy.copy(init_node.children)
        node.weights = [x + random.random()*0.3 - 0.15 for x in copy.copy(init_node.weights)]
        extras.append(node)
        network.append(node)
    leaf_ids, prod_ids, sum_ids, id_node_dict = format_list_of_nodes(network)
    for e in extras:
        for c in e.children:
             id_node_dict[c].parents.append(e.id)
    #determine all the ranks for each node
    rank, id_node_dict = add_ranks(id_node_dict, leaf_ids)
    #turn them all into layers
    top ,rank, id_node_dict = e_add_ranks(id_node_dict, leaf_ids)
    #turn them all into layers
    shuffle_layers, inds = e_create_layers(id_node_dict, rank, top)
    node_layers, shuffle_layers = e_finish_layers(shuffle_layers)
    #create a dict for the position of every node given the ids
    pos_dict, node_layers = e_make_pos_dict(node_layers)

    shuffle_layers[-2] *= out
    temp = []
    for i in xrange(out):
        temp += [x + i for x in inds[-2]]
    inds[-2] = temp

    #getting the ordering right
    leaf_id_order, input_layers, input_orders = clean_up_inputs(node_layers)

    print "generated a network of size", len(id_node_dict), "with", len(node_layers) ,"layers."

    return pos_dict, id_node_dict, node_layers, leaf_id_order, input_layers, input_orders, shuffle_layers, inds
