#Model file
from load_file import *
from nodes import *
import random
import tensorflow as tf
import numpy as np
from e_load import *

# def findMaxesTensor(T, S):


class Model:

    def __init__(self, optimizer=tf.train.AdamOptimizer, file_weights=True):
        #filename: file to load the model from
        #optimizer: the SGD optimizer prefered
        #file_weights: A boolean representing whether to use weights from file or made up weights
        #currently doesn't allow you to build custom SPNs

        #Tensorflow Graph Variables
        self.input_swap = None
        self.summ = tf.placeholder(dtype=tf.string, shape=());
        self.loss_summary = None;
        self.writer = None;
        self.input = None #input tensor
        self.output = None #output tensor
        self.sparse_tensors = [] #all sparse tensors to be used in graph
        self.variables = [] #all variables
        self.layer_computations = [] #all mid computations in graph
        self.norm_value = 1.0
        self.counter_tensors = []
        self.weights = []
        self.num_weights = 0
        self.counting = []
        self.updates = []
        self.sum_of_weights = []
        self.labels = None
        self.norm_weights = []
        self.num = None
        self.step_size = 0.01
        #Nodes loaded from files
        self.out_size = 1
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order = None, None, None, None, None, None
        self.cont = [0, 0, 0]
        #For training and inference
        self.loss = None #loss function, None if uninitialized
        self.optimizer = optimizer
        self.optimizer2 = tf.train.AdamOptimizer
        self.opt_val = None
        self.session = None
        self.initalizer = tf.initialize_all_variables
        self.shuffle = None;
        self.reverse_shuffle = None;
        self.inds = None;
        self.multiclass = False;
        self.cccp_updates = []
        #Do Work

    def build_model_from_file(self, fname, random_weights, mem=False):
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order = load_file(fname, random_weights)

    def build_random_model(self, bfactor, input_length, output_size=1, ctype='b', multiclass=False):
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order, self.shuffle, self.inds = e_build_random_net(bfactor, input_length, output_size, ctype)
        self.multiclass=multiclass

    def build_fast_model(self, bfactor, input_length, ctype='b'):
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order, self.shuffle, self.inds = e_load(bfactor, input_length, ctype)
	self.num_weights = 0;
	for L in self.node_layers[1::2]:
		self.num_weights += len(L)
	self.num_weights += len(self.input_order)


    def compile(self):
        self.build_variables()
        self.build_forward_graph()
        self.start_session()
        self.writer = tf.train.SummaryWriter('logs/minibatch_range3', self.session.graph_def)
        self.get_normal_value()
        self.close_session()

    def build_input_vector(self, leafs):
        ws = []
        inds = []
        for i in range(len(leafs)):
            node = self.id_node_dict[str(leafs[i])]
            a = float(node.weights[0])
            b = float(node.weights[1])
            ws.append(a)
            ws.append(b)
            inds.append(i)
            inds.append(i)
        return tf.Variable(ws, dtype=tf.float64), inds

    def build_reverse_shuffle(self, shuffle):
        reverse_shuffle = []
        for s in shuffle:
            new_s = [0]*len(s)
            for i in range(len(s)):
                new_s[s[i]] = i
            reverse_shuffle.append(new_s)
        return reverse_shuffle

    def unbuild_fast_variables(self):
        weights = self.session.run(self.weights)
        for L in range(1, len(self.node_layers)):
            if isinstance(self.node_layers[L][0], SumNode):
                i = 0
                j = 0
                while i < len(self.node_layers[L]) and isinstance(self.node_layers[L][i], SumNode):
                    for p in range(len(self.node_layers[L][i].weights)):
                        self.id_node_dict[self.node_layers[L][i].id].weights[p] = weights[L][j+p]
                    j += len(self.node_layers[L][i].weights)
                    i += 1

        for i in range(len(self.leaf_id_order)):
            self.id_node_dict[self.leaf_id_order[i]].weights[0] = weights[0][i*2]
            self.id_node_dict[self.leaf_id_order[i]].weights[1] = weights[0][i*2+1]
    
    def save(self, fname):
        nodes = []
        edges = []
        for v in self.id_node_dict.values():
            if not v.alive: continue
            nodes.append(v.serialize_node())
            edges += v.serialize_edges()
        f = open(fname, 'w')
        f.write("####NODES####")
        for n in nodes:
            f.write("\n")
            f.write(n)
        f.write("\n")
        f.write("####EDGES####")
        for e in edges:
            f.write("\n")
            f.write(e)
        f.write("\n")
        f.close()
        
    def clean_nodes(self, thresh=0.05):
        for v in self.id_node_dict.values():
            if not v.alive: continue
            curr = v.dead_children(thresh)
            while len(curr) > 0:
                node = self.id_node_dict[curr.pop()]
                if not node.alive: continue
                node.alive = 0
                node.parents = []
                curr += node.children
                node.children = []

    def build_fast_variables(self):
        with tf.name_scope("Variables"):
            self.input_swap = tf.Variable(self.input_order)
            weights = []
            if self.node_layers[0][0].t != 'b':
                 self.cont[0] = tf.Variable([0.5 + random.random()*0.3 - 0.15 for x in self.leaf_id_order], dtype=tf.float64)
                 self.cont[1] = tf.Variable([1.0]*len(self.leaf_id_order), dtype=tf.float64)
                 self.cont[2] = tf.Variable([1.0]*len(self.leaf_id_order), dtype=tf.float64, trainable=False)
            input_weights, input_inds = self.build_input_vector(self.leaf_id_order)
            weights.append(input_weights)
            for L in range(1, len(self.node_layers)):
                ws = []
                if isinstance(self.node_layers[L][0], SumNode):
                    i = 0;
                    j = 0;
                    while i < len(self.node_layers[L]) and isinstance(self.node_layers[L][i], SumNode):
                        ws += map(lambda x: float(x), self.node_layers[L][i].weights);

                        j += len(self.node_layers[L][i].weights)
                        i += 1;
                weights.append(tf.Variable(ws, dtype=tf.float64))
            self.weights = weights;
            self.reverse_shuffle = self.build_reverse_shuffle(self.shuffle)
            self.inds = map(lambda x: tf.Variable(x), [input_inds]+self.inds)
            self.shuffle = map(lambda x: tf.Variable(x), self.shuffle)
            self.reverse_shuffle = map(lambda x: tf.Variable(x), self.reverse_shuffle)
        # self.input_layers = map(lambda x: [0] + tf.constant(x), self.input_layers)

    def get_norm_weights(self):
	weights = filter(lambda x: x != None, self.norm_weights);
	return self.session.run(weights);

    def normalize_weights(self):
        weights = filter(lambda x: x != None, self.norm_weights)
        new_weights = self.session.run(weights);
        old_weights = filter(lambda x: x.get_shape()[0] > 0, self.weights)
        for w, n in zip(old_weights, new_weights):
           # print w.get_shape()
           # print n.shape
            z = w.assign(n)
            self.session.run(z)

    def build_fast_forward_pass(self, step=0.003):
        computations = []
        bob = 1
        if self.node_layers[0][0].t == 'b':
            bob = 2
        print bob
        with tf.name_scope("input"):
            self.input = tf.placeholder(dtype=tf.float64, 
                                        shape=(None, max(self.input_order)+1, 
                                               bob),
                                        name='Input')
        # self.input = tf.placeholder(dtype=tf.float64, 
        #                                  shape=(len(self.input_order)*2), name='Input')
        #the input to be appended to each layer
        input_splits = []
        #compute the input
        weights = []
        with tf.name_scope("projection"):
            for L in range(len(self.weights)):
                n = tf.constant(0.0001, dtype=tf.float64)
                weights.append(tf.add(tf.nn.relu(tf.sub(self.weights[L], n)), n))

        with tf.name_scope('nomralization'):
            # print (map(lambda x: x.get_shape(), self.inds))
            self.sum_of_weights = [tf.segment_sum(x, y) if x.get_shape()[0] > 0 else None for x, y in zip(weights, self.inds)]
            sum_of_weights = self.sum_of_weights
            self.norm_weights = [tf.div(x, tf.gather(y, z)) if x.get_shape()[0] > 0 else None for x, y, z in zip(weights, self.sum_of_weights, self.inds)]
        # print sums_of_sparses
        
        with tf.name_scope('LEAFS_' + str(len(self.input_order))):
            print len(self.input_order)
            input_gather = tf.reshape(tf.transpose(tf.gather(tf.transpose(self.input, (1, 0, 2)), self.input_swap), (1, 0, 2)), shape=(-1, len(self.input_order)*bob))
            self.counting.append(input_gather)
            # input_gather = tf.Print(input_gather, [input_gather], first_n=25, summarize=25)
            if self.node_layers[0][0].t == 'b':
               input_computation_w = tf.mul(input_gather, weights[0])
               input_computation_s = tf.transpose(tf.segment_sum(tf.transpose(input_computation_w), self.inds[0]))
               input_computation_n = tf.log(tf.div(input_computation_s, sum_of_weights[0]))
               computations.append(input_computation_n)
            else:
               pi = tf.constant(np.pi, tf.float64)
               mus = self.cont[0]
               sigs = tf.nn.relu(self.cont[1] - 0.1) + 0.1
               input_computation_g = tf.div(tf.exp(tf.neg(tf.div(tf.square(input_gather - mus), 2*tf.mul(sigs, sigs)))), tf.sqrt(2*pi)*sigs)
               input_computation_n = tf.log(input_computation_g)
               computations.append(input_computation_n)

        
        #split the input computation and figure out which one goes in each layer
            j = 0
            for i in range(len(self.input_layers)):
                a = tf.constant(j)
                b = self.input_layers[i]
                input_splits.append(tf.slice(input_computation_n, [0, a], [-1, b]))
                # input_splits.append(tf.slice(input_computation_n, [a], [b]))
                j += b;
                # print "size"
                # print size

        current_computation = input_splits[0]

        for i in range(len(self.node_layers[1:])):
            L = i+1 #the layer number
            # print current_computation.get_shape()
            # print self.shuffle[i]
            # print self.inds[L].get_shape()
            # print "yoo"
            
            if self.weights[L].get_shape()[0] == 0: #product
                with tf.name_scope("PRD" + str(self.inds[L].get_shape()[0])):
                    current_computation = tf.transpose(tf.segment_sum(tf.transpose(current_computation), self.inds[L]))
               
            else:
                with tf.name_scope("SUM" + str(self.inds[L].get_shape()[0])):
                    self.counting.append(current_computation)
                    maxes = tf.transpose(tf.segment_max(tf.transpose(current_computation), self.inds[L]))
                    back_maxes = tf.transpose(tf.gather(tf.transpose(maxes), self.inds[L]))
                    current_computation = tf.sub(current_computation, back_maxes)
                    current_computation = tf.exp(current_computation)
                    current_computation = tf.mul(current_computation, weights[L])
                    current_computation = tf.transpose(tf.segment_sum(tf.transpose(current_computation), self.inds[L]))
                    current_computation = tf.div(current_computation, sum_of_weights[L])
                    current_computation = tf.add(tf.log(current_computation), maxes)
                    current_computation = tf.concat(1, [current_computation, input_splits[L]])
                    current_computation = tf.transpose(tf.gather(tf.transpose(current_computation), self.shuffle[L]))
               
            computations.append(current_computation)
        with tf.name_scope('root_node'):
            self.output = current_computation
        with tf.name_scope('loss'):
            if self.multiclass:
                self.labels = tf.placeholder(shape=(None, len(self.node_layers[-1])), dtype=tf.float64)
                self.loss = -tf.reduce_mean(tf.mul(self.output, 0.1*(self.labels-1)+self.labels))
            else:
                self.loss = -tf.reduce_mean(self.output)
            self.loss_summary = tf.scalar_summary(self.summ, self.loss)
        self.opt_val = self.optimizer(0.001).minimize(self.loss)
#        self.opt_val2 = self.optimizer(0.001).minimize(self.loss)
        self.computations = computations

    def cccp(self):
        with tf.name_scope("CCCP"):
            updates =  []
            splits = []
            self.num = tf.placeholder(shape=(None, len(self.node_layers[-1])), dtype=tf.float64)
            curr = self.num
            for i in reversed(range(len(self.node_layers[1:]))):
                L = i+1
                if self.weights[L].get_shape()[0] == 0: #product node
                    curr = tf.transpose(tf.gather(tf.transpose(curr), self.inds[L], name="myprodgather"))
                    factored_comp = tf.transpose(tf.gather(tf.transpose(self.computations[L]), self.inds[L]))
                    curr = curr + factored_comp - self.computations[i][:, :curr.get_shape()[1]]
                  #  print "prd"
                else: #sum node
                  #  print "sum"
                    curr = tf.transpose(tf.gather(tf.transpose(curr), self.reverse_shuffle[L]))
                    if (self.input_layers[L] > 0):
                        curr, split = curr[:, :-self.input_layers[L]], curr[:, -self.input_layers[L]:]
                        splits = [split] + splits;
                    curr = tf.transpose(tf.gather(tf.transpose(curr), self.inds[L], name="mysumgather"))
                    update = self.weights[L]*tf.exp(curr - self.computations[-1] + self.computations[i])
                    curr = curr + tf.log(self.weights[L])
                    updates.append(tf.reduce_sum(update, reduction_indices=0))
            inputs = tf.concat(1, [curr] + splits, name="lolface");
            if self.node_layers[0][0].t == 'b':
                gathered = tf.transpose(tf.gather(tf.transpose(inputs), self.inds[0]))
                update = self.weights[0]*tf.exp(gathered - self.computations[-1])*self.counting[0]+0.000001
                updates.append(tf.reduce_sum(update, reduction_indices=0))
            self.cccp_updates = updates

    def apply_cccp(self, feed_dict, compute_size=100):
        data = feed_dict[self.input]
        updates = []
        for i in range(1+(len(data)-1)//compute_size):
            feed_dict[self.input] = data[i*compute_size:min((i+1)*compute_size, len(data))]
            feed_dict[self.num] = feed_dict[self.num][:-i*compute_size+min((i+1)*compute_size, len(data))]
            update = self.session.run(self.cccp_updates, feed_dict=feed_dict)
            if len(updates) == 0:
                updates = update
            else:
                for i in range(len(update)):
                    updates[i] += update[i]
        # print updates
        a=0
        weights = filter(lambda x: x.get_shape()[0] > 0, self.weights)
        updates = list(reversed(updates))
        for i in range(len(weights)):
            if self.node_layers[0][0].t == 'c' and i == 0:
                for j in range(3):
                    z = tf.assign(self.cont[-j-1], updates[i+j])
                    self.session.run(z)
                a = 2
                continue
            u, w = updates[i+a], weights[i]
            z = tf.assign(w, u)
            self.session.run(z)
        self.normalize_weights()       

    def countss(self):
        with tf.name_scope("Counting"):
            maxed_out = []
            val = tf.constant(0.51, dtype=tf.float64)
            t = lambda x: tf.transpose(x)
            for c in range(len(self.counting)):
                if c == 0 and self.node_layers[0][0].t == 'c':
                    counts = self.counting[c]*0+1
                else:
                    maxes = tf.mul(tf.transpose(tf.segment_max(tf.transpose(self.counting[c]), self.inds[c*2])), val)
                    back_maxes = tf.transpose(tf.gather(tf.transpose(maxes), self.inds[c*2]))
                    counts = tf.nn.relu(tf.round(tf.div(back_maxes, self.counting[c])))
                maxed_out.append(counts)
            updates = []
            splits = []
            self.num2 = tf.placeholder(shape=(None, len(self.node_layers[-1])), dtype=tf.float64)
            curr = self.num2
            for i in reversed(range(len(self.node_layers[1:]))):
                L = i+1
                if self.weights[L].get_shape()[0] == 0: #product node
                    curr = tf.transpose(tf.gather(tf.transpose(curr), self.inds[L], name="myprodgather"))
                  #  print "prd"
                else: #sum node
                  #  print "sum"
                    curr = tf.transpose(tf.gather(tf.transpose(curr), self.reverse_shuffle[L]))
                    if (self.input_layers[L] > 0):
                        curr, split = curr[:, :-self.input_layers[L]], curr[:, -self.input_layers[L]:]
                        splits = [split] + splits;
                    curr = tf.transpose(tf.gather(tf.transpose(curr), self.inds[L], name="mysumgather"))
                    curr = tf.mul(curr, maxed_out[L//2])
                    updates.append(tf.reduce_sum(curr, reduction_indices=0))
            inputs = tf.concat(1, [curr] + splits, name="lolface");
            if self.node_layers[0][0].t == 'b':
                gathered = tf.transpose(tf.gather(tf.transpose(inputs), self.inds[0]))
                updates.append(tf.reduce_sum(tf.mul(gathered, self.counting[0]), reduction_indices=0))
            else:
                important_inputs = tf.reduce_sum(inputs*self.counting[0], reduction_indices=0)
                total = self.cont[2]+tf.reduce_sum(inputs, reduction_indices=0)
                new_mu = (important_inputs + self.cont[0]*self.cont[2])/total
                mu_diff = new_mu - self.cont[0]
                other_diff = self.counting[0] - self.cont[0]
                new_sig = tf.sqrt((self.cont[1]*self.cont[1]*self.cont[2] + tf.reduce_sum(other_diff*other_diff*inputs, reduction_indices=0))/total - mu_diff*mu_diff)
                new_sums = total
                
                updates.append(new_mu)
                updates.append(new_sig)
                updates.append(new_sums)

        self.updates = updates;

    def apply_count(self, feed_dict, compute_size=100, c=1):
        data = feed_dict[self.input]
        updates = []
        for i in range(1+(len(data)-1)//compute_size):
            feed_dict[self.input] = data[i*compute_size:min((i+1)*compute_size, len(data))]
            feed_dict[self.num2] = feed_dict[self.num2][:-i*compute_size+min((i+1)*compute_size, len(data))]
            update = self.session.run(self.updates, feed_dict=feed_dict)
            if len(updates) == 0:
                updates = update
            else:
                for i in range(len(update)):
                    updates[i] += update[i]
        # print updates
        a=0
        weights = filter(lambda x: x.get_shape()[0] > 0, self.weights)
        print [np.sum(x) for x in weights]
        updates = list(reversed(updates))
        for i in range(len(weights)):
            if self.node_layers[0][0].t == 'c' and i == 0:
                for j in range(3):
                    z = tf.assign(self.cont[-j-1], updates[i+j])
                    self.session.run(z)
                a = 2
                continue
            u, w = updates[i+a], weights[i]
            z = tf.assign_add(w, u)
            self.session.run(z)      
 # print updates

    def fast_compile(self, step, tensorboard_dir="logs/"):
        self.build_fast_variables()
        self.build_fast_forward_pass(step)
        self.countss()
        self.cccp()
        self.out_size = len(self.node_layers[-1])
        self.start_session()
        if tensorboard_dir:
            self.writer = tf.train.SummaryWriter(tensorboard_dir, self.session.graph)
        self.close_session()
    def start_session(self):
        assert self.session == None
	config = tf.ConfigProto()
	config.gpu_options.allocator_type = 'BFC'
	config.gpu_options.allow_growth=True
        self.session = tf.Session(config=config)
        self.session.run(self.initalizer())
        return self.session

    def close_session(self):
        assert self.session != None

        self.session.close()
        self.session = None
        return None

    def build_input_matrix(self, leafs):
        inds = []
        ws = []
        s = []
        for i in range(len(leafs)):
            node = self.id_node_dict[str(leafs[i])]
            a = float(node.weights[0])
            b = float(node.weights[1])
            inds.append([i, i*2])
            inds.append([i, i*2+1])
            ws.append(a)
            ws.append(b)
        s = [len(leafs), len(leafs)*2]
        # print inds
        return tf.Variable(ws, dtype=tf.float64), tf.constant(s, dtype=tf.int64), tf.constant(inds, dtype=tf.int64)


    def build_variables(self):
        #Build Variables
        #Input Placeholder
        with tf.name_scope('inputs'):
            self.input = tf.placeholder(dtype=tf.float64, 
                                         shape=(len(self.input_order)*2, 
                                         None), name='Input')
            #Input matrix
            input_weights, input_shape, input_indices = self.build_input_matrix(self.leaf_id_order)
            input_counter = tf.constant([1.0]*len(self.input_order)*2, dtype=tf.float64)
            input_counter_matrix = tf.SparseTensor(input_indices, input_counter, input_shape)
            self.counter_tensors.append(input_counter_matrix)
            input_matrix = tf.SparseTensor(input_indices, tf.add(tf.nn.relu(tf.identity(input_weights)), 0.001), input_shape)

        #Layer Matrices
        layer_matrices = []
        variables = []
        L = 1
        for node_layer in self.node_layers[1:]:
            indices = []
            weights = []
            shape = []
            for i in range(len(node_layer)-self.input_layers[L]):
                for j in range(len(node_layer[i].children)):
                    #get the layer position of the child node
                    a, b = self.pos_dict[node_layer[i].children[j]]
                    indices.append([i, b])
                    if isinstance(node_layer[i], SumNode): 
                        #Sum Node
                        weights.append(float(node_layer[i].weights[j]))
                    else:
                        #Product Node
                        weights.append(1.0)
            if isinstance(node_layer[0], SumNode):
                trainable = True
                name = 'SUM_VARS' + str(L)
            else:
                trainable = False
                name = 'PROD_VARS' + str(L)
            print weights
            winds = zip(weights, indices);
            winds.sort(lambda x, y: -1 if x[1][1] < y[1][1] else 1);
            weights, indices = zip(*winds);
            print indices;
            shape = [len(node_layer)-self.input_layers[L], len(self.node_layers[L-1])]
            with tf.name_scope(name):
                W = tf.Variable(weights, trainable=trainable, dtype=tf.float64)
                I = tf.constant(indices, dtype=tf.int64)
                S = tf.constant(shape, dtype=tf.int64)
                C = tf.constant([1.0]*len(weights), dtype=tf.float64)
                # print shape
                L += 1
                matrix = tf.SparseTensor(I, tf.nn.relu(tf.identity(W)), S)
                counter_matrix = tf.SparseTensor(I, C, S);
            variables.append((W, I, S, shape))
            self.counter_tensors.append(counter_matrix)
            layer_matrices.append(matrix)
        self.sparse_tensors = [input_matrix] + layer_matrices
        self.variables = [(input_weights, input_indices, input_shape, [len(self.input_order), len(self.input_order)*2])] + variables

    def build_forward_graph(self):
        computations = []

        #the input to be appended to each layer
        input_splits = []
        #compute the input
        with tf.name_scope('NORM_FACTOR'):
            sums_of_sparses = [tf.reshape(tf.sparse_reduce_sum(sm, reduction_axes=1), [-1 ,1]) for sm in self.sparse_tensors]
        # print sums_of_sparses
        with tf.name_scope('LEAFS_' + str(len(self.input_order))):
            input_computation = tf.div(tf.sparse_tensor_dense_matmul(self.sparse_tensors[0], self.input), sums_of_sparses[0])
            computations.append(input_computation)

        #split the input computation and figure out which one goes in each layer
            i = 0
            for size in self.input_layers:
                input_splits.append(input_computation[i:i+size])
                i += size

        current_computation = input_splits[0]
        L = 1
        for i in range(len(self.node_layers[1:])):
            node_layer = self.node_layers[i+1]
            matrix = self.sparse_tensors[i+1]
            if isinstance(node_layer[0], SumNode):
                with tf.name_scope('SUM_' + str(self.variables[i+1][3][0])):
                    current_computation = tf.concat(0,
                                          [tf.div(tf.sparse_tensor_dense_matmul(matrix, current_computation, name='ComputeSum'), sums_of_sparses[i+1], name='Normalize'), 
                                                                                      input_splits[L]], name='ConcatenateInputs')
            else:
                with tf.name_scope('PROD_' + str(self.variables[i+1][3][0])):
                    current_computation = tf.exp(tf.sparse_tensor_dense_matmul(matrix, tf.log(current_computation, name='ToLogDomain'), name='ComputeProd'), name='ToNormalDomain')
            L += 1;
            computations.append(current_computation)

        self.output = current_computation
        with tf.name_scope('loss'):
            self.loss = -tf.reduce_mean(tf.log(self.output), reduction_indices=1)
            self.loss_summary = tf.scalar_summary(self.summ, self.loss)
        # self.opt_val = self.optimizer(0.01).minimize(self.loss)
        # self.opt_val2 = self.optimizer2(0.01).minimize(self.loss)
        self.computations = computations
    # def counting_step(self) :


    def get_normal_value(self):
        ones = [[1]*len(self.input_order)*2]
        norm_value = self.output.eval(session=self.session, feed_dict = {self.input: ones})
        self.norm_value = norm_value
        return norm_value


