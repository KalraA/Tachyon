#Model file
from tachyon.load_file import *
from tachyon.nodes import *
import random
import tensorflow as tf
import numpy as np
from tachyon.e_load import *

# def findMaxesTensor(T, S):


class Model:

    def __init__(self, optimizer=tf.train.AdamOptimizer, file_weights=True):
        #filename: file to load the model from
        #optimizer: the SGD optimizer prefered
        #file_weights: A boolean representing whether to use weights from file or made up weights
        #currently doesn't allow you to build custom SPNs
        self.binary = True
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
        self.counting = [] #variables for counting updates
        self.updates = [] #updates to be made for counting
        self.sum_of_weights = [] #normalizing factors for weights
        self.labels = None 
        self.norm_weights = [] #normalized weights
        self.num = None
        self.step_size = 0.01
        #Nodes loaded from files
        self.out_size = 1
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order = None, None, None, None, None, None
        self.cont = [0, 0, 0] #mean, variance, count - only matters for contiuous variables
        self.rand_weights = True
        #For training and inference
        self.loss = None #loss function, None if uninitialized
        self.optimizer = optimizer
        self.opt_val = None #actual value to call from session.run
        self.session = None
        self.initalizer = tf.initialize_all_variables
        self.shuffle = None; #shuffle done between each layer to match nodes
        self.reverse_shuffle = None; #unshuffling them for backward pass
        self.inds = None; #indicies for segmented sum 
        self.multiclass = False; 
        self.cccp_updates = [] 
        self.means = self.vars = [] #mean and variance of the data
        #Do Work

    def set_mv(self, data):
        self.mean = np.mean(data, axis=1)
        self.vars = np.var(data, axis=1)

    def build_model_from_file(self, fname, random_weights, mem=False):
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order = load_file(fname, random_weights)

    def build_random_model(self, bfactor, input_length, output_size=1, ctype='b', multiclass=False, depth=6):
        if ctype=='c':
            self.binary = False
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order, self.shuffle, self.inds = e_build_random_net(bfactor, input_length, output_size, ctype, depth)
        self.multiclass=multiclass
        self.mean = np.array([0]*input_length)
        self.vars = np.array([1]*input_length)

    def build_fast_model(self, fname, random_weights, ctype='b'):
        if ctype=='c':
            self.binary = False 
        self.rand_weights = random_weights        
        self.pos_dict, self.id_node_dict, self.node_layers, self.leaf_id_order, self.input_layers, self.input_order, self.shuffle, self.inds = e_load(fname, random_weights, ctype)
	self.num_weights = 0;
	for L in self.node_layers[1::2]:
		self.num_weights += len(L)
	self.num_weights += len(self.input_order)
        input_length = max(self.input_order)+1
        self.mean = np.array([0]*input_length)
        self.vars = np.array([1]*input_length)


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
        if not self.binary:
            cont = self.session.run(self.cont)
        for L in range(1, len(self.node_layers)):
            if isinstance(self.node_layers[L][0], SumNode):
                i = 0
                j = 0
                while i < len(self.node_layers[L]) and isinstance(self.node_layers[L][i], SumNode):
                    for p in range(len(self.node_layers[L][i].weights)):
                        if self.id_node_dict[self.node_layers[L][i].id].weights[p] == weights[L][j+p]:
			    self.id_node_dict[self.node_layers[L][i].id].flags.append(p)
			self.id_node_dict[self.node_layers[L][i].id].weights[p] = weights[L][j+p]
                    j += len(self.node_layers[L][i].weights)
                    i += 1

        for i in range(len(self.leaf_id_order)):
            if self.binary:
                self.id_node_dict[self.leaf_id_order[i]].weights[0] = weights[0][i*2]
                self.id_node_dict[self.leaf_id_order[i]].weights[1] = weights[0][i*2+1]
            else:
                self.id_node_dict[self.leaf_id_order[i]].weights[0] = cont[1][i]
                self.id_node_dict[self.leaf_id_order[i]].weights[1] = cont[0][i]
    
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
                if self.rand_weights:
                    self.cont[0] = tf.Variable([self.mean[x] + (lambda y: random.random()*y-(y*0.5))(np.sqrt(self.vars[x])) for x in self.input_order], dtype=tf.float64)
                    self.cont[1] = tf.Variable([self.vars[x] for x in self.input_order], dtype=tf.float64)
                    self.cont[2] = tf.Variable([1.0]*len(self.leaf_id_order), dtype=tf.float64, trainable=False)
                else:
                    self.cont[0] = tf.Variable([float(self.id_node_dict[x].weights[1]) for x in self.leaf_id_order], dtype=tf.float64)
                    self.cont[1] = tf.Variable([float(self.id_node_dict[x].weights[0]) for x in self.leaf_id_order], dtype=tf.float64)
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
        new_weights = self.session.run(weights, feed_dict={self.conz: [0.8]});
        old_weights = filter(lambda x: x.get_shape()[0] > 0, self.weights)
        for w, n in zip(old_weights, new_weights):
           # print w.get_shape()
           # print n.shape
            z = w.assign(n)
            self.session.run(z)

    def build_fast_forward_pass(self, step=0.003):
        self.check_op = tf.add_check_numerics_ops()
        computations = []
        bob = 1
        if self.node_layers[0][0].t == 'b':
            bob = 2
        with tf.name_scope("input"):
            self.input = tf.placeholder(dtype=tf.float64, 
                                        shape=(None, max(self.input_order)+1, 
                                               bob),
                                        name='Input')
        # self.input = tf.placeholder(dtype=tf.float64, 
        #                                  shape=(len(self.input_order)*2), name='Input')
        #the input to be appended to each layer
        input_splits = []
        self.conz = tf.placeholder(shape=[1], dtype=tf.float64)
        #compute the input
        weights = []
        with tf.name_scope("projection"):
            n = tf.constant(0.0001, dtype=tf.float64)
            for L in range(len(self.weights)):
                if L != 0:
                    drop = tf.round(tf.random_uniform(self.weights[L].get_shape(), self.conz, 1.0, dtype=tf.float64))
                    weights.append(tf.add(tf.nn.relu(tf.sub(self.weights[L]*drop, n)), n))
                else:
                    weights.append(tf.add(tf.nn.relu(tf.sub(self.weights[L], n)), n))

        with tf.name_scope('nomralization'):
            self.sum_of_weights = [tf.segment_sum(x, y) if x.get_shape()[0] > 0 else None for x, y in zip(weights, self.inds)]
            sum_of_weights = self.sum_of_weights
            self.norm_weights = [tf.div(x, tf.gather(y, z)) if x.get_shape()[0] > 0 else None for x, y, z in zip(weights, self.sum_of_weights, self.inds)]
        
        with tf.name_scope('LEAFS_' + str(len(self.input_order))):
            input_gather = tf.reshape(tf.transpose(tf.gather(tf.transpose(self.input, (1, 0, 2)), self.input_swap), (1, 0, 2)), shape=(-1, len(self.input_order)*bob))
            self.counting.append(input_gather)
            if self.node_layers[0][0].t == 'b': #if contiuous
               input_computation_w = tf.mul(input_gather, weights[0])
               input_computation_s = tf.transpose(tf.segment_sum(tf.transpose(input_computation_w), self.inds[0]))
               input_computation_n = tf.log(tf.div(input_computation_s, sum_of_weights[0]))
               computations.append(input_computation_n)
            else:
               pi = tf.constant(np.pi, tf.float64)
               mus = self.cont[0]
               sigs = tf.nn.relu(self.cont[1] - 0.01) + 0.01 #sigma can't be smaller than 0.01
               #gassian formula
               input_computation_g = tf.div(tf.exp(tf.neg(tf.div(tf.square(input_gather - mus), 2*tf.mul(sigs, sigs)))), tf.sqrt(2*pi)*sigs) + 0.000001
               input_computation_n = tf.log(input_computation_g)
               computations.append(input_computation_n)

        
        #split the input computation and figure out which one goes in each layer
            j = 0
            for i in range(len(self.input_layers)):
                a = tf.constant(j)
                b = self.input_layers[i]
                input_splits.append(tf.slice(input_computation_n, [0, a], [-1, b]))
                j += b;

        current_computation = input_splits[0]

        for i in range(len(self.node_layers[1:])):
            L = i+1 #the layer number
          
            if self.weights[L].get_shape()[0] == 0: #product
                with tf.name_scope("PRD" + str(self.inds[L].get_shape()[0])):
                    #do a segment sum in the log domain
                    current_computation = tf.transpose(tf.segment_sum(tf.transpose(current_computation), self.inds[L]))
               
            else:
                with tf.name_scope("SUM" + str(self.inds[L].get_shape()[0])):
                    self.counting.append(current_computation) #stats for counting and cccp

                    #get the max at each node
                    maxes = tf.transpose(tf.segment_max(tf.transpose(current_computation), self.inds[L]))
                    back_maxes = tf.transpose(tf.gather(tf.transpose(maxes), self.inds[L]))

                    #sub the max at each node
                    current_computation = tf.sub(current_computation, back_maxes)
                    #get out of log domain
                    current_computation = tf.exp(current_computation)
                    #multiply by weights
                    current_computation = tf.mul(current_computation, weights[L])
                    #compute sum node
                    current_computation = tf.transpose(tf.segment_sum(tf.transpose(current_computation), self.inds[L]))
                    #normalize
                    current_computation = tf.div(current_computation, sum_of_weights[L])
                    #re-add the maxes that we took out after entering log domain
                    current_computation = tf.add(tf.log(current_computation), maxes)
                    #concatenate with inputs for the next layer
                    current_computation = tf.concat(1, [current_computation, input_splits[L]])
                    #shuffle so that next node is ready
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
                    #calculate derivatives
                    factored_comp = tf.transpose(tf.gather(tf.transpose(self.computations[L]), self.inds[L]))
                    curr = curr + factored_comp - self.computations[i][:, :curr.get_shape()[1]]
                else: #sum node
                    curr = tf.transpose(tf.gather(tf.transpose(curr), self.reverse_shuffle[L]))
                    if (self.input_layers[L] > 0): #merge with inputs
                        curr, split = curr[:, :-self.input_layers[L]], curr[:, -self.input_layers[L]:]
                        splits = [split] + splits;
                    curr = tf.transpose(tf.gather(tf.transpose(curr), self.inds[L], name="mysumgather"))

                    #create updates and derivatives
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
        for i in range(1+(len(data)-1)//compute_size): #do it in small computations and keep track of the updates
            feed_dict[self.input] = data[i*compute_size:min((i+1)*compute_size, len(data))]
            feed_dict[self.num] = feed_dict[self.num][:-i*compute_size+min((i+1)*compute_size, len(data))]
            _, update = self.session.run([self.check_op, self.cccp_updates], feed_dict=feed_dict)
            if len(updates) == 0:
                updates = update
            else:
                for i in range(len(update)):
                    updates[i] += update[i]
        a=0
        weights = filter(lambda x: x.get_shape()[0] > 0, self.weights)
        updates = list(reversed(updates))
        for i in range(len(weights)): #apply updates
            if self.node_layers[0][0].t == 'c' and i == 0:
                for j in range(3):
                    z = tf.assign(self.cont[-j-1], updates[i+j])
                    self.session.run(z)
                a = 2
                continue
            u, w = updates[i+a], weights[i]
            z = tf.assign(w, u)
            self.session.run(z)

        #must normalize between steps
        self.normalize_weights()       

    def countss(self):
        weights = filter(lambda x: x.get_shape()[0] != 0, self.weights)
        with tf.name_scope("Counting"):
            maxed_out = []
            val = tf.constant(0.51, dtype=tf.float64)
            t = lambda x: tf.transpose(x)
            #calculate max values for each node
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
            #label
            self.num2 = tf.placeholder(shape=(None, len(self.node_layers[-1])), dtype=tf.float64)
            curr = self.num2

            for i in reversed(range(len(self.node_layers[1:]))):
                L = i+1
                if self.weights[L].get_shape()[0] == 0: #product node
                    curr = tf.transpose(tf.gather(tf.transpose(curr), self.inds[L], name="myprodgather"))
            
                else: #sum node
            
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

    def fast_compile(self, step=0.03, tensorboard_dir=""):
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
        
