import tensorflow as tf
import numpy as np
from tachyon.model import *
from tachyon.data import *

class SPN:
	def __init__(self):
		self.model = None  # model
		self.data = Data([]) # data holder 
		self.loss = [] #loss history
		self.val_loss = [] #validation loss history
		self.test_loss = None #current test loss
		self.session_on = False #I don't use this I think
		self.input_order = [] #what is the order of the variables
		self.classify = False #classifier?
		self.continuous = False #continuous?
		self.out = 1 #output size, non 1 not supported yet

	#reshape data for input
	def reshape(self, data): 
		n_data = data
		a = 2
		if self.continuous:
			a = 1
		n_data = n_data[:, self.input_order, :a]
		n_data = np.reshape(n_data, (len(n_data), n_data.shape[1]*n_data.shape[2]))
		return n_data

	def make_random_model(self, bfactor, input_size, output=1, tensorboard_dir="", cont=True, classify=False, data=[], depth=6):
		self.classify = classify
		self.continuous = cont
		self.model = Model()
		if len(data) > 0:
			self.model.set_mv(data) #set mean and variances of the data
		self.out = output
		ctype = 'b' # b for binary
		if cont:
			ctype = 'c' # c for continuous
		self.model.build_random_model(bfactor, input_size, output, ctype=ctype, multiclass=classify, depth=6)
		self.model.fast_compile(tensorboard_dir=tensorboard_dir)
		self.input_order = self.model.input_order

	def make_fast_model_from_file(self, fname, random_weights=False, tensorboard_dir="", cont=True, classify=False):
		self.model = Model()
		self.classify = classify
		self.continuous = cont
		ctype = 'b'
		if cont:
			ctype = 'c'
		self.model.build_fast_model(fname, random_weights, ctype=ctype)
		self.model.fast_compile(tensorboard_dir=tensorboard_dir)
		self.input_order = self.model.input_order

	def add_data(self, filename, dataset='train', mem=False, cont=False):
		if dataset == 'train':
			self.data.load_and_process_train_data_mem(filename, cont=cont)
		elif dataset == 'valid':
			self.data.load_and_process_valid_data(filename, cont=cont)
		elif dataset == 'test':
			self.data.load_and_process_test_data(filename, cont=cont)

	def start_session(self):
		self.model.start_session()
		self.session_on = True

	def close_session(self):
		self.model.close_session()
		self.session_on = False

	def predict(self, data, minibatch_size=1000):#doesn't work yet, but should be fixable
		ms = minibatch_size
		tot_loss = 0
		a = 0
		b = 0
		runs = 1+(len(data)-1)//ms
		for i in range(runs):
			print i+1, "/", runs
			b = min(len(data), a + ms)
			n_data = data[a:b, :, :]#
			if self.classify:
                                    #print np.argmax(labels[a:b], axis=1
				feed_dict = {self.model.input: n_data, self.model.labels: labels[a:b], self.model.num: labels[a:b]}
			else:
				feed_dict = {self.model.input: n_data, self.model.num: [[1.0]*self.out]*(b-a)}
			if (a == b):
				break
			if summ == "":	
				loss = self.model.session.run([self.model.loss], feed_dict=feed_dict)
			else:
				feed_dict[self.model.summ] = summ
				summary, loss = self.model.session.run([self.model.loss_summary, self.model.loss], feed_dict=feed_dict)
				self.model.writer.add_summary(summary, epoch*len(data) + b)
			tot_loss += (b-a)*np.sum(loss)
			a += ms
		tot_loss /= float(len(data))
		return tot_lossmodel.session.run(self.model.output, feed_dict=feed_dict)
		return output

	def evaluate(self, data, labels=None,summ="", minibatch_size=1000, epoch=0):
		ms = minibatch_size
		tot_loss = 0
		a = 0 #start of data for a btach
		b = 0 #end of data for a batch
		runs = 1+(len(data)-1)//ms
		for i in range(runs):
			print i+1, "/", runs
			b = min(len(data), a + ms)
			n_data = data[a:b, :, :] #get data from a to b
			if self.classify:
                                    #print np.argmax(labels[a:b], axis=1
				feed_dict = {self.model.conz: [0.9], self.model.input: n_data, self.model.labels: labels[a:b], self.model.num: labels[a:b]}
			else:
				feed_dict = {self.model.conz: [0.9], self.model.input: n_data, self.model.num: [[1.0]*self.out]*(b-a)}
			if (a == b):
				break
			if summ == "":	
				loss = self.model.session.run([self.model.loss], feed_dict=feed_dict)
			else: #record summaries
				feed_dict[self.model.summ] = summ
				_, summary, loss = self.model.session.run([self.model.check_op, self.model.loss_summary, self.model.loss], feed_dict=feed_dict)
				self.model.writer.add_summary(summary, epoch*len(data) + b)
			tot_loss += (b-a)*np.sum(loss)
			a += ms
		tot_loss /= float(len(data))
		return tot_loss

	def test(self, inp):
		feed_dict = {self.model.input: inp}
		vals = self.model.session.run(self.model.computations, feed_dict=feed_dict)
		print vals
		return vals;

	def train(self, epochs, data=[], labels=[], minibatch_size=512, valid_data=[], gd=True, compute_size=1000, count=False, cccp=False, patience=100, summ=False, dropout=0.0):
		if data == []:
			data = self.data.train
			
		if self.classify:
			assert labels != []

		a = 0
		valid = len(valid_data) > 0 #validation or not!
		bad = 0
		b = 0
		s = 0 #summary number
		val_sum = ""
		if summ:
			val_sum = "valid"
		if valid:
			prev_valid = self.evaluate(valid_data, summ=val_sum, minibatch_size=compute_size, epoch=0)
		history = dict(train_loss=[], valid_loss=[])
		for e in xrange(epochs):
			a = 0
			b = 0
			print 'Epoch ' + str(e)
			tot_loss = 0
			ms = minibatch_size
			for i in range(1+(len(data)-1)//ms):
				print i+1, "/", 1+(len(data)-1)//ms
				b = min(len(data), a + ms)
				n_data = data[a:b, :, :]
				dropout_val = 0.5-(dropout/2)
				if self.classify:
					feed_dict = {self.model.conz: [dropout_val], self.model.input: n_data, self.model.labels: labels[a:b], self.model.num: labels[a:b]}
				else:
					feed_dict = {self.model.conz: [dropout_val], self.model.input: n_data, self.model.num: [[0.0]*self.out]*(b-a), self.model.num2: [[1.0]*self.out]*(b-a)}
				if (a == b):
					break
				if cccp:
					print "yo"
					self.model.apply_cccp(feed_dict, compute_size=compute_size)
				if count:
					self.model.apply_count(feed_dict, compute_size=compute_size)
				if gd:
					_ = self.model.session.run([self.model.check_op, self.model.opt_val], feed_dict = feed_dict)
				if not summ:
					loss = 0#self.model.session.run([self.model.loss], feed_dict=feed_dict)
				else:
					feed_dict[self.model.summ] = "batch_loss"
					summary, loss = self.model.session.run([self.model.loss_summary, self.model.loss], feed_dict=feed_dict)
					self.model.writer.add_summary(summary, s)
					s += 1
				tot_loss += (b-a)*np.mean(loss)
				a += ms
			tot_loss /= float(len(data))
			print tot_loss
			history["train_loss"].append(tot_loss)
			if valid:
				valid_loss = self.evaluate(valid_data, summ=val_sum, minibatch_size=compute_size ,epoch=e+1)
				history["valid_loss"] = valid_loss
				print valid_loss
				if valid_loss > prev_valid:
					bad += 1
					if bad == patience:
						break;
				else:
					bad = 0
					prev_valid = valid_loss

			np.random.shuffle(data)
		return history
			

	def get_size(self):
		return len(self.model.id_node_dict) + len(self.model.input_order)
	
	def get_weights(self):
		return self.model.num_weights
