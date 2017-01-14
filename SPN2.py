import tensorflow as tf
import numpy as np
from model import *
from data import *

class SPN:
	def __init__(self):
		self.model = None
		self.data = None
		self.loss = []
		self.val_loss = []
		self.test_loss = None
		self.session_on = False
		self.input_order = []
		self.classify = False
		self.continuous = False
		self.out = 1

	def reshape(self, data):
		n_data = data
		a = 2
		if self.continuous:
			a = 1
		n_data = n_data[:, self.input_order, :a]
		n_data = np.reshape(n_data, (len(n_data), n_data.shape[1]*n_data.shape[2]))
		return n_data

	def make_model_from_file(self, fname, random_weights=False, step=0.003):
		self.model = Model()
		self.model.build_model_from_file(fname, random_weights)
		self.model.compile(step)
		self.data = Data(self.model.input_order)
		self.input_order = self.model.input_order

	def make_random_model(self, bfactor, input_size, output=1, step=0.003, cont=True, classify=False):
		self.classify = classify
		self.continuous = cont
		self.model = Model()
		self.out = output
		ctype = 'b'
		if cont:
			ctype = 'c'
		self.model.build_random_model(bfactor, input_size, output, ctype=ctype, multiclass=classify)
		self.model.fast_compile(step)
		self.data = Data(self.model.input_order)
		self.input_order = self.model.input_order

	def make_fast_model_from_file(self, fname, random_weights=False, step=0.003):
		self.model = Model()
		self.model.build_fast_model(fname, random_weights)
		self.model.fast_compile(step)
		self.data = Data(self.model.input_order)
		self.input_order = self.model.input_order

	def add_data(self, filename, dataset='train', mem=False):
		if dataset == 'train':
			if mem:
				print "zebra"
				self.data.load_and_process_train_data_mem(filename)
			else:
				print "zebro"
				self.data.load_and_process_train_data(filename)
		elif dataset == 'valid':
			self.data.load_and_process_valid_data(filename)
		elif dataset == 'test':
			self.data.load_and_process_test_data(filename)

	def start_session(self):
		self.model.start_session()
		self.session_on = True

	def close_session(self):
		self.model.close_session()
		self.session_on = False

	def predict(self, inp):
		feed_dict = {self.model.input: self.reshape(inp)}
		output = self.model.session.run(self.model.output, feed_dict=feed_dict)
		return output

	def evaluate(self, inp, labels=None,summ="evaluation"):
		if self.classify:
			assert labels != None
			feed_dict = {self.model.input: self.reshape(inp), self.model.labels: labels}
		else:
			feed_dict = {self.model.input: self.reshape(inp)}
		loss = self.model.session.run([self.model.loss], feed_dict=feed_dict)
		return loss

	def test(self, inp):
		feed_dict = {self.model.input: inp}
		vals = self.model.session.run(self.model.computations, feed_dict=feed_dict)
		print vals
		return vals;

	def train(self, epochs, data=[], labels=[], minibatch_size=512, valid=False, test=False, gd=True, count=False):
		if data == []:
			data = self.data.train
			print data.shape
		if self.classify:
			assert labels != []
		# if (valid):	
		# 	val_loss, val_sum = self.evaluate(self.data.valid.T, 'valid_loss')
		# 	self.model.writer.add_summary(val_sum, 0)
		# 	print val_loss

		# if (test):
		# 	test_loss, test_sum = self.evaluate(self.data.test.T, 'test_loss')
		# 	self.model.writer.add_summary(test_sum, 0)
		a = 0
		b = 0
		for e in xrange(epochs):
			a = 0
			b = 0
			print 'Epoch ' + str(e)
			ms = minibatch_size
			for i in range(1+(len(data)-1)//ms):
				print i+1, "/", 1+(len(data)-1)//ms
				b = min(len(data), a + ms)
				n_data = data[a:b]#self.reshape(data[a:b])
				print n_data.shape
				if self.classify:
                                        #print np.argmax(labels[a:b], axis=1
					feed_dict = {self.model.input: n_data, self.model.labels: labels[a:b], self.model.num: labels[a:b]}
				else:
					feed_dict = {self.model.input: n_data, self.model.num: [[1.0]*self.out]*(b-a)}
				if (a == b):
					break
				loss = 0
				if count:
					self.model.apply_count(feed_dict, 1.0)
					loss = self.model.session.run([self.model.loss], feed_dict=feed_dict)
				if gd:
					_, loss = self.model.session.run([self.model.opt_val, self.model.loss], feed_dict = feed_dict)
				a += ms
			print loss 
			# np.random.shuffle(data)
			# for m in xrange(data.shape[0]//minibatch_size+1):
			# 	n_data = data[m*minibatch_size:min(data.shape[0], (m+1)*minibatch_size)]
			# 	n_data = n_data[:, self.input_order, :]
			# 	n_data = np.reshape(n_data, (len(n_data), len(self.input_order)*2))
			# 	feed_dict = {self.model.input: n_data, self.model.summ: "minibatch_loss"}
			# 	if e == 0:
			# 		_, loss, result, summary = self.model.session.run([self.model.opt_val, 
			# 												  self.model.loss,
			# 												  self.model.output, 
			# 												  self.model.loss_summary], 
			# 												  feed_dict=feed_dict)
			# 	else:
			# 		_, loss, result, summary = self.model.session.run([self.model.opt_val2, 
			# 												  self.model.loss,
			# 												  self.model.output, 
			# 												  self.model.loss_summary], 
			# 												  feed_dict=feed_dict)
			# 	self.model.writer.add_summary(summary, e*data.shape[0]//minibatch_size+1 + m)
			# 	self.loss.append(loss)
			# 	self.model.get_normal_value()
				# print self.model.norm_value
				# print "Loss: " + str(loss)
			# if (valid):
			# 	val_loss, val_sum = self.evaluate(self.data.valid.T, 'valid_loss')
			# 	self.model.writer.add_summary(val_sum, e+1)
			# 	print val_loss

			# if (test):
			# 	test_loss, test_sum = self.evaluate(self.data.test.T, 'test_loss')
			# 	self.model.writer.add_summary(test_sum, e+1)
			# print self.model.get_normal_value()
				# print "Min: " + str(result)
				# print list(data[:, m*500:min(data.shape[1], (m+1)*500)])
				# print map(lambda x: self.model.session.run(x), self.model.sparse_tensors)

	def get_size(self):
		return len(self.model.id_node_dict) + len(self.model.input_order)
	
	def get_weights(self):
		return self.model.num_weights
