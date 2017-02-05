import numpy as np

class Data:
    def __init__(self, input_order):
        self.input_order = input_order
        self.train = None
        self.valid = None
        self.test = None

    def load_data_from_file(self, fname, cont=False):
        D = open(fname, 'r')
        Data = []
        print fname
        i = 'lol'
        while (i != ''):
            i = D.readline()
            if i == '':
                break;
            if not cont:
                Data.append([[float(x), 0] if float(x) == 1.0 else [float(x), 1] for x in i.split(',')])
            else:
                Data.append([[float(x)] for x in i.split(',')])
        return np.array(Data)


    def load_and_process_train_data(self, fname):
        D = self.load_data_from_file(fname)
        P1 = D[:, self.input_order, :]
        P2 = np.reshape(P1, (len(P1), P1.shape[1]*P1.shape[2]))
        self.train = P2.T

    def load_and_process_train_data_mem(self, fname, cont=False):
        D = self.load_data_from_file(fname, cont)
        self.train = D

    def load_and_process_valid_data(self, fname, cont=False):
        D = self.load_data_from_file(fname, cont)
	self.valid = D

    def load_and_process_test_data(self, fname, cont=False):
        D = self.load_data_from_file(fname, cont)
      #  P1 = D[:, self.input_order, :]
      #  P2 = np.reshape(P1, (len(P1), P1.shape[1]*P1.shape[2]))
        self.test = D# P2.T
        
