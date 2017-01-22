import time
from model import Model 
from SPN2 import SPN
import numpy as np
import random
import sys
import json


exp_name = sys.argv[1]
dataset_name = sys.argv[2]
minibatch_size = int(sys.argv[3])
step = 1
gd = 0
cccp = 0
patience = 2
loss = 10000
bad = 0

spn = SPN()
spn.make_fast_model_from_file('../models/' + dataset_name +'.spn.txt', random_weights=True, cont=False, classify=False)
spn.add_data('../Dataz/' + dataset_name + '.ts.data', 'train', True)
spn.add_data('../Dataz/' + dataset_name + '.valid.data', 'valid')
spn.add_data('../Dataz/' + dataset_name + '.test.data', 'test')
start = time.time()
spn.start_session()
spn.train(10, spn.data.train, patience=patience, valid_data=spn.data.valid, minibatch_size=minibatch_size, count=step, gd=gd, cccp=cccp)#, ngd=2)
end = time.time() - start
spn.model.unbuild_fast_variables()
spn.model.save("logs/nltcs." + exp_name + ".tespn.txt")
test_loss = spn.evaluate(spn.data.test)
end = time.time() - start
print "Test Loss:"
print test_loss

results = dict(test_loss=test_loss, time=end)

myfile = open('offlogs/' + dataset_name + ".off"  + exp_name + '.json', 'w')
myfile.write(json.dumps(results))
myfile.close()
