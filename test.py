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
compute_size = int(sys.argv[4])
step = 0
gd = 0
cccp = 1
patience = 2
loss = 10000
bad = 0

spn = SPN()
spn.make_fast_model_from_file('structlogz/' + dataset_name +"."+exp_name+'.smallspn.txt', random_weights=False, cont=False, classify=False)
spn.add_data('../Dataz/' + dataset_name + '.test.data', 'test')
spn.start_session()
test_loss = spn.evaluate(spn.data.test, minibatch_size=compute_size)
print "Test Loss:"
print test_loss
end = 10
results = dict(test_loss=test_loss, time=end)

myfile = open('structlogz/' + dataset_name + ".ss"  + exp_name + '.json', 'w')
myfile.write(json.dumps(results))
myfile.close()
