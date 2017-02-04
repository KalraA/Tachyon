import time
from model import Model 
from SPN2 import SPN
import numpy as np
import random
import sys
import json

cont = True
exp_name = sys.argv[1]
dataset_name = sys.argv[2]
minibatch_size = int(sys.argv[3])
compute_size = int(sys.argv[4])
step = 0
gd = 1
cccp = 0
patience = 2
loss = 10000
bad = 0
bfactor = ((2, 6), (15, 18))
spn = SPN()
#spn.make_fast_model_from_file('../Modelz/' + dataset_name +'.spn.txt', random_weights=True, cont=cont, classify=False)
spn.add_data('../Dataz/' + dataset_name + '.ts.data', 'train', True, cont=cont)
#spn.add_data('../Dataz/' + dataset_name + '.valid.data', 'valid')
spn.add_data('../Dataz/' + dataset_name + '.test.data', 'test', cont=cont)
spn.make_random_model(bfactor, 8, 1, data=spn.data.train)
start = time.time()
spn.start_session()
spn.train(1, spn.data.train, minibatch_size=minibatch_size, compute_size=compute_size, count=1, gd=0, cccp=0)
spn.train(50, spn.data.train, patience=patience, valid_data=spn.data.test, minibatch_size=minibatch_size, compute_size=compute_size, count=step, gd=gd, cccp=cccp)#, ngd=2)
end = time.time() - start
#spn.model.normalize_weights()
#spn.model.unbuild_fast_variables()
#spn.model.save("structlogz/"+ dataset_name +"." + exp_name + ".largespn.txt")
#spn.model.unbuild_fast_variables()
#spn.model.clean_nodes(0.01)
#spn.model.save("structlogz/"+dataset_name+"."+exp_name+".smallspn.txt")
test_loss = spn.evaluate(spn.data.test, minibatch_size=compute_size)
end = time.time() - start
print "Test Loss:"
print test_loss

results = dict(test_loss=test_loss, time=end)

myfile = open('structlogz/' + dataset_name + "."  + exp_name + '.json', 'w')
myfile.write(json.dumps(results))
myfile.close()
