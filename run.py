import time
from tachyon.model import Model 
from tachyon.SPN2 import SPN
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
gd = 0
cccp = 0
patience = 2
loss = 10000
bad = 0
bfactor = ((2, 6), (15, 18))
spn = SPN()
spn.add_data('../Dataz/' + dataset_name + '.ts.data', 'train', True, cont=cont)
#spn.add_data('../Dataz/' + dataset_name + '.valid.data', 'valid')
spn.add_data('../Dataz/' + dataset_name + '.test.data', 'test', cont=cont)
spn.make_random_model(bfactor, 8, 1, data=spn.data.train, cont=cont, tensorboard_dir="tester/")
#spn.make_fast_model_from_file('../Modelz/' + dataset_name +'.spn.txt', random_weights=True, cont=cont, classify=False)

start = time.time()
spn.start_session()
spn.train(2, spn.data.train, minibatch_size=minibatch_size, compute_size=compute_size, count=1, gd=0, cccp=0, dropout=0.0)
#spn.train(50, spn.data.train, patience=patience, valid_data=spn.data.test, minibatch_size=minibatch_size, compute_size=compute_size, count=step, gd=gd, cccp=cccp)#, ngd=2)
end = time.time() - start
#spn.model.normalize_weights()
spn.model.unbuild_fast_variables()
f = ""+ dataset_name +"." + exp_name + ".largespn.txt"
f2 = ""+ dataset_name +"." + exp_name + ".smallspn.txt"
spn.model.save(f)
spn.model.clean_nodes(0.01)
spn.model.save(f2)

spn2 = SPN()
spn2.make_fast_model_from_file(f2, random_weights=False, cont=cont, classify=False)
spn2.start_session()
print spn2.get_size()
print spn.get_size()
#spn.model.unbuild_fast_variables()
test_loss = spn.evaluate(spn.data.test, minibatch_size=compute_size)
test_loss2 = spn2.evaluate(spn.data.test, minibatch_size=compute_size)
end = time.time() - start
print "Test Loss:"
print test_loss, test_loss2

results = dict(test_loss=test_loss, time=end)

myfile = open('structlogz/' + dataset_name + "."  + exp_name + '.json', 'w')
myfile.write(json.dumps(results))
myfile.close()
