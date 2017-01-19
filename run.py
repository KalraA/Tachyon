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
step = int(sys.argv[4])
gd = int(sys.argv[5])

spn = SPN()
spn.make_fast_model_from_file('../Modelz/' + dataset_name +'.spn.txt', random_weights=True, cont=False, classify=False)
spn.add_data('../Dataz/' + dataset_name + '.ts.data', 'train', True)
spn.add_data('../Dataz/' + dataset_name + '.test.data', 'test')
start = time.time()
spn.start_session()
spn.train(1, spn.data.train, minibatch_size=minibatch_size, count=step, gd=gd)
end = time.time() - start
spn.model.unbuild_fast_variables()
spn.model.save("logs/nltcs." + exp_name + ".tespn.txt")
test_loss = spn.evaluate(spn.data.test)

print "Test Loss:"
print test_loss

results = dict(test_loss=test_loss, time=end)

myfile = open('logs/' + exp_name + '.json', 'w')
myfile.write(json.dumps(results))
myfile.close()
