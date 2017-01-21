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
patience = 5
loss = 10000
bad = 0

spn = SPN()
spn.make_fast_model_from_file('logs/' + dataset_name +'.'+ exp_name +'.tespn.txt', random_weights=False, cont=False, classify=False)
#spn.start_session()
#spn.model.unbuild_fast_variables()
#spn.model.clean_nodes(0.01)
#spn.model.save("logs/small.txt")
#spn = SPN()
#spn.make_fast_model_from_file("logs/small.txt", random_weights=False, cont=False, classify=False)

spn.add_data('../Dataz/' + dataset_name + '.ts.data', 'train', True)
spn.add_data('../Dataz/' + dataset_name + '.valid.data', 'valid')
spn.add_data('../Dataz/' + dataset_name + '.test.data', 'test')
start = time.time()
spn.start_session()
spn.model.normalize_weights()
best_test = spn.evaluate(spn.data.test, minibatch_size = minibatch_size)
loss = spn.evaluate(spn.data.valid, minibatch_size = minibatch_size)
print loss
for i in range(50):
    if bad == patience: break
    spn.train(1, spn.data.train, minibatch_size=minibatch_size, count=0, gd=1)
    valid_loss = spn.evaluate(spn.data.valid, minibatch_size = minibatch_size)
    if valid_loss > loss:
        bad += 1
    else:
        bad = 0
        loss = valid_loss
        best_test = spn.evaluate(spn.data.test)
        spn.model.unbuild_fast_variables()
        spn.model.save("offlogs/" + dataset_name + "." + exp_name + ".offspn.txt")
    print "Valid Loss:", valid_loss

end = time.time() - start
test_loss = best_test
print "Test Loss:"
print test_loss

results = dict(test_loss=test_loss, time=end)

myfile = open('offlogs/' + dataset_name + ".off"  + exp_name + '.json', 'w')
myfile.write(json.dumps(results))
myfile.close()
