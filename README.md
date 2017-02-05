
SPN-Z Documentation
===================
SPN-Z is an open source library for sum product networks on GPUs. It's extemely fast and accurate.

### Table of contents

[TOC]

### Getting started

Getting started is as easy as the following code:

```
# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('nltcs.ts.data', 'train', cont=False)
spn.add_data('nltcs.valid.data', 'valid', cont=False)
spn.add_data('nltcs.test.data', 'test', cont=False)
# create a valid sum product network

sum_branch_factor = (3, 6)
prod_branch_factor = (5, 8)
variables = 16

spn.make_random_model((prod_branch_factor, sum_branch_factor), variables)

# start the session
spn.start_session()

# train
epochs = 10
# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test
spn.train(epochs, train)
test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 6.182

```
### Docs

#### SPN:

##### init

```
__init__()
```
Just initializes the holder of your spn model.

##### make_model_from_file
```
make_fast_model_from_file(self, fname, random_weights=False, cont=False, classify=False)
```

Parameters

 - **random_weights** - use the weights in the file or generate random ones
 - **cont** - True for continuous variables, False for binary variables (Currently only working consistently for binary variables)
 - **classify** - is this doing classification (Note: this doesn't work yet)

This takes an SPN file with the following format:
```
###NODES###
id SUM
id PRD
id LEAVE var_num w1 w2
###EDGES###
parent_id child_id weight 
```
and creates an spn model. 

##### make_random_model
```
make_random_model(self, bfactor, input_size, output=1, cont=False, classify=False, data=[])
```

Parameters

 - **bfactor** - a tuple of tuples in the format (prod branch factor, sum branch factor)  where a branch factor looks like (lower_bound, upper_bound)
 - **input_size** - the number of variables total
 - **output_size** - 1 unless you are doing classification, but classification doesn't work yet so don't touch
 - **cont** - True for continuous variables, False for binary variables
 - **classify** - is this doing classification (Note: this doesn't work yet)
 - **data** - the training data so if you are doing continuou variables the mean and variance of the leaf nodes can be derived

This generates an spn model with a specific branching factor. Max depth is capped at 11 layers. The algorithm goes like this:
```
bfactor = ((a, b), (c, d))
start = SumNode

--- Recursively for all children ---
if curr_node = SUM:
   if scope has only one variable convert into leaf node 
   else generate a number of children product nodes between c and d
if product node:
   generate a number of children sum nodes between a and b
   randomly factor the scope of this node between the children
```
This guarentees a valid spn (complete and decomposable)


##### add_data
```
add_data(self, filename, dataset='train', cont=False)
```

Parameters

 - **filename** - the file that contains the data
 - **dataset** - a string that's either "train", "valid", or "test" and determines if you want this data to be accessed by spn.data.train, spn.data.valid, or spn.data.test
 - **cont** - True for continuous variables, False for binary variables

Processed data from a text file and loads it into memory. 

##### start_session
```
start_session(self)
```

Starts the tensorflow session inside the holder object

##### close_session
```
close_session(self)
```

Closes the tensorflow session

##### evaluate
```
evaluate(self, data, labels=None, summ="", minibatch_size=1000, epoch=0)
```
Parameters:

 - **data** - A data matrix do evaluate
 - **labels** - Labels for the data. Optional and will only matter if classify=True. If labels aren't given, simply loss will be outputed
 - **summ** - tensorflow summary name
 - **minibatch_size** - the minibatch size for making predictions. 
 - **epoch** - The global_step that goes with the tensorflow summary

evaluate the model on some data

##### train
```
train(self, epochs, data=[], labels=[], minibatch_size=512, valid_data=[], gd=True, compute_size=1000, count=False, cccp=False, patience=100, summ=True):
```
Parameters:

 - **epochs** - number of iterations across the data
 - **data** - training data matrix
 - **labels** - optional, only if classify=True
 - **minibatch_size** - the size of the batches in batch training
 - **valid_data** - validation data
 - **gd** - use gradient descent? (both binary and continuous)
 - **compute_size** - size to do computations. Sometimes you want large batch sizes but cannot fit into memory. So you computes large batches in smaller amounts so it can fit on your GPU. only affects counting and cccp, not gradient descent
 - **count** - use counting? (both binary and continuous)
 - **cccp** - use em/cccp? (only binary)
 - **patience** - early stopping, if the validation loss stops decreasing for x number of iterations, kill the training
 - **summ** - use tensorboard logs?

##### get_size
```
get_size(self)
```

returns an integer representing the number of nodes in the SPN
##### get_weights
```
get_weights(self)
```
returns the number of parameters in a model

#### SPN.model
These are available if you call: spn.model

##### unbuild_fast_variables
```
unbuild_fast_variables(self)
```
take all the weights from tensorflow and put them back onto the initial model. Must be called before saving

##### save
```
save(self, fname)
```
save the model to a specific filename. Include the path in the fname variable

##### clean_nodes
```
clean_nodes(self)
```
Prunes all unused parts of the SPN. Used with counting algorithm.

##### normalize_weights
```
normalize_weight(self)
```
normalizes all weights in the SPN

### Examples

These are different ways of using the library.

#### Offline Learning

```
# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('nltcs.ts.data', 'train', cont=False)
spn.add_data('nltcs.valid.data', 'valid', cont=False)
spn.add_data('nltcs.test.data', 'test', cont=False)
# create a valid sum product network

sum_branch_factor = (3, 6)
prod_branch_factor = (5, 8)
variables = 16

#binary model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=False, classify=False)

# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = True
gd = False
count = False

#large minibatches with a small number at a time
minibatch_size=1000
compute_size=10

# other stuff
epochs = 1000
patience = 2

# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test


spn.train(epochs, train, valid_data=spn.data.valid, patience=patience, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size, compute_size=compute_size)
test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 6.034

```

#### Online Learning

```
# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('abalone.ts.data', 'train', cont=True)
spn.add_data('abalone.valid.data', 'valid', cont=True)
spn.add_data('abalone.test.data', 'test', cont=True)
# create a valid sum product network

sum_branch_factor = (3, 6)
prod_branch_factor = (5, 8)
variables = 8

#continuous model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=True, classify=False, data=spn.data.train)

# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = False
gd = True
count = True #take a step of counting then a step of gradient descent

#large minibatches with a small number at a time
minibatch_size=1

# other stuff
epochs = 1

# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test


spn.train(epochs, train, valid_data=valid,cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size)
test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 3.013

```

#### Counting Structure Learning

```
# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('nltcs.ts.data', 'train', cont=False)
spn.add_data('nltcs.valid.data', 'valid', cont=False)
spn.add_data('nltcs.test.data', 'test', cont=False)
# create a valid sum product network

sum_branch_factor = (3, 6)
prod_branch_factor = (5, 8)
variables = 8

#binary model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=False, classify=False, data=spn.data.train)

# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = False
gd = False
count = True

#large minibatches with a small number at a time
minibatch_size=100

# other stuff
epochs = 5

# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test


spn.train(epochs, train, valid_data=valid,cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size)
test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 6.083

spn.model.unbuild_fast_variables() #pull weights from tensorflow.
spn.model.save("Models/large.spn.txt")
spn.model.clean_nodes()
spn.model.save("Models/small.spn.txt")

spn2 = SPN()
spn2.make_fast_model_from_file("Models/small.spn.txt", random_weights=False, cont=False, classify=False)

print "Loss:", spn2.evaluate(test)
# Loss: 6.083

# continue training with more sophisticated algorithm

spn.train(1000, train, valid_data=valid,cccp=True, gd=False, count=False, minibatch_size=1000000, compute_size=100, patience=2)

print "Loss:", spn2.evaluate(test)
# Loss: 6.037

```

#### Datasets that don't fit into memory

```
# make an SPN holder
spn = SPN()
# include training and testing data
#let's assume nltcs is split into 2 datasets
#nltcs.ts1.data, nltcs.ts2.data
spn.add_data('abalone.valid.data', 'valid', cont=False)
spn.add_data('abalone.test.data', 'test', cont=False)
# create a valid sum product network

sum_branch_factor = (3, 6)
prod_branch_factor = (5, 8)
variables = 8

#binary model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=False, classify=False, data=spn.data.train)

# start the session
spn.start_session()

# train
filenames = ['nltcs.ts1.data', 'nltcs.ts2.data']
# pick the optimization algorithm
cccp = False
gd = True
count = False

#large minibatches with a small number at a time
minibatch_size=100

# other stuff
epochs = 5

# access the data
valid = spn.data.valid
test = spn.data.test

for e in range(epochs):
	for filename in filenames:
	    spn.add_data(filename, 'train', cont=False) 
	    train = spn.data.train
		spn.train(1, train, valid_data=valid,cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size)


test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 6.083

spn.model.unbuild_fast_variables() #pull weights from tensorflow.
spn.model.save("Models/nltcs.spn.txt")

```

#### Combining training algorithms

```
# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('abalone.ts.data', 'train', cont=True)
spn.add_data('abalone.valid.data', 'valid', cont=True)
spn.add_data('abalone.test.data', 'test', cont=True)
# create a valid sum product network

sum_branch_factor = (3, 6)
prod_branch_factor = (5, 8)
variables = 8

#continuous model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=True, classify=False, data=spn.data.train)

# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = False
gd = False
count = True 

#large minibatches with a small number at a time
minibatch_size=1

# other stuff
epochs = 1

# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test

#train with counting first
spn.train(epochs, train, valid_data=valid,cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size)
#train with gradient descent 
spn.train(1000, train, valid_data=valid, cccp=false, gd=True, count=False, minibatch_size=200, patience=1)


test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 1.89

```

> Written with [StackEdit](https://stackedit.io/).