
Tachyon Documentation
===================
Tachyon is an open source library for sum product networks on GPUs. It's extemely fast and accurate.

### Table of contents

1. [About](#about)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Documentation](#docs)
5. [Examples](#examples)

### About
Tachyon is a sum product network build ontop of tensorflow and numpy. It supports both univariate binary and contiuous leaf nodes, and has 3 parameter learning algorithms: CCCP/EM (binary only), Gradient Descent (Adam, RMSProp, Momentum, Adagrad ... etc), Counting. As well as a structure pruning algorithm. It is built by the University of Waterloo computer science department. Please email a6kalra [AT] uwaterloo [DOT] ca for inquiries. 

Why Tachyon? Tachyon is a hypothetical particle faster than the speed of light. Our library usually has on average 2x-100x speedups over CPU C++ implementations of counting, cccp, or gradient descent. 

### Installation

```bash
pip install tachyon
```

and for the models and data:

```bash
mkdir my_working_directory
cd my_working_directory
mkdir models
mkdir data
cd models
wget https://www.dropbox.com/s/i6txi9q2x6b9fgy/spn_models.zip
unzip spn_models.zip
rm spn_models.zip
cd ../data
wget https://www.dropbox.com/s/axu6xi6xtida4y0/Dataz.zip
wget https://www.dropbox.com/s/08aw5j42dcuyp5v/abalone.zip
unzip *.zip
rm *.zip
cd ..

```

### Getting started
For detailed examples check the [Examples](#examples) section
Getting started is as easy as the following code:

```python
from tachyon.SPN2 import SPN

# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('data/nltcs.ts.data', 'train', cont=False)
spn.add_data('data/nltcs.valid.data', 'valid', cont=False)
spn.add_data('data/nltcs.test.data', 'test', cont=False)
# create a valid sum product network

sum_branch_factor = (8, 12)
prod_branch_factor = (2, 4)
variables = 16

spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=False)

# start the session
spn.start_session()

# train
epochs = 10
# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test
spn.train(epochs, train, minibatch_size=100)
test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 6.263
```

### Docs

#### SPN:

##### init

```python
__init__()
```
Just initializes the holder of your spn model.

##### make_fast_model_from_file
```python
make_fast_model_from_file(self, fname, random_weights=False, tensorboard_dir="", cont=False, classify=False)
```

Parameters

 - **fname** - location of the name of the model file
 - **random_weights** - use the weights in the file or generate random ones
 - **tensorboard_dir** - location of the tensorboard logs. If "" then tensorboard logs will not be produced.
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
```python
make_random_model(self, bfactor, input_size, output=1, tensorboard_dir="", cont=False, classify=False, data=[])
```

Parameters

 - **bfactor** - a tuple of tuples in the format (prod branch factor, sum branch factor)  where a branch factor looks like (lower_bound, upper_bound)
 - **input_size** - the number of variables total
 - **tensorboard_dir** - location of the tensorboard logs. If "" then tensorboard logs will not be produced.
 - **output_size** - 1 unless you are doing classification, but classification doesn't work yet so don't touch
 - **cont** - True for continuous variables, False for binary variables
 - **classify** - is this doing classification (Note: this doesn't work yet)
 - **data** - the training data so if you are doing continuou variables the mean and variance of the leaf nodes can be derived

This generates an spn model with a specific branching factor. Max depth is capped at 11 layers. The algorithm goes like this:
```python
bfactor = ((a, b), (c, d))
start = SumNode

# --- Recursively for all children ---
if curr_node = SUM:
   if scope has only one variable convert into leaf node 
   else generate a number of children product nodes between c and d
if product node:
   generate a number of children sum nodes between a and b
   randomly factor the scope of this node between the children
```
This guarentees a valid spn (complete and decomposable)


##### add_data
```python
add_data(self, filename, dataset='train', cont=False)
```

Parameters

 - **filename** - the file that contains the data
 - **dataset** - a string that's either "train", "valid", or "test" and determines if you want this data to be accessed by spn.data.train, spn.data.valid, or spn.data.test
 - **cont** - True for continuous variables, False for binary variables

Processed data from a text file and loads it into memory. 

##### start_session
```python
start_session(self)
```

Starts the tensorflow session inside the holder object

##### close_session
```python
close_session(self)
```

Closes the tensorflow session

##### evaluate
```python
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
```python
train(self, epochs, data=[], labels=[], minibatch_size=512, valid_data=[], gd=True, compute_size=1000, count=False, cccp=False, patience=100, summ=True, dropout=0.0):
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
 - **dropout** - what is the dropout percentage for network. Dropout doesn't do that well for SPNs but sometimes it does good. 

##### get_size
```python
get_size(self)
```

returns an integer representing the number of nodes in the SPN
##### get_weights
```python
get_weights(self)
```
returns the number of parameters in a model

#### SPN.model
These are available if you call: spn.model

##### unbuild_fast_variables
```python
unbuild_fast_variables(self)
```
take all the weights from tensorflow and put them back onto the initial model. Must be called before saving

##### save
```python
save(self, fname)
```
save the model to a specific filename. Include the path in the fname variable

##### clean_nodes
```python
clean_nodes(self)
```
Prunes all unused parts of the SPN. Used with counting algorithm.

##### normalize_weights
```python
normalize_weight(self)
```
normalizes all weights in the SPN

### Examples

These are different ways of using the library.

#### Offline Learning

```python
from tachyon.SPN2 import SPN

# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('data/nltcs.ts.data', 'train', cont=False)
spn.add_data('data/nltcs.valid.data', 'valid', cont=False)
spn.add_data('data/nltcs.test.data', 'test', cont=False)
# create a valid sum product network

sum_branch_factor = (8, 12)
prod_branch_factor = (2, 5)
variables = 16

#binary model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=False, classify=False, tensorboard_dir="./logs")
# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = True
gd = False
count = False

#large minibatches with a small number at a time
minibatch_size=8100
compute_size=1000

# other stuff
epochs = 1000
patience = 2

# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test


spn.train(epochs, train, valid_data=spn.data.valid, patience=patience, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size, compute_size=compute_size, summ=True)
test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 6.114

```

Then to see the graphs:
```bash
tensorboard --logdir=./logs
```

#### Online Learning

```python
from tachyon.SPN2 import SPN

# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('data/abalone.ts.data', 'train', cont=True)
spn.add_data('data/abalone.valid.data', 'valid', cont=True)
spn.add_data('data/abalone.test.data', 'test', cont=True)
# create a valid sum product network

sum_branch_factor = (3, 6)
prod_branch_factor = (2, 4)
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


spn.train(epochs, train, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size)
test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 4.513

```

#### Counting Structure Learning

```python
from tachyon.SPN2 import SPN

# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('data/abalone.ts.data', 'train', cont=True)
spn.add_data('data/abalone.valid.data', 'valid', cont=True)
spn.add_data('data/abalone.test.data', 'test', cont=True)
# create a valid sum product network

sum_branch_factor = (8, 12)
prod_branch_factor = (2, 4)
variables = 8

#continuous model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=True, classify=False, data=spn.data.train)

# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = False
gd = False
count = True #take a step of counting then a step of gradient descent

#large minibatches with a small number at a time
minibatch_size=100

# other stuff
epochs = 1

# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test


spn.train(epochs, train, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size)
test_loss = spn.evaluate(test)

spn.model.unbuild_fast_variables()
small = "./small.spn.txt"
big = "./big.spn.txt"
spn.model.save(big)
spn.model.clean_nodes()
spn.model.save(small)

spn2 = SPN()
spn2.make_fast_model_from_file(small, cont=True, random_weights=False)
spn2.start_session()

test_loss2 = spn2.evaluate(test)



print 'Loss:', test_loss, test_loss2
print 'Sizes:', spn.get_size(), spn2.get_size()
# Loss: 3.684, 3.683
# Sizes: 1876, 884


```

#### Datasets that don't fit into memory

```python
from tachyon.SPN2 import SPN

# make an SPN holder
spn = SPN()
# include training and testing data
files = ['data/nltcs.ts1.data', 'data/nltcs.ts2.data'] # we split nltcs into 2 parts
spn.add_data('data/nltcs.valid.data', 'valid', cont=False)
spn.add_data('data/nltcs.test.data', 'test', cont=False)
# create a valid sum product network

sum_branch_factor = (8, 12)
prod_branch_factor = (2, 5)
variables = 16

#binary model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=False, classify=False, tensorboard_dir="./logs")
# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = True
gd = False
count = False

#large minibatches with a small number at a time
minibatch_size=8100
compute_size=1000

# other stuff
epochs = 20

# access the data
valid = spn.data.valid
test = spn.data.test

for e in epochs:
	for f in files:
		spn.add_data(f, 'train', cont=False)
		train = spn.data.train
		spn.train(epochs, train, patience=patience, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size, compute_size=compute_size, summ=True)
		
test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 6.154

spn.model.unbuild_fast_variables() #pull weights from tensorflow.
spn.model.save("Models/nltcs.spn.txt")

```

#### Combining training algorithms

```python
from tachyon.SPN2 import SPN

# make an SPN holder
spn = SPN()
# include training and testing data
spn.add_data('data/nltcs.ts.data', 'train', cont=False)
spn.add_data('data/nltcs.valid.data', 'valid', cont=False)
spn.add_data('data/nltcs.test.data', 'test', cont=False)
# create a valid sum product network

sum_branch_factor = (10, 14)
prod_branch_factor = (2, 5)
variables = 16

#binary model
spn.make_random_model((prod_branch_factor, sum_branch_factor), variables, cont=False, classify=False)
# start the session
spn.start_session()

# train

# pick the optimization algorithm
cccp = False
gd = False
count = True

#large minibatches with a small number at a time
minibatch_size=100
compute_size=100

# other stuff
epochs = 1000
patience = 2

# access the data
train = spn.data.train
valid = spn.data.valid
test = spn.data.test


spn.train(1, train, valid_data=spn.data.valid, patience=patience, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size, compute_size=compute_size, summ=True)

cccp = False
gd = True
count = False

spn.train(1000, train, valid_data=spn.data.valid, patience=patience, cccp=cccp, gd=gd, count=count, minibatch_size=minibatch_size, compute_size=compute_size, summ=True)
test_loss = spn.evaluate(test)
print 'Loss:', test_loss
# Loss: 6.091

```

> Written with [StackEdit](https://stackedit.io/).
