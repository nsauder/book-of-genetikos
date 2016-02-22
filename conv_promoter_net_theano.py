#Input for this is in the form of a list of tuples, where each tuple has two positions, the first position is an ND.array which is 
#4xthe length of the promoter sequence represented, where A,G,C,T is the order (so if position 0 is 1, then it is an A, there is only one
#one for every 4 nucleotides)
#In second position is an NDarray containing the expression data for all the cell types, in which each position in the 1D tensor represents 
#a different cell type

#Import modules

import theano
import theano.tensor as T
import lasagne
import numpy as np
import pickle 

#Import data and extract information about the size of it
input_data = pickle.load(open('promoters_microarray.p','rb'))
number_of_microarrays = input_data[0][1].shape[0]
promoter_size = input_data[0][0].shape[0]
#Turn the list of tuples into two NDarrays
#FIXME: reshape into nX4
(promoters,microarrays) = zip(*input_data)
(all_promoters,all_microarrays) = np.asarray(promoters),np.asarray(microarrays)
all_promoters = all_promoters.astype(np.float32).reshape(-1,500,4).transpose(0,2,1)
all_microarrays = all_microarrays.astype(np.float32)

l_in = lasagne.layers.InputLayer(
    shape=(None, 4, 500),
)

l = lasagne.layers.Conv1DLayer(
    l_in,
    num_filters=32,
    filter_size=(5,),
    pad=1,
)

l = lasagne.layers.Conv1DLayer(
    l,
    num_filters=32,
    filter_size=(5,),
    pad=1,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform(),
)

l = lasagne.layers.Conv1DLayer(
    l,
    num_filters=32,
    filter_size=(5,),
    pad=1,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform(),
)

l = lasagne.layers.DenseLayer(
    l,
    num_units=13,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform(),
)

label = T.matrix()
cost = T.mean((lasagne.layers.get_output(l) - label)**2)

all_params = lasagne.layers.get_all_params(l)

updates = lasagne.updates.adam(
    loss_or_grads=cost,
    params=all_params,
    )

train_fn = theano.function(inputs=[l_in.input_var, label],
                           outputs=[cost],
                           updates=updates)


for i in xrange(2000):
        i = i % 200
	batch = (all_promoters[50*i:50*(i+1)],all_microarrays[50*i:50*(i+1)])
	if i%10==0:
		print "Step %d, cross entropy " %(i)
		print train_fn(batch[0], batch[1])
        train_fn(batch[0], batch[1])


