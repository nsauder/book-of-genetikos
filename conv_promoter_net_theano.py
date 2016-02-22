#nput for this is in the form of a list of tuples, where each tuple has two positions, the first position is an ND.array which is 
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
    num_filters=16,
    filter_size=(3,),
    pad=1,
)

layer_letters = 'ccpcccpcccpcccp'

for layer_type in layer_letters:
    if layer_type == 'c':
            l = lasagne.layers.Conv1DLayer(
                l,
                num_filters=16,
                filter_size=(3,),
                pad=1,
            )

            l = lasagne.layers.batch_norm(l)

    if layer_type == 'p':
            l = lasagne.layers.MaxPool1DLayer(
                    l,
                    pool_size=(2,),
                    stride=2,
            )
l = lasagne.layers.DropoutLayer(l, p=0.5)

l = lasagne.layers.DenseLayer(
    l,
    num_units=13,
    nonlinearity=lasagne.nonlinearities.rectify,
    W=lasagne.init.GlorotUniform(),
)

label = T.matrix()
deterministic_output = lasagne.layers.get_output(l, deterministic=True)
stochastic_output = lasagne.layers.get_output(l, deterministic=False)

def squared_loss(output, label):
        return T.mean((output - label)**2)

all_params = lasagne.layers.get_all_params(l,
                                           trainable=True
)

updates = lasagne.updates.adam(
        loss_or_grads=squared_loss(stochastic_output, label),
        params=all_params,
    )

train_fn = theano.function(inputs=[l_in.input_var, label],
                           outputs=[squared_loss(stochastic_output, label)],
                           updates=updates)


for i in xrange(2000):
        i = i % 200
	batch = (all_promoters[50*i:50*(i+1)],all_microarrays[50*i:50*(i+1)])
	if i%10==0:
		print "Step %d, cross entropy " %(i)
		print train_fn(batch[0], batch[1])
        train_fn(batch[0], batch[1])

valid_fn = theano.function(inputs=[l_in.input_var, label],
                           outputs=[squared_loss(deterministic_output, label)])
        
valid_fn(all_promoters[10000:], all_microarrays[10000:])

def visualize_first_layer_filters(weights):
        assert weights.shape[-2:] == (4,3)

        weights = np.abs(weights)
        maximal_nucleitides = np.argmax(weights, axis=1)

        letters = 'AGCT'
        filters = []

        for seq in maximal_nucleitides:
                filters.append(''.join([letters[nucleitide_num]
                                        for nucleitide_num in list(seq)]))

        return filters
