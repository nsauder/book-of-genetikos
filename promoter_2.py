"""The point of this program is to be able to separate a 40 or 20 nucleotide sequence that is a promoter
from a randomly generated DNA sequence of the same length. 

A secondary goal is to backtrack from the learned network, which can do this efficiently, to the motifs in the 
promoter sequence which it has picked up on"""

import numpy as np
import random


def open_FASTA(filename):
	##This turns a FASTA list of promoter sequences into a list of tuples with the first position 
	##Returns a list of tuples with the NDarray for input in the first position and an array of [1] in the second position
	##The NDarray is 4x the length of the promoter and codes for the bases in binary fashion (i.e. if position 1 is an a,
	##array[0] is 1, if position 2 is a c, array[1] is 1, etc. 
	master_list = []
	with open(filename) as f:
		#Note the below have an \n character at the end
		lines = f.readlines()
	return lines




def gen_vectorized(sequence, y):

        master_list = []
        
        length_line = len(sequence)
        length_seq = len(sequence[0])-1
        
        array_y = np.zeros((1,1))
        array_y[0] = y

	for i in xrange(1,length_line):
		##This turns all the input sequences into NDarrays 4x the length
		array = np.zeros((length_seq*4,1))
		for j in xrange(length_seq):
			base = sequence[i][j]
			if base == 'A':
				array[j*4] = 1
			if base == 'C':
				array[j*4+1] = 1
			if base == 'G':
				array[j*4+2] = 1
			if base == 'T':
				array[j*4+3] = 1
		                master_list.append((array, array_y))

        return master_list
                                

def clean_FASTA(lines, y=1):
        """
        cleans fasta files 
        """


	master_list = []
        # pull out sequences and remove random metadata
        # TODO HACK
        sequences = [elem[0][:-1] + elem[1][:-1] for
                     elem in zip(lines[1::3], lines[2::3])]
        
                                
	return gen_vectorized(sequences, y)




def make_random_sequence(length_sequence, length_list):
	##Random sequence generator for comparison
	list_of_seq = []
	for i in xrange(length_list):	
		b = ''
		for x in range(length_sequence+1):
			b+=random.choice('ACGT')
		list_of_seq.append(b)

	return gen_vectorized(list_of_seq,0)

"""Neural net specific below"""
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

image_gradients = []

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self,
            training_data,
            epochs,
            mini_batch_size,
            eta=0.1,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
       # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp 
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        global image_gradients
        image_gradients.append((x, np.dot(self.weights[0].transpose(), delta)))
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
        num_right = sum(int(np.rint(xf)[0] == np.rint(y)[0])
                       for (xf, y) in test_results)
        accuracy = num_right / (len(test_data)+0.0)
        print accuracy
        return accuracy

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

raw_data = clean_FASTA(open_FASTA('100_length_sequences.pl')[:2000])

seq_length = len(raw_data[0][0])/4
num_promoters = len(raw_data)

tr_promoters = raw_data[:int(num_promoters*0.7)]
te_promoters = raw_data[int(num_promoters*0.7):]
tr_random_data = make_random_sequence(seq_length, len(tr_promoters))
te_random_data = make_random_sequence(seq_length, len(te_promoters))
net = Network([seq_length*4,10,1])
net.SGD(tr_promoters+tr_random_data,
        100,
        10,
        eta=.05,
        test_data=te_promoters+te_random_data)

nucleitides = 'ACGT'

#	open ()
	##Turns a FASTA file into a list of promoter sequences
#def 
	##This turns a list of promoter sequences into an NDarray with 4 input neurons
	##per position, representing the different possible base choices at that position
	##Returns a tuple with the NDarray for input in the first position and an array of [1] in the second position


##Random library generator (length)
##When called with **base pair** length returns a vector 4x that length with a randomly generated sequence

##Neural net library

##Also build 5x convolutional net

##To run, put in inputs with labels (so 160 or 80 long inputs with corresponding all '1'), 
