import logging
import math
import tensorflow as tf
import numpy as np

class NeuralNetImpl():
    '''
    Neural network implementation.

    Args:
        num_params (int): The number of params.
    '''

    def __init__(self,
                 num_params = None):

        self.log = logging.getLogger(__name__)
        self.log.debug('Initialising neural network impl')
        if num_params is None:
            self.log.error("num_params must be provided")
            raise ValueError
        self.num_params = num_params

        self.tf_session = tf.InteractiveSession()

        # Initial hyperparameters
        self.num_layers = 1
        self.layer_dim = 128
        self.train_epochs = 300
        self.batch_size = 64

        # Inputs
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, self.num_params])
        self.output_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
        self.keep_prob = tf.placeholder_with_default(1., shape=[])
        self.regularisation_coefficient = tf.placeholder_with_default(0., shape=[])

        self._create_neural_net()

    def _create_neural_net(self):
        '''
        Creates the neural net with topology specified by the current hyperparameters.

        '''
        self.log.debug('Creating neural network')
        # Forget about any old weights/biases
        self.weights = []
        self.biases = []

        # Input + internal nodes
        # TODO: Use length scale for setting initial weights?
        prev_layer_dim = self.num_params
        prev_h = self.input_placeholder
        for dim in [self.layer_dim] * self.num_layers:
            self.weights.append(tf.Variable(tf.random_normal([prev_layer_dim, dim], stddev=0.1)))
            self.biases.append(tf.Variable(tf.random_normal([dim])))
            prev_layer_dim = dim
            prev_h = tf.nn.dropout(
                  tf.nn.sigmoid(tf.matmul(prev_h, self.weights[-1]) + self.biases[-1]),
                  keep_prob=self.keep_prob)

        # Output node
        self.weights.append(tf.Variable(tf.random_normal([prev_layer_dim, 1])))
        self.biases.append(tf.Variable(tf.random_normal([1])))
        self.output_var = tf.matmul(prev_h, self.weights[-1]) + self.biases[-1]

        # Loss function and training
        loss_func = (
                tf.reduce_mean(tf.reduce_sum(tf.square(self.output_var - self.output_placeholder),
                                             reduction_indices=[1]))
                + self.regularisation_coefficient * sum([tf.nn.l2_loss(W) for W in self.weights]))
        self.train_step = tf.train.AdamOptimizer().minimize(loss_func)

        self.tf_session.run(tf.initialize_all_variables())

    def fit_neural_net(self, all_params, all_costs):
        '''
        Determine the appropriate number of layers for the NN given the data.

        Fit the Neural Net with the appropriate topology to the data

        Args:
            all_params (array): array of all parameter arrays
            all_costs (array): array of costs (associated with the corresponding parameters)
        '''
        self.log.debug('Fitting neural network')
        if len(all_params) == 0:
            self.log.error('No data provided.')
            raise ValueError
        if not len(all_params) == len(all_costs):
            self.log.error("Params and costs must have the same length")
            raise ValueError

        # TODO: Fit hyperparameters.

        for i in range(self.train_epochs):
            # Split the data into random batches, and train on each batch
            all_indices = np.random.permutation(len(all_params))
            for j in range(math.ceil(len(all_params) / self.batch_size)):
                batch_indices = all_indices[j * self.batch_size : (j + 1) * self.batch_size]
                batch_input = [all_params[index] for index in batch_indices]
                batch_output = [[all_costs[index]] for index in batch_indices]
                self.tf_session.run(self.train_step,
                                    feed_dict={self.input_placeholder: batch_input,
                                               self.output_placeholder: batch_output})

    def predict_cost(self,params):
        '''
        Produces a prediction of cost from the neural net at params.

        Returns:
            float : Predicted cost at parameters
        '''
        return self.tf_session.run(self.output_var, feed_dict={self.input_placeholder: [params]})[0][0]
