import logging
import math
import tensorflow as tf
import numpy as np

class SingleNeuralNet():
    '''
    A single neural network with fixed hyperparameters/topology.

    This must run in the same process in which it's created.

    This class should be considered private to this module.

    Args:
        num_params: The number of params.
        layer_dims: The number of nodes in each layer.
        layer_activations: The activation function for each layer.
        train_epochs: Epochs per train.
        batch_size: The training batch size.
        keep_prob: The dropoout keep probability.
        regularisation_coefficient: The regularisation coefficient.
    '''

    def __init__(self,
                 num_params,
                 layer_dims,
                 layer_activations,
                 train_epochs,
                 batch_size,
                 keep_prob,
                 regularisation_coefficient):
        self.log = logging.getLogger(__name__)
        self.graph = tf.Graph()
        self.tf_session = tf.Session(graph=self.graph)

        if not len(layer_dims) == len(layer_activations):
            self.log.error('len(layer_dims) != len(layer_activations)')
            raise ValueError

        # All member variables of this class are constants. The only things that change are the TF
        # variables.

        self.num_params = num_params
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.regularisation_coefficient = regularisation_coefficient

        with self.graph.as_default():
            # Inputs
            self.input_placeholder = tf.placeholder(tf.float32, shape=[None, self.num_params])
            self.output_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
            self.keep_prob_placeholder = tf.placeholder_with_default(1., shape=[])
            self.regularisation_coefficient_placeholder = tf.placeholder_with_default(0., shape=[])

            # Parameters
            self.weights = []
            self.biases = []

            # Input + internal nodes
            # TODO: Use length scale for setting initial weights?
            prev_layer_dim = self.num_params
            prev_h = self.input_placeholder
            for (i, (dim, act)) in enumerate(zip(layer_dims, layer_activations)):
                self.weights.append(tf.Variable(tf.random_normal([prev_layer_dim, dim], stddev=0.1), name="weight_"+str(i)))
                self.biases.append(tf.Variable(tf.random_normal([dim]), name="bias_"+str(i)))
                prev_layer_dim = dim
                prev_h = tf.nn.dropout(
                      act(tf.matmul(prev_h, self.weights[-1]) + self.biases[-1]),
                      keep_prob=self.keep_prob_placeholder)

            # Output node
            self.weights.append(tf.Variable(tf.random_normal([prev_layer_dim, 1]), name="weight_out"))
            self.biases.append(tf.Variable(tf.random_normal([1]), name="bias_out"))
            self.output_var = tf.matmul(prev_h, self.weights[-1]) + self.biases[-1]

            # Loss function and training
            self.loss_func = (
                    tf.reduce_mean(tf.reduce_sum(tf.square(self.output_var - self.output_placeholder),
                                                 reduction_indices=[1]))
                    + self.regularisation_coefficient_placeholder
                            * tf.reduce_mean([tf.nn.l2_loss(W) for W in self.weights]))
            # TODO: Set learning rate based on length scale?
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss_func)

            # Gradient
            self.output_var_gradient = tf.gradients(self.output_var, self.input_placeholder)

            # Initialiser for ... initialising
            self.initialiser = tf.initialize_all_variables()

            # Saver for saving and restoring params
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def destroy(self):
        self.tf_session.close()

    def init(self):
        '''
        Initializes the net.
        '''
        self.tf_session.run(self.initialiser)

    def load(self, archive):
        '''
        Imports the net from an archive dictionary. You must call exactly one of this and init() before calling any other methods.
        '''
        self.log.debug("Loading neural network")
        self.saver.restore(self.tf_session, "./" + str(archive['saver_path']))

    def save(self):
        '''
        Exports the net to an archive dictionary.
        '''
        # TODO: Use a proper timestamped filename, maybe?
        path = self.saver.save(self.tf_session, "net.ckpt")
        self.log.debug("Saving neural network to: " + path)
        return {'saver_path': path}

    def _loss(self, params, costs):
        '''
        Returns the loss and unregularised loss for the given params and costs.
        '''
        return (self.tf_session.run(
            self.loss_func,
            feed_dict={self.input_placeholder: params,
                       self.output_placeholder: [[c] for c in costs],
                       self.regularisation_coefficient_placeholder: self.regularisation_coefficient,
                       }),
            self.tf_session.run(
                self.loss_func,
                feed_dict={self.input_placeholder: params,
                           self.output_placeholder: [[c] for c in costs],
                           self.regularisation_coefficient_placeholder: 0,
                           }))

    def fit(self, params, costs):
        '''
        Fit the neural net to the provided data

        Args:
            params (array): array of parameter arrays
            costs (array): array of costs (associated with the corresponding parameters)
        '''
        self.log.debug('Fitting neural network')
        if len(params) == 0:
            self.log.error('No data provided.')
            raise ValueError
        if not len(params) == len(costs):
            self.log.error("Params and costs must have the same length")
            raise ValueError

        # The general training procedure is as follows:
        # - set a threshold of 80% of the current loss
        # - train for train_epochs epochs
        # - if the new loss is greater than the threshold then we haven't improved much, so stop
        # - else start from the top
        while True:
            threshold = 0.8 * self._loss(params, costs)[0]
            for i in range(self.train_epochs):
                # Split the data into random batches, and train on each batch
                indices = np.random.permutation(len(params))
                for j in range(math.ceil(len(params) / self.batch_size)):
                    batch_indices = indices[j * self.batch_size : (j + 1) * self.batch_size]
                    batch_input = [params[index] for index in batch_indices]
                    batch_output = [[costs[index]] for index in batch_indices]
                    self.tf_session.run(self.train_step,
                                        feed_dict={self.input_placeholder: batch_input,
                                                   self.output_placeholder: batch_output,
                                                   self.regularisation_coefficient_placeholder: self.regularisation_coefficient,
                                                   self.keep_prob_placeholder: self.keep_prob,
                                                   })
            (l, ul) = self._loss(params, costs)
            self.log.debug('Fit neural network with total training cost ' + str(l)
                    + ', with unregularized cost ' + str(ul))
            if l > threshold:
                break
            self.log.debug('Cost decreased by a lot, train again')

    def cross_validation_loss(self, params, costs):
        '''
        Returns the loss of the network on a cross validation set.

        Args:
            params (array): array of parameter arrays
            costs (array): array of costs (associated with the corresponding parameters)
        '''
        return self.tf_session.run(self.loss_func,
                                  feed_dict={self.input_placeholder: params,
                                  self.output_placeholder: [[c] for c in costs],
                                  })

    def predict_cost(self,params):
        '''
        Produces a prediction of cost from the neural net at params.

        Returns:
            float : Predicted cost at parameters
        '''
        return self.tf_session.run(self.output_var, feed_dict={self.input_placeholder: [params]})[0][0]

    def predict_cost_gradient(self,params):
        '''
        Produces a prediction of the gradient of the cost function at params.

        Returns:
            float : Predicted gradient at parameters
        '''
        return self.tf_session.run(self.output_var_gradient, feed_dict={self.input_placeholder: [params]})[0][0]


class NeuralNetImpl():
    '''
    Neural network implementation. This may actually create multiple neural networks with different
    topologies or hyperparameters, and switch between them based on the data.

    This must run in the same process in which it's created.

    Args:
        num_params (int): The number of params.
        fit_hyperparameters (bool): Whether to try to fit the hyperparameters to the data.
    '''

    def __init__(self,
                 num_params = None,
                 fit_hyperparameters = False):

        self.log = logging.getLogger(__name__)
        self.log.debug('Initialising neural network impl')
        if num_params is None:
            self.log.error("num_params must be provided")
            raise ValueError

        # Constants.
        self.num_params = num_params
        self.fit_hyperparameters = fit_hyperparameters

        # Tracking variables. These need to be set when importing and saved when exporting.
        self.last_hyperfit = 0
        self.last_net_reg = 0.01

        self.net = self._make_net(0.01)

    def _make_net(self, reg):
        '''
        Helper method to create a new net with a specified regularisation coefficient. The net is not initialised, so you must call init() or load() on it before any other method.

        Args:
            reg (float): Regularisation coefficient.
        '''
        def gelu_fast(_x):
            return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))
        def amazing_abs(_x):
            return tf.maximum(1 - tf.abs(_x), 0)
        return SingleNeuralNet(
                self.num_params,
                [32, 32, 32, 32],#, 32], # layer_dims
                [tf.abs, tf.nn.relu, tf.abs, tf.nn.relu], # layer_activations
                1000, # train_epochs
                64, # batch_size
                1., # keep_prob
                reg)

    def init(self):
        '''
        Initializes the net.
        '''
        self.net.init()

    def load(self, archive):
        '''
        Imports the net from an archive dictionary. You must call exactly one of this and init() before calling any other methods.
        '''
        self.last_hyperfit = int(archive['last_hyperfit'])
        self.last_net_reg = float(archive['last_net_reg'])

        # Destroy the old net, and replace it with the new loaded one.
        self.net.destroy()
        self.net = self._make_net(self.last_net_reg)
        self.net.load(dict(archive['net']))

    def save(self):
        '''
        Exports the net to an archive dictionary.
        '''
        return {'last_hyperfit': self.last_hyperfit,
                'last_net_reg': self.last_net_reg,
                'net': self.net.save(),
                }

    def fit_neural_net(self, all_params, all_costs):
        '''
        Fits the neural net with the appropriate topology to the data

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

        # TODO: Consider adding some kind of "cost capping". Our NNs will never predict costs going
        # off to infinity, so we could be "wasting" training cost due to totally irrelevant points.
        # If we capped the costs to some value then this might help. Note that this is really just
        # another form of cost scaling.

        if self.fit_hyperparameters:
            # Every 20 fits (starting at 5, just because), re-fit the hyperparameters
            if int(len(all_params + 5) / 20) > self.last_hyperfit:
                self.last_hyperfit = int(len(all_params + 5) / 20)

                # Fit regularisation

                # Split the data into training and cross validation
                cv_size = int(len(all_params) / 10)
                train_params = all_params[:-cv_size]
                train_costs = all_costs[:-cv_size]
                cv_params = all_params[cv_size:]
                cv_costs = all_costs[cv_size:]

                orig_cv_loss = self.net.cross_validation_loss(cv_params, cv_costs)
                best_cv_loss = orig_cv_loss

                self.log.debug("Fitting regularisation, current cv loss=" + str(orig_cv_loss))

                # Try a bunch of different regularisation parameters, switching to a new one if it
                # does significantly better on the cross validation set than the old one.
                for r in [0.001, 0.01, 0.1, 1, 10]:
                    net = self._make_net(r)
                    net.init()
                    net.fit(train_params, train_costs)
                    this_cv_loss = net.cross_validation_loss(cv_params, cv_costs)
                    if this_cv_loss < best_cv_loss and this_cv_loss < 0.1 * orig_cv_loss:
                        best_cv_loss = this_cv_loss
                        self.log.debug("Switching to reg=" + str(r) + ", cv loss=" + str(best_cv_loss))
                        self.last_net_reg = r
                        self.net.destroy()
                        self.net = net
                    else:
                        net.destroy()

                # TODO: Fit depth

        self.net.fit(all_params, all_costs)

    def predict_cost(self,params):
        '''
        Produces a prediction of cost from the neural net at params.

        Returns:
            float : Predicted cost at parameters
        '''
        return self.net.predict_cost(params)

    def predict_cost_gradient(self,params):
        '''
        Produces a prediction of the gradient of the cost function at params.

        Returns:
            float : Predicted gradient at parameters
        '''
        return self.net.predict_cost_gradient(params)
