import logging
import math
import time

import numpy as np
import sklearn.preprocessing as skp
import tensorflow as tf

class SingleNeuralNet():
    '''
    A single neural network with fixed hyperparameters/topology.

    This must run in the same process in which it's created.

    This class should be considered private to this module.

    Args:
        num_params: The number of params.
        layer_dims: The number of nodes in each layer.
        layer_activations: The activation function for each layer.
        train_threshold_ratio: (Relative) loss improvement per train under which training should
            terminate. E.g. 0.1 means we will train (train_epochs at a time) until the improvement
            in loss is less than 0.1 of the loss when that train started (so lower values mean we
            will train for longer). Alternatively, you can think of this as the smallest gradient
            we'll allow before deciding that the loss isn't improving any more.
        batch_size: The training batch size.
        keep_prob: The dropoout keep probability.
        regularisation_coefficient: The regularisation coefficient.
        losses_list: A list to which this object will append training losses.
    '''

    def __init__(self,
                 num_params,
                 layer_dims,
                 layer_activations,
                 train_threshold_ratio,
                 batch_size,
                 keep_prob,
                 regularisation_coefficient,
                 losses_list):
        self.log = logging.getLogger(__name__)
        self.graph = tf.Graph()
        self.tf_session = tf.Session(graph=self.graph)

        if not len(layer_dims) == len(layer_activations):
            self.log.error('len(layer_dims) != len(layer_activations)')
            raise ValueError

        # Hyperparameters for the net. These are all constant.
        self.num_params = num_params
        self.train_threshold_ratio = train_threshold_ratio
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.regularisation_coefficient = regularisation_coefficient

        self.losses_list = losses_list

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
            stddev=0.1
            for (i, (dim, act)) in enumerate(zip(layer_dims, layer_activations)):
                self.weights.append(tf.Variable(
                    tf.random_normal([prev_layer_dim, dim], stddev=stddev),
                    name="weight_"+str(i)))
                self.biases.append(tf.Variable(
                    tf.random_normal([dim], stddev=stddev),
                    name="bias_"+str(i)))
                prev_layer_dim = dim
                prev_h = tf.nn.dropout(
                      act(tf.matmul(prev_h, self.weights[-1]) + self.biases[-1]),
                      keep_prob=self.keep_prob_placeholder)

            # Output node
            self.weights.append(tf.Variable(
                tf.random_normal([prev_layer_dim, 1], stddev=stddev),
                name="weight_out"))
            self.biases.append(tf.Variable(
                tf.random_normal([1], stddev=stddev),
                name="bias_out"))
            self.output_var = tf.matmul(prev_h, self.weights[-1]) + self.biases[-1]

            # Loss function and training
            self.loss_unreg = tf.reduce_mean(tf.reduce_sum(
                tf.square(self.output_var - self.output_placeholder),
                reduction_indices=[1]))
            self.loss_reg = (self.regularisation_coefficient_placeholder
                * tf.reduce_mean([tf.nn.l2_loss(W) for W in self.weights]))
            self.loss_total = self.loss_unreg + self.loss_reg

            # TODO: Set learning rate based on length scale?
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss_total)

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
        return self.tf_session.run(
            [self.loss_total, self.loss_unreg],
            feed_dict={self.input_placeholder: params,
                       self.output_placeholder: [[c] for c in costs],
                       self.regularisation_coefficient_placeholder: self.regularisation_coefficient,
                       })

    def fit(self, params, costs, epochs):
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
        # - set a threshold based on the current loss
        # - train for train_epochs epochs
        # - if the new loss is greater than the threshold then we haven't improved much, so stop
        # - else start from the top
        start = time.time()
        while True:
            threshold = (1 - self.train_threshold_ratio) * self._loss(params, costs)[0]
            self.log.debug("Training with threshold " + str(threshold))
            if threshold == 0:
                break
            tot = 0
            run_start = time.time()
            for i in range(epochs):
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
                self.losses_list.append(l)
                tot += l
                if i % 10 == 0:
                    self.log.debug('Fit neural network with total training cost ' + str(l)
                            + ', with unregularized cost ' + str(ul))
            self.log.debug("Run trained for: " + str(time.time() - run_start))

            (l, ul) = self._loss(params, costs)
            al = tot / float(epochs)
            self.log.debug('Loss ' + str(l) + ', average loss ' + str(al))
            if l > threshold:
                break
        self.log.debug("Total trained for: " + str(time.time() - start))

    def cross_validation_loss(self, params, costs):
        '''
        Returns the loss of the network on a cross validation set.

        Args:
            params (array): array of parameter arrays
            costs (array): array of costs (associated with the corresponding parameters)
        '''
        return self.tf_session.run(self.loss_total,
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
        #runs = 100
        ## Do some runs with dropout, and return the smallest. This is kind of LCB.
        #results = [y[0] for y in self.tf_session.run(self.output_var, feed_dict={
        #        self.input_placeholder: [params] * runs,
        #        self.keep_prob_placeholder: 0.99})]
        #results.sort()
        #return results[int(runs * 0.2)]

    def predict_cost_gradient(self,params):
        '''
        Produces a prediction of the gradient of the cost function at params.

        Returns:
            float : Predicted gradient at parameters
        '''
        return self.tf_session.run(self.output_var_gradient, feed_dict={self.input_placeholder: [params]})[0][0]

class SampledNeuralNet():
    def __init__(self,
                 net_creator,
                 count):
        self.log = logging.getLogger(__name__)
        self.net_creator = net_creator
        self.nets = [self.net_creator() for _ in range(count)]
        self.fit_count = 0

    def _random_net(self):
        return self.nets[np.random.randint(0, len(self.nets))]

    def destroy(self):
        for n in self.nets:
            n.destroy()

    def init(self):
        for n in self.nets:
            n.init()

    def load(self, archive):
        for i, n in enumerate(self.nets):
            #n.load(archive[str(i)])
            n.load(archive)

    def save(self):
        return self.nets[0].save()
        #ret = {}
        #for i, n in enumerate(self.nets):
        #    ret[str(i)] = n.save()
        #return ret

    def fit(self, params, costs, epochs):
        self.fit_count += 1
        # Every per'th fit we clear out a net and re-train it.
        #per = 2
        #if self.fit_count % per == 0:
        #    index = int(self.fit_count / per) % len(self.nets)
        #    self.log.debug("Re-creating net " + str(index))
        #    self.nets[index].destroy()
        #    self.nets[index] = self.net_creator()
        #    self.nets[index].init()

        for n in self.nets:
            n.fit(params, costs, epochs)

    def cross_validation_loss(self, params, costs):
        return np.mean([n.cross_validation_loss(params, costs) for n in self.nets])

    def predict_cost(self,params):
        return self._random_net().predict_cost(params)
        #return np.mean([n.predict_cost(params) for n in self.nets])

    def predict_cost_gradient(self,params):
        return self._random_net().predict_cost_gradient(params)
        #return np.mean([n.predict_cost_gradient(params) for n in self.nets])


class NeuralNetImpl():
    '''
    Neural network implementation. This may actually create multiple neural networks with different
    topologies or hyperparameters, and switch between them based on the data.

    This must run in the same process in which it's created.

    This handles scaling of parameters and costs internally, so there is no need to ensure that these
    values are scaled or normalised in any way.

    All parameters should be considered private to this class. That is, you should only interact with
    this class via the methods documented to be public.

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

        self.epochs = 100

        # Variables for tracking the current state of hyperparameter fitting.
        self.last_hyperfit = 0
        self.last_net_reg = 1e-6

        # The samples used to fit the scalers. When set, this will be a tuple of
        # (params samples, cost samples).
        self.scaler_samples = None

        # The training losses incurred by the network. This is a concatenation of the losses
        # associated with each instance of SingleNeuralNet.
        self.losses_list = []

        self.net = None

    # Private helper methods.

    def _make_net(self, reg):
        '''
        Helper method to create a new net with a specified regularisation coefficient. The net is not
        initialised, so you must call init() or load() on it before any other method.

        Args:
            reg (float): Regularisation coefficient.
        '''
        def gelu_fast(_x):
            return 0.5 * _x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (_x + 0.044715 * tf.pow(_x, 3))))
        def amazing_abs(_x):
            return tf.maximum(1 - tf.abs(_x), 0)
        creator = lambda: SingleNeuralNet(
                    self.num_params,
                    [64]*5, [tf.nn.relu]*5,
                    0.5, # train_threshold_ratio
                    16, # batch_size
                    1., # keep_prob
                    reg,
                    self.losses_list)
        return SampledNeuralNet(creator, 3)

    def _fit_scaler(self):
        if self.scaler_samples is None:
            self.log.error("_fit_scaler() called before samples set")
            raise ValueError
        self._cost_scaler = skp.StandardScaler(with_mean=True, with_std=True)
        self._param_scaler = skp.StandardScaler(with_mean=True, with_std=True)

        self._param_scaler.fit(self.scaler_samples[0])
        # Cost is scalar but numpy doesn't like scalars, so reshape to be a 0D vector instead.
        self._cost_scaler.fit(np.array(self.scaler_samples[1]).reshape(-1,1))

        self._mean_offset = 0

        # Now that the scaler is fitted, calculate the parameters we'll need to unscale gradients.
        # We need to know which unscaled gradient would correspond to a scaled gradient of [1,...1],
        # which we can calculate as the unscaled gradient associated with a scaled rise of 1 and a
        # scaled run of [1,...1]:
        rise_unscaled = (
            self._unscale_cost(np.float64(1))
            - self._unscale_cost(np.float64(0)))
        run_unscaled = (
            self._unscale_params([np.float64(1)]*self.num_params)
            - self._unscale_params([np.float64(0)]*self.num_params))
        self._gradient_unscale = rise_unscaled / run_unscaled

    def _scale_params_and_cost_list(self, params_list_unscaled, cost_list_unscaled):
        params_list_scaled = self._param_scaler.transform(params_list_unscaled)
        # As above, numpy doesn't like scalars, so we need to do some reshaping.
        cost_vector_list_unscaled = np.array(cost_list_unscaled).reshape(-1,1)
        cost_vector_list_scaled = (self._cost_scaler.transform(cost_vector_list_unscaled)
                + self._mean_offset)
        cost_list_scaled = cost_vector_list_scaled[:,0]
        return params_list_scaled, cost_list_scaled

    def _scale_params(self, params_unscaled):
        return self._param_scaler.transform([params_unscaled])[0]

    def _unscale_params(self, params_scaled):
        return self._param_scaler.inverse_transform([params_scaled])[0]

    def _unscale_cost(self, cost_scaled):
        return self._cost_scaler.inverse_transform([[cost_scaled - self._mean_offset]])[0][0]

    def _unscale_gradient(self, gradient_scaled):
        return np.multiply(gradient_scaled, self._gradient_unscale)

    # Public methods.

    def init(self):
        '''
        Initializes the net. You must call exactly one of this and load() before calling any other
        methods.
        '''
        if not self.net is None:
            self.log.error("Called init() when already initialised/loaded")
            raise ValueError

        self.net = self._make_net(self.last_net_reg)
        self.net.init()

    def load(self, archive):
        '''
        Imports the net from an archive dictionary. You must call exactly one of this and init()
        before calling any other methods.

        You must only load a net from an archive if that archive corresponds to a net with the same
        constructor parameters.
        '''
        if not self.net is None:
            self.log.error("Called load() when net already initialised/loaded")
            raise ValueError

        self.last_hyperfit = int(archive['last_hyperfit'])
        self.last_net_reg = float(archive['last_net_reg'])

        self.losses_list = list(archive['losses_list'])

        self.scaler_samples = archive['scaler_samples']
        if not self.scaler_samples is None:
            self._fit_scaler()

        self.net = self._make_net(self.last_net_reg)
        self.net.load(dict(archive['net']))

    def save(self):
        '''
        Exports the net to an archive dictionary.
        '''
        return {'last_hyperfit': self.last_hyperfit,
                'last_net_reg': self.last_net_reg,
                'losses_list': self.losses_list,
                'scaler_samples': self.scaler_samples,
                'net': self.net.save(),
                }

    def fit_neural_net(self, all_params, all_costs):
        '''
        Fits the neural net to the data.

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

        # If we haven't initialised the scaler yet, do it now.
        if self.scaler_samples is None:
            first_fit = True
            self.scaler_samples = (all_params.copy(), all_costs.copy())
            self._fit_scaler()
        else:
            first_fit = False

        all_params, all_costs = self._scale_params_and_cost_list(all_params, all_costs)

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
                    net.fit(train_params, train_costs, self.epochs)
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

        self.net.fit(all_params, all_costs, self.epochs if first_fit else int(self.epochs / 10))

    def predict_cost(self,params):
        '''
        Produces a prediction of cost from the neural net at params.

        Must not be called before fit_neural_net().

        Returns:
            float : Predicted cost at parameters
        '''
        return self._unscale_cost(self.net.predict_cost(self._scale_params(params)))

    def predict_cost_gradient(self,params):
        '''
        Produces a prediction of the gradient of the cost function at params.

        Must not be called before fit_neural_net().

        Returns:
            float : Predicted gradient at parameters
        '''
        return self._unscale_gradient(self.net.predict_cost_gradient(self._scale_params(params)))

    # Public mmethods to be used only for debugging/analysis.

    def get_losses(self):
        '''
        Returns a list of training losses experienced by the network.
        '''
        return self.losses_list
