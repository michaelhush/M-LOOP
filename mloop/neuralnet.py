import datetime
import logging
import math
import os
import time
import base64
from distutils.version import LooseVersion

import mloop.utilities as mlu
import numpy as np
import numpy.random as nr
import sklearn.preprocessing as skp
# Import tensorflow with backwards compatability if version is 2.0.0 or newer.
from tensorflow import __version__ as tf_version
if LooseVersion(tf_version) < LooseVersion('2.0.0'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

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
        learner_archive_dir (Optional str): The path to a directory in which to
            save the neural net. If set to None, then the default value of
            archive_foldername in utilities.py will be used. Default None.
        start_datetime (Optional datetime.datetime): The timestamp to use in the
            filenames when saving the neural nets. If set to None, then the
            time at which the neural net is saved will be used as the timestamp.
            Default KNone.
    '''

    def __init__(self,
                 num_params,
                 layer_dims,
                 layer_activations,
                 train_threshold_ratio,
                 batch_size,
                 keep_prob,
                 regularisation_coefficient,
                 losses_list,
                 learner_archive_dir=None,
                 start_datetime=None):
        self.log = logging.getLogger(__name__)
        start = time.time()

        if learner_archive_dir is None:
            learner_archive_dir = mlu.archive_foldername
        filename_suffix = mlu.generate_filename_suffix(
            'ckpt',
            file_datetime=start_datetime,
            random_bytes=True)
        filename = 'neural_net_archive' + filename_suffix
        self.start_datetime = start_datetime
        self.learner_archive_dir = learner_archive_dir
        self.save_archive_filename = os.path.join(learner_archive_dir, filename)

        self.log.info("Constructing net")
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
            ## Inputs
            self.input_placeholder = tf.placeholder(tf.float32, shape=[None, self.num_params])
            self.output_placeholder = tf.placeholder(tf.float32, shape=[None, 1])
            self.keep_prob_placeholder = tf.placeholder_with_default(1., shape=[])
            self.regularisation_coefficient_placeholder = tf.placeholder_with_default(0., shape=[])

            ## Initialise the network

            weights = []
            biases = []

            # Input + internal nodes
            prev_layer_dim = self.num_params
            bias_stddev=0.5
            for (i, dim) in enumerate(layer_dims):
                weights.append(tf.Variable(
                    tf.random_normal([prev_layer_dim, dim], stddev=1.4/np.sqrt(prev_layer_dim)),
                    name="weight_"+str(i)))
                biases.append(tf.Variable(
                    tf.random_normal([dim], stddev=bias_stddev),
                    name="bias_"+str(i)))
                prev_layer_dim = dim

            # Output node
            weights.append(tf.Variable(
                tf.random_normal([prev_layer_dim, 1], stddev=1.4/np.sqrt(prev_layer_dim)),
                name="weight_out"))
            biases.append(tf.Variable(
                tf.random_normal([1], stddev=bias_stddev),
                name="bias_out"))

            # Get the output var given an input var
            def get_output_var(input_var):
                prev_h = input_var
                for w, b, act in zip(weights[:-1], biases[:-1], layer_activations):
                    prev_h = tf.nn.dropout(
                          act(tf.matmul(prev_h, w) + b),
                          keep_prob=self.keep_prob_placeholder)
                return tf.matmul(prev_h, weights[-1]) + biases[-1]

            ## Define tensors for evaluating the output var and gradient on the full input
            self.output_var = get_output_var(self.input_placeholder)
            self.output_var_gradient = tf.gradients(self.output_var, self.input_placeholder)

            ## Declare common loss functions

            # Get the raw loss given the expected and actual output vars
            def get_loss_raw(expected, actual):
                return tf.reduce_mean(tf.reduce_sum(
                    tf.square(expected - actual),
                    reduction_indices=[1]))

            # Regularisation component of the loss.
            loss_reg = (self.regularisation_coefficient_placeholder
                * tf.reduce_mean([tf.nn.l2_loss(W) for W in weights]))

            ## Define tensors for evaluating the loss on the full input
            self.loss_raw = get_loss_raw(self.output_placeholder, self.output_var)
            self.loss_total = self.loss_raw + loss_reg

            ## Training
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss_total)

            # Initialiser for ... initialising
            self.initialiser = tf.global_variables_initializer()

            # Saver for saving and restoring params
            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        self.log.debug("Finished constructing net in: " + str(time.time() - start))

    def destroy(self):
        self.tf_session.close()

    def init(self):
        '''
        Initializes the net.
        '''
        self.tf_session.run(self.initialiser)

    def load(self, archive, extra_search_dirs=None):
        '''
        Imports the net from an archive dictionary. You must call exactly one of this and init() before calling any other methods.
        
        Args:
            archive (dict): An archive dictionary containg the data from a
                neural net learner archive, typically created using
                get_dict_from_file() from utilities.py.
            extra_search_dirs (Optional list of str): If the neural net archive
                is not in at the location sepcified for it in the archive
                dictionary, then the directories in extra_search_dirs will be
                checked for a file of the saved name. If found, that file will
                be used. If set to None, no other directories will be checked.
                Default None.
        '''
        # Set default value of extra_search_dirs if necessary.
        if extra_search_dirs is None:
            extra_search_dirs = []
        
        # Get the saved filename and construct a list of directories to in which
        # to look for it.
        self.log.info("Loading neural network")
        saver_path = str(archive['saver_path'])
        saved_dirname, filename = os.path.split(saver_path)
        saved_dirname = os.path.join('.', saved_dirname)
        search_dirs = [saved_dirname] + extra_search_dirs
        
        # Check each directory for the file.
        for dirname in search_dirs:
            try:
                full_path = os.path.join(dirname, filename)
                self.saver.restore(self.tf_session, full_path)
                return
            except ValueError:
                pass
        
        # If the method hasn't returned by now then it's run out of places to
        # look.
        message = "Could not find neural net archive {filename}.".format(filename=filename)
        self.log.error(message)
        raise ValueError(message)

    def save(self):
        '''
        Exports the net to an archive dictionary.
        '''
        path = self.saver.save(self.tf_session, self.save_archive_filename)
        self.log.info("Saving neural network to: " + path)
        return {'saver_path': path}

    def _loss(self, params, costs):
        '''
        Returns the loss and unregularised loss for the given params and costs.
        '''
        return self.tf_session.run(
            [self.loss_total, self.loss_raw],
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
        self.log.info('Fitting neural network')
        if len(params) == 0:
            self.log.error('No data provided.')
            raise ValueError
        if not len(params) == len(costs):
            self.log.error("Params and costs must have the same length")
            raise ValueError

        lparams = np.array(params)
        lcosts = np.expand_dims(np.array(costs), axis=1)

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
                for j in range(int(math.ceil(len(params) / self.batch_size))):
                    batch_indices = indices[j * self.batch_size : (j + 1) * self.batch_size]
                    batch_input = lparams[batch_indices]
                    batch_output = lcosts[batch_indices]
                    self.tf_session.run(self.train_step,
                                        feed_dict={self.input_placeholder: batch_input,
                                                   self.output_placeholder: batch_output,
                                                   self.regularisation_coefficient_placeholder: self.regularisation_coefficient,
                                                   self.keep_prob_placeholder: self.keep_prob,
                                                   })
                if i % 10 == 0:
                    (l, ul) = self._loss(params, costs)
                    self.losses_list.append(l)
                    self.log.info('Fit neural network with total training cost ' + str(l)
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
    '''
    A "neural network" that tracks a collection of SingleNeuralNet objects, and predicts the landscape
    by sampling from that collection.

    This must run in the same process in which it's created.

    This class should be considered private to this module.

    Args:
        net_creator: Callable that creates and returns a new SingleNeuralNet.
        count: The number of individual networks to track.
    '''

    def __init__(self,
                 net_creator,
                 count):
        self.log = logging.getLogger(__name__)
        self.net_creator = net_creator
        self.nets = [self.net_creator() for _ in range(count)]
        self.fit_count = 0
        self.opt_net = None

    def _random_net(self):
        return self.nets[np.random.randint(0, len(self.nets))]

    def destroy(self):
        for n in self.nets:
            n.destroy()

    def init(self):
        for n in self.nets:
            n.init()

    def load(self, archive, extra_search_dirs=None):
        for i, n in enumerate(self.nets):
            #n.load(archive[str(i)])
            n.load(archive, extra_search_dirs=extra_search_dirs)

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
        if self.opt_net:
            return self.opt_net.predict_cost(params)
        else:
            return self._random_net().predict_cost(params)
        #return np.mean([n.predict_cost(params) for n in self.nets])

    def predict_cost_gradient(self,params):
        if self.opt_net:
            return self.opt_net.predict_cost_gradient(params)
        else:
            return self._random_net().predict_cost_gradient(params)
        #return np.mean([n.predict_cost_gradient(params) for n in self.nets])

    def start_opt(self):
        self.opt_net = self._random_net()

    def stop_opt(self):
        self.opt_net = None

class NeuralNet():
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
        learner_archive_dir (Optional str): The path to a directory in which to
            save the neural nets. If set to None, then the default value for the
            SingleNeuralNet class will be used. Default None.
        start_datetime (Optional datetime.datetime): The timestamp to use in the
            filenames when saving the neural nets. If set to None, then the
            default value for the SingleNeuralNet class will be used. Default
            None.
    '''

    def __init__(self,
                 num_params = None,
                 fit_hyperparameters = False,
                 learner_archive_dir = None,
                 start_datetime = None):

        self.log = logging.getLogger(__name__)
        self.log.info('Initialising neural network impl')
        if num_params is None:
            self.log.error("num_params must be provided")
            raise ValueError

        # Constants.
        self.num_params = num_params
        self.fit_hyperparameters = fit_hyperparameters
        self.learner_archive_dir = learner_archive_dir
        self.start_datetime = start_datetime

        self.initial_epochs = 100
        self.subsequent_epochs = 20

        # Variables for tracking the current state of hyperparameter fitting.
        self.last_hyperfit = 0
        self.last_net_reg = 1e-8

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
        creator = lambda: SingleNeuralNet(
                    self.num_params,
                    [64]*5, [gelu_fast]*5,
                    0.2, # train_threshold_ratio
                    16, # batch_size
                    1., # keep_prob
                    reg,
                    self.losses_list,
                    learner_archive_dir=self.learner_archive_dir,
                    start_datetime=self.start_datetime)
        return SampledNeuralNet(creator, 1)

    def _fit_scaler(self):
        '''
        Fits the cost and param scalers based on the scaler_samples member variable.
        '''
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

    def load(self, archive, extra_search_dirs=None):
        '''
        Imports the net from an archive dictionary. You must call exactly one of this and init()
        before calling any other methods.

        You must only load a net from an archive if that archive corresponds to a net with the same
        constructor parameters.
        
        Args:
            archive (dict): An archive dictionary containg the data from a
                neural net learner archive, typically created using
                get_dict_from_file() from utilities.py.
            extra_search_dirs (Optional list of str): If the neural net archive
                is not in at the location sepcified for it in the archive
                dictionary, then the directories in extra_search_dirs will be
                checked for a file of the saved name. If found, that file will
                be used. If set to None, no other directories will be checked.
                Default None.
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
        self.net.load(dict(archive['net']), extra_search_dirs=extra_search_dirs)

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

    def destroy(self):
        '''
        Destroys the net.
        '''
        if not self.net is None:
            self.net.destroy()

    def fit_neural_net(self, all_params, all_costs):
        '''
        Fits the neural net to the data.

        Args:
            all_params (array): array of all parameter arrays
            all_costs (array): array of costs (associated with the corresponding parameters)
        '''
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

        if self.fit_hyperparameters:
            # Every 20 runs, re-fit the hyperparameters.
            n_fits = len(all_params)
            n_hyperfit = int(n_fits / 20.0)  # int() rounds down.
            if n_hyperfit > self.last_hyperfit:
                self.last_hyperfit = n_hyperfit

                # Fit regularisation

                # Split the data into training and cross validation
                training_fraction = 0.9
                split_index = int(training_fraction * len(all_params))
                train_params = all_params[:split_index]
                train_costs = all_costs[:split_index]
                cv_params = all_params[split_index:]
                cv_costs = all_costs[split_index:]

                orig_cv_loss = self.net.cross_validation_loss(cv_params, cv_costs)
                best_cv_loss = orig_cv_loss

                self.log.debug("Fitting regularisation, current cv loss=" + str(orig_cv_loss))

                # Try a bunch of different regularisation parameters, switching to a new one if it
                # does significantly better on the cross validation set than the old one.
                for r in [0.001, 0.01, 0.1, 1, 10]:
                    net = self._make_net(r)
                    net.init()
                    net.fit(train_params, train_costs, self.initial_epochs)
                    this_cv_loss = net.cross_validation_loss(cv_params, cv_costs)
                    if this_cv_loss < best_cv_loss and this_cv_loss < 0.1 * orig_cv_loss:
                        best_cv_loss = this_cv_loss
                        self.log.debug("Switching to reg=" + str(r) + ", cv loss=" + str(best_cv_loss))
                        self.last_net_reg = r
                        self.net.destroy()
                        self.net = net
                    else:
                        net.destroy()

        self.net.fit(
                all_params,
                all_costs,
                self.initial_epochs if first_fit else self.subsequent_epochs)

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

    def start_opt(self):
        '''
        Starts an optimisation run. Until stop_opt() is called, predict_cost() and
        predict_cost_gradient() will return consistent values.
        '''
        self.net.start_opt()

    def stop_opt(self):
        '''
        Stops an optimisation run.
        '''
        self.net.stop_opt()

    # Public mmethods to be used only for debugging/analysis.

    def get_losses(self):
        '''
        Returns a list of training losses experienced by the network.
        '''
        return self.losses_list
