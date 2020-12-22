'''
Module of learners used to determine what parameters to try next given previous cost evaluations.

Each learner is created and controlled by a controller.
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import threading
import numpy as np
import random
import numpy.random as nr
import scipy.optimize as so
import logging
import datetime
import os
import mloop.utilities as mlu
import multiprocessing as mp

import sklearn.gaussian_process as skg
import sklearn.gaussian_process.kernels as skk
import sklearn.preprocessing as skp

from mloop import __version__
import mloop.neuralnet as mlnn
#Lazy import of scikit-learn and tensorflow

learner_thread_count = 0
default_learner_archive_filename = 'learner_archive'
default_learner_archive_file_type = 'txt'

class LearnerInterrupt(Exception):
    '''
    Exception that is raised when the learner is ended with the end flag or event.
    '''
    def __init__(self):
        '''
        Create LearnerInterrupt.
        '''
        super(LearnerInterrupt,self).__init__()


class Learner():
    '''
    Base class for all learners. Contains default boundaries and some useful functions that all learners use.

    The class that inherits from this class should also inherit from threading.Thread or multiprocessing.Process, depending if you need the learner to be a genuine parallel process or not.

    Keyword Args:
        num_params (Optional [int]): The number of parameters to be optimized. If None defaults to 1. Default None.
        min_boundary (Optional [array]): Array with minimum values allowed for each parameter. Note if certain values have no minimum value you can set them to -inf for example [-1, 2, float('-inf')] is a valid min_boundary. If None sets all the boundaries to '-1'. Default None.
        max_boundary (Optional [array]): Array with maximum values allowed for each parameter. Note if certain values have no maximum value you can set them to +inf for example [0, float('inf'),3,-12] is a valid max_boundary. If None sets all the boundaries to '1'. Default None.
        learner_archive_filename (Optional [string]): Name for python archive of the learners current state. If None, no archive is saved. Default None. But this is typically overloaded by the child class.
        learner_archive_file_type (Optional [string]):  File type for archive. Can be either 'txt' a human readable text file, 'pkl' a python dill file, 'mat' a matlab file or None if there is no archive. Default 'mat'.
        log_level (Optional [int]): Level for the learners logger. If None, set to warning. Default None.
        start_datetime (Optional [datetime]): Start date time, if None, is automatically generated.
        param_names (Optional [list of str]): A list of names of the parameters for use e.g. in plot legends. Number of elements must equal num_params. If None, each name will be set to an empty sting. Default None.

    Attributes:
        params_out_queue (queue): Queue for parameters created by learner.
        costs_in_queue (queue): Queue for costs to be used by learner.
        end_event (event): Event to trigger end of learner.
        all_params (array): Array containing all parameters sent to learner.
        all_costs (array): Array containing all costs sent to learner.
        all_uncers (array): Array containing all uncertainties sent to learner.
        bad_run_indexs (list): list of indexes to all runs that were marked as
            bad.
    '''

    def __init__(self,
                 num_params=None,
                 min_boundary=None,
                 max_boundary=None,
                 learner_archive_filename=default_learner_archive_filename,
                 learner_archive_file_type=default_learner_archive_file_type,
                 start_datetime=None,
                 param_names=None,
                 **kwargs):

        super(Learner,self).__init__()

        self._prepare_logger()

        self.learner_wait=float(1)

        self.remaining_kwargs = kwargs

        self.params_out_queue = mp.Queue()
        self.costs_in_queue = mp.Queue()
        self.end_event = mp.Event()

        if num_params is None:
            self.log.warning('num_params not provided, setting to default value of 1.')
            self.num_params = 1
        else:
            self.num_params = int(num_params)
        if self.num_params <= 0:
            self.log.error('Number of parameters must be greater than zero:' + repr(self.num_params))
            raise ValueError
        if min_boundary is None:
            self.min_boundary = np.full((self.num_params,), -1.0)
        else:
            self.min_boundary = np.array(min_boundary, dtype=np.float)
        if self.min_boundary.shape != (self.num_params,):
            self.log.error('min_boundary array the wrong shape:' + repr(self.min_boundary.shape))
            raise ValueError
        if max_boundary is None:
            self.max_boundary = np.full((self.num_params,), 1.0)
        else:
            self.max_boundary = np.array(max_boundary, dtype=np.float)
        if self.max_boundary.shape != (self.num_params,):
            self.log.error('max_boundary array the wrong shape:' + self.min_boundary.shape)
            raise ValueError
        self.diff_boundary = self.max_boundary - self.min_boundary
        if not np.all(self.diff_boundary>0.0):
            self.log.error('All elements of max_boundary are not larger than min_boundary')
            raise ValueError
        if start_datetime is None:
            self.start_datetime = datetime.datetime.now()
        else:
            self.start_datetime = start_datetime
        if mlu.check_file_type_supported(learner_archive_file_type):
            self.learner_archive_file_type = learner_archive_file_type
        else:
            self.log.error('File in type is not supported:' + learner_archive_file_type)
            raise ValueError
        if learner_archive_filename is None:
            self.learner_archive_filename = None
        else:
            # Store self.learner_archive_filename without any path, but include
            # any path components in learner_archive_filename when constructing
            # the full path.
            learner_archive_filename = str(learner_archive_filename)
            self.learner_archive_filename = os.path.basename(learner_archive_filename)
            filename_suffix = mlu.generate_filename_suffix(
                self.learner_archive_file_type,
                file_datetime=self.start_datetime,
            )
            filename = learner_archive_filename + filename_suffix
            self.total_archive_filename = os.path.join(mlu.archive_foldername, filename)

            # Include any path info from learner_archive_filename when creating
            # directory for archive files.
            learner_archive_dir = os.path.dirname(self.total_archive_filename)
            self.learner_archive_dir = learner_archive_dir
            if not os.path.exists(learner_archive_dir):
                os.makedirs(learner_archive_dir)
        # Interpret/check param_names.
        if param_names is None:
            self.param_names = [''] * self.num_params
        else:
            self.param_names = param_names
        # Ensure that there are the correct number of entries.
        if len(self.param_names) != self.num_params:
            message = ('param_names has {n_names} elements but there are '
                       '{n_params} parameters.').format(
                           n_names=len(self.param_names),
                           n_params=self.num_params)
            self.log.error(message)
            raise ValueError(message)
        # Ensure that all of the entries are strings.
        self.param_names = [str(name) for name in self.param_names]

        #Storage variables, archived
        self.all_params = np.array([], dtype=float)
        self.all_costs = np.array([], dtype=float)
        self.all_uncers = np.array([], dtype=float)
        self.bad_run_indexs = []

        self.archive_dict = {'mloop_version':__version__,
                             'archive_type':'learner',
                             'num_params':self.num_params,
                             'min_boundary':self.min_boundary,
                             'max_boundary':self.max_boundary,
                             'start_datetime':mlu.datetime_to_string(self.start_datetime),
                             'param_names':self.param_names}

        self.log.debug('Learner init completed.')

    def _prepare_logger(self):
        '''
        Prepare the logger.

        If `self.log` already exists, then this method silently returns without
        changing anything.
        '''
        if not hasattr(self, 'log'):
            global learner_thread_count
            learner_thread_count += 1
            name = __name__ + '.' + str(learner_thread_count)
            self.log = logging.getLogger(name)

    def check_num_params(self,param):
        '''
        Check the number of parameters is right.
        '''
        return param.shape == (self.num_params,)

    def check_in_boundary(self,param):
        '''
        Check given parameters are within stored boundaries.

        Args:
            param (array): array of parameters

        Returns:
            bool : True if the parameters are within boundaries, False otherwise.
        '''
        param = np.array(param)
        testbool = np.all(param >= self.min_boundary) and np.all(param <= self.max_boundary)
        return testbool

    def check_in_diff_boundary(self,param):
        '''
        Check given distances are less than the boundaries.

        Args:
            param (array): array of distances

        Returns:
            bool : True if the distances are smaller or equal to boundaries, False otherwise.
        '''
        param = np.array(param)
        testbool = np.all(param<=self.diff_boundary)
        return testbool

    def put_params_and_get_cost(self, params, **kwargs):
        '''
        Send parameters to queue and whatever additional keywords.

        Also saves sent and received variables in appropriate storage arrays.

        Args:
            params (array) : array of values to be sent to file

        Returns:
            cost from the cost queue
        '''
        #self.log.debug('Learner params='+repr(params))
        if not self.check_num_params(params):
            self.log.error('Incorrect number of parameters sent to queue. Params' + repr(params))
            raise ValueError
        if not self.check_in_boundary(params):
            self.log.warning('Parameters sent to queue are not within boundaries. Params:' + repr(params))
        #self.log.debug('Learner puts params.')
        self.params_out_queue.put(params)
        #self.log.debug('Learner waiting for costs.')
        self.save_archive()
        while not self.end_event.is_set():
            try:
                message = self.costs_in_queue.get(True, self.learner_wait)
            except mlu.empty_exception:
                continue
            else:
                break
        else:
            self.log.debug('Learner end signal received. Ending')
            # Check for one more message which may have been was lost in a race
            # with the end_event being set.
            try:
                message = self.costs_in_queue.get(True, self.learner_wait)
            except mlu.empty_exception:
                pass
            else:
                params, cost, uncer, bad = self._parse_cost_message(message)
                self._update_run_data_attributes(params, cost, uncer, bad)
            raise LearnerInterrupt
        #self.log.debug('Learner cost='+repr(cost))
        # Record values.
        params, cost, uncer, bad = self._parse_cost_message(message)
        self._update_run_data_attributes(params, cost, uncer, bad)
        return cost

    def _parse_cost_message(self, message):
        '''
        Parse a message sent from the controller via `self.costs_in_queue`.

        Args:
            message (tuple): A tuple put in `self.costs_in_queue` by the
                controller. It should be of the form
                `(params, cost, uncer, bad)` where `params` is an array
                specifying the parameter values used, `cost` is the measured
                cost for those parameter values, `uncer` is the uncertainty
                measured for those parameter values, and `bad` is a boolean
                indicating whether the run was bad.

        Raises:
            ValueError: A `ValueError` is raised if the number of parameters in
                the provided `params` doesn't match `self.num_params`.

        Returns:
            tuple: A tuple of the form `(params, cost, uncer, bad)`. For more
                information on the meaning of those parameters, see the entry
                for the `message` argument above.
        '''
        params, cost, uncer, bad = message
        params = np.array(params, dtype=float)
        if not self.check_num_params(params):
            msg = ('Expected {num_params} parameters, but parameters were: '
                   '{params}.').format(
                       num_params=self.num_params,
                       params=repr(params),
                   )
            self.log.error(msg)
            raise ValueError(msg)
        if not self.check_in_boundary(params):
            self.log.warning('Parameters provided to learner not in boundaries:' + repr(params))
        cost = float(cost)
        uncer = float(uncer)
        if uncer < 0:
            self.log.error('Provided uncertainty must be larger or equal to zero:' + repr(uncer))
        return params, cost, uncer, bad

    def _update_run_data_attributes(self, params, cost, uncer, bad):
        '''
        Update attributes that store the results returned by the controller.

        Args:
            params (array): Array of control parameter values.
            cost (float): The cost measured for `params`.
            uncer (float): The uncertainty measured for `params`.
            bad (bool): Whether or not the run was bad.
        '''
        if self.all_params.size==0:
            self.all_params = np.array([params], dtype=float)
            self.all_costs = np.array([cost], dtype=float)
            self.all_uncers = np.array([uncer], dtype=float)
        else:
            # params
            params_array = np.array([params], dtype=float)
            self.all_params = np.append(self.all_params, params_array, axis=0)
            # cost
            cost_array = np.array([cost], dtype=float)
            self.all_costs = np.append(self.all_costs, cost_array, axis=0)
            # uncer
            uncer_array = np.array([uncer], dtype=float)
            self.all_uncers = np.append(self.all_uncers, uncer_array, axis=0)
        if bad:
            cost_index = len(self.all_costs) - 1
            self.bad_run_indexs.append(cost_index)

    def save_archive(self):
        '''
        Save the archive associated with the learner class. Only occurs if the filename for the archive is not None. Saves with the format previously set.
        '''
        self.update_archive()
        if self.learner_archive_filename is not None:
            mlu.save_dict_to_file(self.archive_dict, self.total_archive_filename, self.learner_archive_file_type)

    def update_archive(self):
        '''
        Update the dictionary of parameters and values to save to the archive.

        Child classes should call this method and also updated
        `self.archive_dict` with any other parameters and values that need to be
        saved to the learner archive.
        '''
        new_values_dict = {
            'all_params':self.all_params,
            'all_costs':self.all_costs,
            'all_uncers':self.all_uncers,
            'bad_run_indexs':self.bad_run_indexs,
        }
        self.archive_dict.update(new_values_dict)

    def _set_trust_region(self,trust_region):
        '''
        Sets trust region properties for learner that have this. Common function for learners with trust regions.

        Args:
            trust_region (float or array): Property defines the trust region.
        '''
        if trust_region is None:
            self.trust_region = float('nan')
            self.has_trust_region = False
        else:
            self.has_trust_region = True
            if isinstance(trust_region , float):
                if trust_region > 0 and trust_region < 1:
                    self.trust_region = trust_region * self.diff_boundary
                else:
                    self.log.error('Trust region, when a float, must be between 0 and 1: '+repr(trust_region))
                    raise ValueError
            else:
                self.trust_region = np.array(trust_region, dtype=float)

        if self.has_trust_region:
            if not self.check_num_params(self.trust_region):
                self.log.error('Shape of the trust_region does not match the number of parameters:' + repr(self.trust_region))
                raise ValueError
            if not np.all(self.trust_region>0):
                self.log.error('All trust_region values must be positive:' + repr(self.trust_region))
                raise ValueError
            if not self.check_in_diff_boundary(self.trust_region):
                self.log.error('The trust_region must be smaller than the range of the boundaries:' + repr(self.trust_region))
                raise ValueError

    def _shut_down(self):
        '''
        Shut down and perform one final save of learner.
        '''
        self.log.debug('Performing shut down of learner.')
        self.save_archive()


class RandomLearner(Learner, threading.Thread):
    '''
    Random learner. Simply generates new parameters randomly with a uniform distribution over the boundaries. Learner is perhaps a misnomer for this class.

    Args:
        **kwargs (Optional dict): Other values to be passed to Learner.

    Keyword Args:
        min_boundary (Optional [array]): If set to None, overrides default learner values and sets it to a set of value 0. Default None.
        max_boundary (Optional [array]): If set to None overides default learner values and sets it to an array of value 1. Default None.
        first_params (Optional [array]): The first parameters to test. If None will just randomly sample the initial condition.
        trust_region (Optional [float or array]): The trust region defines the maximum distance the learner will travel from the current best set of parameters. If None, the learner will search everywhere. If a float, this number must be between 0 and 1 and defines maximum distance the learner will venture as a percentage of the boundaries. If it is an array, it must have the same size as the number of parameters and the numbers define the maximum absolute distance that can be moved along each direction.
    '''

    def __init__(self,
                 trust_region=None,
                 first_params=None,
                 **kwargs):

        super(RandomLearner,self).__init__(**kwargs)

        if not np.all(self.diff_boundary>0.0):
            self.log.error('All elements of max_boundary are not larger than min_boundary')
            raise ValueError
        if ((np.all(np.isfinite(self.min_boundary))&np.all(np.isfinite(self.max_boundary)))==False):
            self.log.error('Minimum and/or maximum boundaries are NaN or inf. Must both be finite for random learner. Min boundary:' + repr(self.min_boundary) +'. Max boundary:' + repr(self.max_boundary))
            raise ValueError
        if first_params is None:
            self.first_params = None
        else:
            self.first_params = np.array(first_params, dtype=float)
            if not self.check_num_params(self.first_params):
                self.log.error('first_params has the wrong number of parameters:' + repr(self.first_params))
                raise ValueError
            if not self.check_in_boundary(self.first_params):
                self.log.error('first_params is not in the boundary:' + repr(self.first_params))
                raise ValueError

        # Keep track of best parameters to implement trust region.
        self.best_cost = None
        self.best_parameters = None

        self._set_trust_region(trust_region)

        self.archive_dict.update({'archive_type':'random_learner'})

        self.log.debug('Random learner init completed.')

    def run(self):
        '''
        Puts the next parameters on the queue which are randomly picked from a uniform distribution between the minimum and maximum boundaries when a cost is added to the cost queue.
        '''
        self.log.debug('Starting Random Learner')
        if self.first_params is None:
            next_params = self.min_boundary + nr.rand(self.num_params) * self.diff_boundary
        else:
            next_params = self.first_params
        while not self.end_event.is_set():
            try:
                cost = self.put_params_and_get_cost(next_params)
            except LearnerInterrupt:
                break
            else:
                # Update best parameters if necessary.
                if self.best_cost is None or cost < self.best_cost:
                    self.best_cost = cost
                    self.best_params = self.all_params[-1]
                if self.has_trust_region:
                    temp_min = np.maximum(self.min_boundary, self.best_params - self.trust_region)
                    temp_max = np.minimum(self.max_boundary, self.best_params + self.trust_region)
                    next_params = temp_min + nr.rand(self.num_params) * (temp_max - temp_min)
                else:
                    next_params =  self.min_boundary + nr.rand(self.num_params) * self.diff_boundary

        self._shut_down()
        self.log.debug('Ended Random Learner')

class NelderMeadLearner(Learner, threading.Thread):
    '''
    Nelder–Mead learner. Executes the Nelder–Mead learner algorithm and stores the needed simplex to estimate the next points.

    Args:
        params_out_queue (queue): Queue for parameters from controller.
        costs_in_queue (queue): Queue for costs for nelder learner. The queue should be populated with cost (float) corresponding to the last parameter sent from the Nelder–Mead Learner. Can be a float('inf') if it was a bad run.
        end_event (event): Event to trigger end of learner.

    Keyword Args:
        initial_simplex_corner (Optional [array]): Array for the initial set of parameters, which is the lowest corner of the initial simplex. If None the initial parameters are randomly sampled if the boundary conditions are provided, or all are set to 0 if boundary conditions are not provided.
        initial_simplex_displacements (Optional [array]): Array used to construct the initial simplex. Each array is the positive displacement of the parameters above the init_params. If None and there are no boundary conditions, all are set to 1. If None and there are boundary conditions assumes the initial conditions are scaled. Default None.
        initial_simplex_scale (Optional [float]): Creates a simplex using a the boundary conditions and the scaling factor provided. If None uses the init_simplex if provided. If None and init_simplex is not provided, but boundary conditions are is set to 0.5. Default None.

    Attributes:
        init_simplex_corner (array): Parameters for the corner of the initial simple used.
        init_simplex_disp (array): Parameters for the displacements about the simplex corner used to create the initial simple.
        simplex_params (array): Parameters of the current simplex
        simplex_costs (array): Costs associated with the parameters of the current simplex

    '''

    def __init__(self,
                 initial_simplex_corner=None,
                 initial_simplex_displacements=None,
                 initial_simplex_scale=None,
                 **kwargs):

        super(NelderMeadLearner,self).__init__(**kwargs)

        self.num_boundary_hits = 0
        self.rho = 1
        self.chi = 2
        self.psi = 0.5
        self.sigma = 0.5

        if initial_simplex_displacements is None and initial_simplex_scale is None:
            self.init_simplex_disp = self.diff_boundary * 0.6
            self.init_simplex_disp[self.init_simplex_disp==float('inf')] =  1
        elif initial_simplex_scale is not None:
            initial_simplex_scale = float(initial_simplex_scale)
            if initial_simplex_scale>1 or initial_simplex_scale<=0:
                self.log.error('initial_simplex_scale must be bigger than 0 and less than 1')
                raise ValueError
            self.init_simplex_disp = self.diff_boundary * initial_simplex_scale
        elif initial_simplex_displacements is not None:
            self.init_simplex_disp = np.array(initial_simplex_displacements, dtype=float)
        else:
            self.log.error('initial_simplex_displacements and initial_simplex_scale can not both be provided simultaneous.')

        if not self.check_num_params(self.init_simplex_disp):
            self.log.error('There is the wrong number of elements in the initial simplex displacement:' + repr(self.init_simplex_disp))
            raise ValueError
        if np.any(self.init_simplex_disp<0):
            self.log.error('initial simplex displacements generated from configuration must all be positive')
            raise ValueError
        if not self.check_in_diff_boundary(self.init_simplex_disp):
            self.log.error('Initial simplex displacements must be within boundaries. init_simplex_disp:'+ repr(self.init_simplex_disp) + '. diff_boundary:' +repr(self.diff_boundary))
            raise ValueError

        if initial_simplex_corner is None:
            diff_roll = (self.diff_boundary - self.init_simplex_disp) * nr.rand(self.num_params)
            diff_roll[diff_roll==float('+inf')]= 0
            self.init_simplex_corner = self.min_boundary
            self.init_simplex_corner[self.init_simplex_corner==float('-inf')]=0
            self.init_simplex_corner += diff_roll
        else:
            self.init_simplex_corner = np.array(initial_simplex_corner, dtype=float)

        if not self.check_num_params(self.init_simplex_corner):
            self.log.error('There is the wrong number of elements in the initial simplex corner:' + repr(self.init_simplex_corner))
        if not self.check_in_boundary(self.init_simplex_corner):
            self.log.error('Initial simplex corner outside of boundaries:' + repr(self.init_simplex_corner))
            raise ValueError

        if not np.all(np.isfinite(self.init_simplex_corner + self.init_simplex_disp)):
            self.log.error('Initial simplex corner and simplex are not finite numbers. init_simplex_corner:'+ repr(self.init_simplex_corner) + '. init_simplex_disp:' +repr(self.init_simplex_disp))
            raise ValueError
        if not self.check_in_boundary(self.init_simplex_corner + self.init_simplex_disp):
            self.log.error('Largest boundary of simplex not inside the boundaries:' + repr(self.init_simplex_corner + self.init_simplex_disp))
            raise ValueError

        self.simplex_params = np.zeros((self.num_params + 1, self.num_params), dtype=float)
        self.simplex_costs = np.zeros((self.num_params + 1,), dtype=float)

        self.archive_dict.update({'archive_type':'nelder_mead_learner',
                                  'initial_simplex_corner':self.init_simplex_corner,
                                  'initial_simplex_displacements':self.init_simplex_disp})

        self.log.debug('Nelder–Mead learner init completed.')

    def run(self):
        '''
        Runs Nelder–Mead algorithm to produce new parameters given costs, until end signal is given.
        '''

        self.log.info('Starting Nelder–Mead Learner')

        N = int(self.num_params)

        one2np1 = list(range(1, N + 1))

        self.simplex_params[0] = self.init_simplex_corner

        try:
            self.simplex_costs[0] = self.put_params_and_get_cost(self.init_simplex_corner)
        except ValueError:
            self.log.error('Outside of boundary on initial condition. THIS SHOULD NOT HAPPEN')
            raise
        except LearnerInterrupt:
            self.log.info('Ended Nelder–Mead before end of simplex')
            return

        for k in range(0, N):
            y = np.array(self.init_simplex_corner, copy=True)
            y[k] = y[k] + self.init_simplex_disp[k]
            self.simplex_params[k + 1] = y
            try:
                f = self.put_params_and_get_cost(y)
            except ValueError:
                self.log.error('Outside of boundary on initial condition. THIS SHOULD NOT HAPPEN')
                raise
            except LearnerInterrupt:
                self.log.info('Ended Nelder–Mead before end of simplex')
                return

            self.simplex_costs[k + 1] = f

        ind = np.argsort(self.simplex_costs)
        self.simplex_costs = np.take(self.simplex_costs, ind, 0)
        # sort so sim[0,:] has the lowest function value
        self.simplex_params = np.take(self.simplex_params, ind, 0)

        while not self.end_event.is_set():

            xbar = np.add.reduce(self.simplex_params[:-1], 0) / N
            xr = (1 +self.rho) * xbar -self.rho * self.simplex_params[-1]

            if self.check_in_boundary(xr):
                try:
                    fxr = self.put_params_and_get_cost(xr)
                except ValueError:
                    self.log.error('Outside of boundary on first reduce. THIS SHOULD NOT HAPPEN')
                    raise
                except LearnerInterrupt:
                    break
            else:
                #Hit boundary so set the cost to positive infinite to ensure reflection
                fxr = float('inf')
                self.num_boundary_hits+=1
                self.log.debug('Hit boundary (reflect): '+str(self.num_boundary_hits)+' times.')

            doshrink = 0

            if fxr < self.simplex_costs[0]:
                xe = (1 +self.rho *self.chi) * xbar -self.rho *self.chi * self.simplex_params[-1]

                if self.check_in_boundary(xe):
                    try:
                        fxe = self.put_params_and_get_cost(xe)
                    except ValueError:
                        self.log.error('Outside of boundary when it should not be. THIS SHOULD NOT HAPPEN')
                        raise
                    except LearnerInterrupt:
                        break
                else:
                    #Hit boundary so set the cost above maximum this ensures the algorithm does a contracting reflection
                    fxe = fxr+1.0
                    self.num_boundary_hits+=1
                    self.log.debug('Hit boundary (expand): '+str(self.num_boundary_hits)+' times.')

                if fxe < fxr:
                    self.simplex_params[-1] = xe
                    self.simplex_costs[-1] = fxe
                else:
                    self.simplex_params[-1] = xr
                    self.simplex_costs[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < self.simplex_costs[-2]:
                    self.simplex_params[-1] = xr
                    self.simplex_costs[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < self.simplex_costs[-1]:
                        xc = (1 +self.psi *self.rho) * xbar -self.psi *self.rho * self.simplex_params[-1]
                        try:
                            fxc = self.put_params_and_get_cost(xc)
                        except ValueError:
                            self.log.error('Outside of boundary on contraction: THIS SHOULD NOT HAPPEN')
                            raise
                        except LearnerInterrupt:
                            break
                        if fxc <= fxr:
                            self.simplex_params[-1] = xc
                            self.simplex_costs[-1] = fxc
                        else:
                            doshrink = 1
                    else:
                        # Perform an inside contraction
                        xcc = (1 -self.psi) * xbar +self.psi * self.simplex_params[-1]
                        try:
                            fxcc = self.put_params_and_get_cost(xcc)
                        except ValueError:
                            self.log.error('Outside of boundary on inside contraction: THIS SHOULD NOT HAPPEN')
                            raise
                        except LearnerInterrupt:
                            break
                        if fxcc < self.simplex_costs[-1]:
                            self.simplex_params[-1] = xcc
                            self.simplex_costs[-1] = fxcc
                        else:
                            doshrink = 1
                    if doshrink:
                        for j in one2np1:
                            self.simplex_params[j] = self.simplex_params[0] +self.sigma * (self.simplex_params[j] - self.simplex_params[0])
                            try:
                                self.simplex_costs[j] = self.put_params_and_get_cost(self.simplex_params[j])
                            except ValueError:
                                self.log.error('Outside of boundary on shrink contraction: THIS SHOULD NOT HAPPEN')
                                raise
                            except LearnerInterrupt:
                                break

            ind = np.argsort(self.simplex_costs)
            self.simplex_params = np.take(self.simplex_params, ind, 0)
            self.simplex_costs = np.take(self.simplex_costs, ind, 0)

        self._shut_down()
        self.log.info('Ended Nelder–Mead')

    def update_archive(self):
        '''
        Update the archive.
        '''
        super(NelderMeadLearner, self).update_archive()
        new_values_dict = {
            'simplex_parameters':self.simplex_params,
            'simplex_costs':self.simplex_costs,
        }
        self.archive_dict.update(new_values_dict)

class DifferentialEvolutionLearner(Learner, threading.Thread):
    '''
    Adaption of the differential evolution algorithm in scipy.

    Args:
        params_out_queue (queue): Queue for parameters sent to controller.
        costs_in_queue (queue): Queue for costs for gaussian process. This must be tuple
        end_event (event): Event to trigger end of learner.

    Keyword Args:
        first_params (Optional [array]): The first parameters to test. If None will just randomly sample the initial condition. Default None.
        trust_region (Optional [float or array]): The trust region defines the maximum distance the learner will travel from the current best set of parameters. If None, the learner will search everywhere. If a float, this number must be between 0 and 1 and defines maximum distance the learner will venture as a percentage of the boundaries. If it is an array, it must have the same size as the number of parameters and the numbers define the maximum absolute distance that can be moved along each direction.
        evolution_strategy (Optional [string]): the differential evolution strategy to use, options are 'best1', 'best2', 'rand1' and 'rand2'. The default is 'best1'.
        population_size (Optional [int]): multiplier proportional to the number of parameters in a generation. The generation population is set to population_size * parameter_num. Default 15.
        mutation_scale (Optional [tuple]): The mutation scale when picking new points. Otherwise known as differential weight. When provided as a tuple (min,max) a mutation constant is picked randomly in the interval. Default (0.5,1.0).
        cross_over_probability (Optional [float]): The recombination constand or crossover probability, the probability a new points will be added to the population.
        restart_tolerance (Optional [float]): when the current population have a spread less than the initial tolerance, namely stdev(curr_pop) < restart_tolerance stdev(init_pop), it is likely the population is now in a minima, and so the search is started again.

    Attributes:
        has_trust_region (bool): Whether the learner has a trust region.
        num_population_members (int): The number of parameters in a generation.
        params_generations (list): History of the parameters generations. A list of all the parameters in the population, for each generation created.
        costs_generations (list): History of the costs generations. A list of all the costs in the population, for each generation created.
        init_std (float): The initial standard deviation in costs of the population. Calculated after sampling (or resampling) the initial population.
        curr_std (float): The current standard deviation in costs of the population. Calculated after sampling each generation.
    '''

    def __init__(self,
                 first_params = None,
                 trust_region = None,
                 evolution_strategy='best1',
                 population_size=15,
                 mutation_scale=(0.5, 1),
                 cross_over_probability=0.7,
                 restart_tolerance=0.01,
                 **kwargs):

        super(DifferentialEvolutionLearner,self).__init__(**kwargs)

        if first_params is None:
            self.first_params = float('nan')
        else:
            self.first_params = np.array(first_params, dtype=float)
            if not self.check_num_params(self.first_params):
                self.log.error('first_params has the wrong number of parameters:' + repr(self.first_params))
                raise ValueError
            if not self.check_in_boundary(self.first_params):
                self.log.error('first_params is not in the boundary:' + repr(self.first_params))
                raise ValueError

        self._set_trust_region(trust_region)

        if evolution_strategy == 'best1':
            self.mutation_func = self._best1
        elif evolution_strategy == 'best2':
            self.mutation_func = self._best2
        elif evolution_strategy == 'rand1':
            self.mutation_func = self._rand1
        elif evolution_strategy == 'rand2':
            self.mutation_func = self._rand2
        else:
            self.log.error('Please select a valid mutation strategy')
            raise ValueError

        self.evolution_strategy = evolution_strategy
        self.restart_tolerance = restart_tolerance

        if len(mutation_scale) == 2 and (np.any(np.array(mutation_scale) <= 2) or np.any(np.array(mutation_scale) > 0)):
            self.mutation_scale = mutation_scale
        else:
            self.log.error('Mutation scale must be a tuple with (min,max) between 0 and 2. mutation_scale:' + repr(mutation_scale))
            raise ValueError

        if cross_over_probability <= 1 and cross_over_probability >= 0:
            self.cross_over_probability = cross_over_probability
        else:
            self.log.error('Cross over probability must be between 0 and 1. cross_over_probability:' + repr(cross_over_probability))

        if population_size >= 5:
            self.population_size = population_size
        else:
            self.log.error('Population size must be greater or equal to 5:' + repr(population_size))

        self.num_population_members = self.population_size * self.num_params

        self.first_sample = True

        self.params_generations = []
        self.costs_generations = []
        self.generation_count = 0

        self.min_index = 0
        self.init_std = 0
        self.curr_std = 0

        self.archive_dict.update({'archive_type':'differential_evolution',
                                  'evolution_strategy':self.evolution_strategy,
                                  'mutation_scale':self.mutation_scale,
                                  'cross_over_probability':self.cross_over_probability,
                                  'population_size':self.population_size,
                                  'num_population_members':self.num_population_members,
                                  'restart_tolerance':self.restart_tolerance,
                                  'first_params':self.first_params,
                                  'has_trust_region':self.has_trust_region,
                                  'trust_region':self.trust_region})


    def run(self):
        '''
        Runs the Differential Evolution Learner.
        '''
        try:

            self.generate_population()

            while not self.end_event.is_set():

                self.next_generation()

                if self.curr_std < self.restart_tolerance * self.init_std:
                    self.generate_population()

        except LearnerInterrupt:
            return

    def save_generation(self):
        '''
        Save history of generations.
        '''
        self.params_generations.append(np.copy(self.population))
        self.costs_generations.append(np.copy(self.population_costs))
        self.generation_count += 1

    def generate_population(self):
        '''
        Sample a new random set of variables
        '''

        self.population = []
        self.population_costs = []
        self.min_index = 0

        if np.all(np.isfinite(self.first_params)) and self.first_sample:
            curr_params = self.first_params
            self.first_sample = False
        else:
            curr_params = self.min_boundary + nr.rand(self.num_params) * self.diff_boundary

        curr_cost = self.put_params_and_get_cost(curr_params)

        self.population.append(curr_params)
        self.population_costs.append(curr_cost)

        for index in range(1, self.num_population_members):

            if self.has_trust_region:
                temp_min = np.maximum(self.min_boundary,self.population[self.min_index] - self.trust_region)
                temp_max = np.minimum(self.max_boundary,self.population[self.min_index] + self.trust_region)
                curr_params = temp_min + nr.rand(self.num_params) * (temp_max - temp_min)
            else:
                curr_params = self.min_boundary + nr.rand(self.num_params) * self.diff_boundary

            curr_cost = self.put_params_and_get_cost(curr_params)

            self.population.append(curr_params)
            self.population_costs.append(curr_cost)

            if curr_cost < self.population_costs[self.min_index]:
                self.min_index = index

        self.population = np.array(self.population)
        self.population_costs = np.array(self.population_costs)

        self.init_std = np.std(self.population_costs)
        self.curr_std = self.init_std

        self.save_generation()

    def next_generation(self):
        '''
        Evolve the population by a single generation
        '''

        self.curr_scale = nr.uniform(self.mutation_scale[0], self.mutation_scale[1])

        for index in range(self.num_population_members):

            curr_params = self.mutate(index)

            curr_cost = self.put_params_and_get_cost(curr_params)

            if curr_cost < self.population_costs[index]:
                self.population[index] = curr_params
                self.population_costs[index] = curr_cost

                if curr_cost < self.population_costs[self.min_index]:
                    self.min_index = index

        self.curr_std = np.std(self.population_costs)

        self.save_generation()

    def mutate(self, index):
        '''
        Mutate the parameters at index.

        Args:
            index (int): Index of the point to be mutated.
        '''

        fill_point = nr.randint(0, self.num_params)
        candidate_params = self.mutation_func(index)
        crossovers = nr.rand(self.num_params) < self.cross_over_probability
        crossovers[fill_point] = True
        mutated_params = np.where(crossovers, candidate_params, self.population[index])

        if self.has_trust_region:
            temp_min = np.maximum(self.min_boundary,self.population[self.min_index] - self.trust_region)
            temp_max = np.minimum(self.max_boundary,self.population[self.min_index] + self.trust_region)
            rand_params = temp_min + nr.rand(self.num_params) * (temp_max - temp_min)
        else:
            rand_params = self.min_boundary + nr.rand(self.num_params) * self.diff_boundary

        projected_params = np.where(np.logical_or(mutated_params < self.min_boundary, mutated_params > self.max_boundary), rand_params, mutated_params)

        return projected_params

    def _best1(self, index):
        '''
        Use best parameters and two others to generate mutation.

        Args:
            index (int): Index of member to mutate.
        '''
        r0, r1 = self.random_index_sample(index, 2)
        return (self.population[self.min_index] + self.curr_scale *(self.population[r0] - self.population[r1]))

    def _rand1(self, index):
        '''
        Use three random parameters to generate mutation.

        Args:
            index (int): Index of member to mutate.
        '''
        r0, r1, r2 = self.random_index_sample(index, 3)
        return (self.population[r0] + self.curr_scale * (self.population[r1] - self.population[r2]))

    def _best2(self, index):
        '''
        Use best parameters and four others to generate mutation.

        Args:
            index (int): Index of member to mutate.
        '''
        r0, r1, r2, r3 = self.random_index_sample(index, 4)
        return self.population[self.min_index] + self.curr_scale * (self.population[r0] + self.population[r1] - self.population[r2] - self.population[r3])

    def _rand2(self, index):
        '''
        Use five random parameters to generate mutation.

        Args:
            index (int): Index of member to mutate.
        '''
        r0, r1, r2, r3, r4 = self.random_index_sample(index, 5)
        return self.population[r0] + self.curr_scale * (self.population[r1] + self.population[r2] - self.population[r3] - self.population[r4])

    def random_index_sample(self, index, num_picks):
        '''
        Randomly select a num_picks of indexes, without index.

        Args:
            index(int): The index that is not included
            num_picks(int): The number of picks.
        '''
        rand_indexes = list(range(self.num_population_members))
        rand_indexes.remove(index)
        return random.sample(rand_indexes, num_picks)

    def update_archive(self):
        '''
        Update the archive.
        '''
        super(DifferentialEvolutionLearner, self).update_archive()
        new_values_dict = {
            'params_generations':self.params_generations,
            'costs_generations':self.costs_generations,
            'population':self.population,
            'population_costs':self.population_costs,
            'init_std':self.init_std,
            'curr_std':self.curr_std,
            'generation_count':self.generation_count,
        }
        self.archive_dict.update(new_values_dict)


class MachineLearner(Learner):
    '''
    A parent class for more specific machine learer classes.

    This class is not intended to be used directly.

    Keyword Args:
        trust_region (Optional [float or array]): The trust region defines the
            maximum distance the learner will travel from the current best set
            of parameters. If None, the learner will search everywhere. If a
            float, this number must be between 0 and 1 and defines maximum
            distance the learner will venture as a percentage of the boundaries.
            If it is an array, it must have the same size as the number of
            parameters and the numbers define the maximum absolute distance that
            can be moved along each direction.
        default_bad_cost (Optional [float]): If a run is reported as bad and
            `default_bad_cost` is provided, the cost for the bad run is set to
            this default value. If `default_bad_cost` is `None`, then the worst
            cost received is set to all the bad runs. Default `None`.
        default_bad_uncertainty (Optional [float]): If a run is reported as bad
            and `default_bad_uncertainty` is provided, the uncertainty for the
            bad run is set to this default value. If `default_bad_uncertainty`
            is `None`, then the uncertainty is set to a tenth of the best to
            `worst cost range. Default `None`.
        minimum_uncertainty (Optional [float]): The minimum uncertainty
            associated with provided costs. Must be above zero to avoid fitting
            errors. Default `1e-8`.
        predict_global_minima_at_end (Optional [bool]): If `True` finds the
            global minima when the learner is ended. Does not if `False`.
            Default `True`.
        training_filename (Optional [str]): The name of a learner archive from a
            previous optimization from which to extract past results for use in
            the current optimization. If `None`, no past results will be used.
            Default `None`.
        training_file_type (Optional [str]): File type of the training archive.
            Can be `'txt'`, `'pkl'`, `'mat'`, or `None`. If set to `None`, then
            the file type will be determined automatically. This argument has no
            effect if `training_filename` is set to `None`. Default `None`.

    Attributes:
        all_params (array): Array containing all parameters sent to learner.
        all_costs (array): Array containing all costs sent to learner.
        all_uncers (array): Array containing all uncertainties sent to learner.
        scaled_costs (array): Array contaning all the costs scaled to have zero mean and a standard deviation of 1. Needed for training the learner.
        bad_run_indexs (list): list of indexes to all runs that were marked as bad.
        best_cost (float): Minimum received cost, updated during execution.
        best_params (array): Parameters of best run. (reference to element in params array).
        best_index (int): index of the best cost and params.
        worst_cost (float): Maximum received cost, updated during execution.
        worst_index (int): index to run with worst cost.
        cost_range (float): Difference between worst_cost and best_cost
        params_count (int): Counter for the number of parameters asked to be evaluated by the learner.
        has_trust_region (bool): Whether the learner has a trust region.
    '''

    def __init__(self,
                 trust_region=None,
                 default_bad_cost = None,
                 default_bad_uncertainty = None,
                 minimum_uncertainty = 1e-8,
                 predict_global_minima_at_end = True,
                 training_filename=None,
                 training_file_type=None,
                 **kwargs):
        # Prepare logger now so that logging can be done before calling parent's
        # __init__() method.
        self._prepare_logger()

        if training_filename is not None:
            # Automatically determine training_file_type if necessary.
            training_filename = str(training_filename)
            if training_file_type is None:
                training_file_type = mlu.get_file_type(training_filename)
            training_file_type = str(training_file_type)
            if not mlu.check_file_type_supported(training_file_type):
                msg = 'Training file type not supported: ' + repr(training_file_type)
                self.log.error(msg)
                raise ValueError(msg)
            self.training_file_dir = os.path.dirname(training_filename)

            # Get the training dictionary.
            training_dict = mlu.get_dict_from_file(
                training_filename,
                training_file_type,
            )
            self.training_dict = training_dict

            # Parameters that must match the values in the training archive.
            num_params = int(training_dict['num_params'])
            kwargs['num_params'] = self._reconcile_kwarg_and_training_val(
                kwargs,
                'num_params',
                num_params,
            )

            # Run parent's __init__() now so that it gets the updated value for
            # num_params but its empty values for all_params, etc., get
            # overwritten below.
            super(MachineLearner, self).__init__(**kwargs)

            # Data that must be present in any archive type.
            self.all_params = np.array(training_dict['all_params'], dtype=float)
            self.all_costs = mlu.safe_cast_to_array(training_dict['all_costs'])
            self.all_uncers = mlu.safe_cast_to_array(training_dict['all_uncers'])
            self.bad_run_indexs = mlu.safe_cast_to_list(training_dict['bad_run_indexs'])

            # Data that may be in the archive, but can easily be calculated if
            # necessary.
            # costs_count
            costs_count = training_dict.get(
                'costs_count',
                len(self.all_costs),
            )
            self.costs_count = int(costs_count)
            # best_index
            best_index = training_dict.get(
                'best_index',
                np.argmin(self.all_costs),
            )
            self.best_index = int(best_index)
            # best_cost
            best_cost = training_dict.get(
                'best_cost',
                self.all_costs[self.best_index],
            )
            self.best_cost = float(best_cost)
            # best_params
            best_params = training_dict.get(
                'best_params',
                self.all_params[self.best_index],
            )
            self.best_params = mlu.safe_cast_to_array(best_params)
            # worst_index
            worst_index = training_dict.get(
                'worst_index',
                np.argmax(self.all_costs),
            )
            self.worst_index = int(worst_index)
            # worst_cost
            worst_cost = training_dict.get(
                'worst_cost',
                self.all_costs[self.worst_index],
            )
            self.worst_cost = float(worst_cost)
            # cost_range
            cost_range = training_dict.get(
                'cost_range',
                (self.worst_cost - self.best_cost),
            )
            self.cost_range = float(cost_range)

            # Parameters that must be the same in keyword arguments and in the
            # training archive in order to load some of the data.
            # learner type
            self._learner_type_matches_training_archive = True
            learner_type_train = self.training_dict['archive_type']
            if learner_type_train != self._ARCHIVE_TYPE:
                self._learner_type_matches_training_archive = False
            # min_boundary
            self._boundaries_match_training_archive = True
            min_boundary_train = self.training_dict['min_boundary']
            min_boundary_train = mlu.safe_cast_to_array(min_boundary_train)
            are_same = np.array_equal(
                kwargs.get('min_boundary'),
                min_boundary_train,
            )
            if not are_same:
                self._boundaries_match_training_archive = False
            # max_boundary
            max_boundary_train = self.training_dict['max_boundary']
            max_boundary_train = mlu.safe_cast_to_array(max_boundary_train)
            are_same = np.array_equal(
                kwargs.get('max_boundary'),
                max_boundary_train,
            )
            if not are_same:
                self._boundaries_match_training_archive = False

        else:
            super(MachineLearner, self).__init__(**kwargs)
            self._learner_type_matches_training_archive = False
            self._boundaries_match_training_archive = False
            #Storage variables, archived
            self.best_cost = float('inf')
            self.best_params = float('nan')
            self.best_index = 0
            self.worst_cost = float('-inf')
            self.worst_index = 0
            self.cost_range = float('inf')
            self.costs_count = 0

        # Parameters that should only be loaded if a training archive was
        # provided and it has the same learner type and min/max boundaries.
        same_learner_type = self._learner_type_matches_training_archive
        same_boundaries = self._boundaries_match_training_archive
        if same_learner_type and same_boundaries:
            training_dict = self.training_dict
            # Counters
            self.params_count = int(training_dict['params_count'])

            # Predicted optimum
            try:
                self.predicted_best_parameters = mlu.safe_cast_to_array(
                    training_dict['predicted_best_parameters']
                )
                self.predicted_best_cost = float(
                    training_dict['predicted_best_cost']
                )
                self.predicted_best_uncertainty = float(
                    training_dict['predicted_best_uncertainty']
                )
                self.has_global_minima = True
            except KeyError:
                self.has_global_minima = False
        else:
            # Counters
            self.params_count = 0

            # Predicted optimum
            self.has_global_minima = False

        # Multiprocessor controls
        self.new_params_event = mp.Event()

        # Storage variables and counters
        self.search_params = []
        self.scaled_costs = None

        # Constants, limits and tolerances
        self.search_precision = 1.0e-6
        self.parameter_searches = max(10, self.num_params)
        self.bad_uncer_frac = 0.1 # Fraction of cost range to set a bad run uncertainty

        # Optional user set variables
        self._set_trust_region(trust_region)
        self.predict_global_minima_at_end = bool(predict_global_minima_at_end)
        self.minimum_uncertainty = float(minimum_uncertainty)
        if default_bad_cost is not None:
            self.default_bad_cost = float(default_bad_cost)
        else:
            self.default_bad_cost = None
        if default_bad_uncertainty is not None:
            self.default_bad_uncertainty = float(default_bad_uncertainty)
        else:
            self.default_bad_uncertainty = None
        if (self.default_bad_cost is None) and (self.default_bad_uncertainty is None):
            self.bad_defaults_set = False
        elif (self.default_bad_cost is not None) and (self.default_bad_uncertainty is not None):
            self.bad_defaults_set = True
        else:
            self.log.error('Both the default cost and uncertainty must be set for a bad run or they must both be set to None.')
            raise ValueError
        if self.minimum_uncertainty <= 0:
            self.log.error('Minimum uncertainty must be larger than zero for the learner.')
            raise ValueError

        #Search bounds
        self.search_min = self.min_boundary
        self.search_max = self.max_boundary
        self.search_diff = self.search_max - self.search_min
        self.search_region = list(zip(self.search_min, self.search_max))

    def _reconcile_kwarg_and_training_val(self, kwargs_, name, training_value):
        '''Utility function for comparing values from kwargs to training values.

        When a training archive is specified there can be two values specified
        for some parameters; one from user's config/kwargs and one from the
        training archive. This function compares the values. If the values are
        the same then the value is returned, and if they are different a
        `ValueError` is raised. Care is taken not to raise that error though if
        one of the values is `None` since that can mean that a value wasn't
        specified. In that case the other value is returned, or `None` is
        returned if they are both `None`.

        Args:
            kwargs_ ([dict]): The dictionary of keyword arguments passed to
                `__init__()`.
            name ([str]): The name of the parameter.
            training_value ([any]): The value for the parameter in the training
                archive.

        Raises:
            ValueError: A `ValueError` is raised if the value of the parameter
                in the keyword arguments doesn't match the value from the
                training archive.

        Returns:
            [any]: The value for the parameter, taken from either `kwargs_` or
                `training_value`, or both if they are the same.
        '''
        if kwargs_.get(name) is None:
            # No non-default value provided in kwargs_, so use the training
            # value.
            return training_value
        elif training_value is None:
            # Have a non-default value in kwargs_ but training_value is None, so
            # use the value from kwargs_.
            return kwargs_[name]
        else:
            # In this case both kwargs_ and and training_value are non-default.
            # If they are the same, then return their common value. If they are
            # different raise an error to alert the user.
            if isinstance(kwargs_[name], np.ndarray) or isinstance(training_value, np.ndarray):
                same = np.array_equal(kwargs_[name], training_value)
            else:
                same = (kwargs_[name] == training_value)
            if same:
                return training_value
            else:
                msg = ("Value passed for {name} ({kwargs_val}) does not match "
                       "value in training archive ({training_value}).").format(
                           name=name,
                           kwargs_val=kwargs_[name],
                           training_value=training_value,
                       )
                self.log.error(msg)
                raise ValueError(msg)

    def update_archive(self):
        '''
        Update the archive.
        '''
        super(MachineLearner, self).update_archive()
        new_values_dict = {
            'best_cost':self.best_cost,
            'best_params':self.best_params,
            'best_index':self.best_index,
            'worst_cost':self.worst_cost,
            'worst_index':self.worst_index,
            'cost_range':self.cost_range,
            'costs_count':self.costs_count,
            'params_count':self.params_count,
        }
        self.archive_dict.update(new_values_dict)


class GaussianProcessLearner(MachineLearner, mp.Process):
    '''
    Gaussian process learner.

    Generates new parameters based on a gaussian process fitted to all previous
    data.

    Args:
        params_out_queue (queue): Queue for parameters sent to controller.
        costs_in_queue (queue): Queue for costs for gaussian process. This must
            be tuple.
        end_event (event): Event to trigger end of learner.

    Keyword Args:
        length_scale (Optional [array]): The initial guess for length scale(s)
            of the gaussian process. The array can either of size one or the
            number of parameters or `None`. If it is size one, it is assumed
            that all of the correlation lengths are the same. If it is an array
            with length equal to the number of the parameters then all the
            parameters have their own independent length scale. If it is set to
            `None` and a learner archive from a Gaussian process optimization
            is provided for `gp_training_filename`, then it will be set to the
            value recorded for `length_scale` in that learner archive. If set to
            `None` but `gp_training_filename` does not specify a learner archive
            from a Guassian process optimization, then it is assumed that all of
            the length scales should be independent and they are all given an
            initial value of `1`. Default `None`.
        length_scale_bounds (Optional [array]): The limits on the fitted length
            scale values, specified as a single pair of numbers e.g.
            `[min, max]`, or a list of pairs of numbers, e.g.
            `[[min_0, max_0], ..., [min_N, max_N]]`. This only has an effect if
            `update_hyperparameters` is set to `True`. If one pair is provided,
            the same limits will be used for all length scales. Alternatively
            one pair of `[min, max]` can be provided for each length scale. For
            example, possible valid values include `[1e-5, 1e5]` and
            `[[1e-2, 1e2], [5, 5], [1.6e-4, 1e3]]` for optimizations with three
            parameters. If set to `None`, the value `[1e-5, 1e5]` will be used.
            Default `None`.
        update_hyperparameters (Optional [bool]): Whether the length scales and
            noise estimate should be updated when new data is provided. Default
            `True`.
        cost_has_noise (Optional [bool]): If `True` the learner assumes there is
            common additive white noise that corrupts the costs provided. This
            noise is assumed to be on top of the uncertainty in the costs (if it
            is provided). If `False`, it is assumed that there is no noise in
            the cost (or if uncertainties are provided no extra noise beyond the
            uncertainty). Default `True`.
        noise_level (Optional [float]): The initial guess for the noise level
            (variance, not standard deviation) in the costs. This is only used
            if `cost_has_noise` is `True`. If it is set to `None` and a learner
            archive from a Gaussian process optimization is provided for
            `gp_training_filename`, then it will be set to the value recorded
            for `noise_level` in that learner archive. If set to `None` but
            `gp_training_filename` does not specify a learner archive from a
            Guassian process optimization, then it will automatically be set to
            the variance of the training data costs.
        noise_level_bounds (Optional [array]): The limits on the fitted
            `noise_level` values, specified as a single pair of numbers
            `[min, max]`. This only has an effect if `update_hyperparameters`
            and `cost_has_noise` are both set to `True`. If set to `None`, the
            value `[1e-5 * var, 1e5 * var]` will be used where `var` is the
            variance of the training data costs. Default `None`.
        gp_training_filename (Optional [str]): The name of a learner archive
            from a previous optimization from which to extract past results for
            use in the current optimization. If `None`, no past results will be
            used. Default `None`.
        gp_training_file_type (Optional [str]): File type of the training
            archive. Can be `'txt'`, `'pkl'`, `'mat'`, or `None`. If set to
            `None`, then the file type will be determined automatically. This
            argument has no effect if `gp_training_filename` is set to `None`.
            Default `None`.
        trust_region (Optional [float or array]): The trust region defines the
            maximum distance the learner will travel from the current best set
            of parameters. If `None`, the learner will search everywhere. If a
            float, this number must be between 0 and 1 and defines maximum
            distance the learner will venture as a percentage of the boundaries.
            If it is an array, it must have the same size as the number of
            parameters and the numbers define the maximum absolute distance that
            can be moved along each direction.
        default_bad_cost (Optional [float]): If a run is reported as bad and
            `default_bad_cost` is provided, the cost for the bad run is set to
            this default value. If `default_bad_cost` is `None`, then the worst
            cost received is set to all the bad runs. Default `None`.
        default_bad_uncertainty (Optional [float]): If a run is reported as bad
            and `default_bad_uncertainty` is provided, the uncertainty for the
            bad run is set to this default value. If `default_bad_uncertainty`
            is `None`, then the uncertainty is set to a tenth of the best to
            worst cost range. Default `None`.
        minimum_uncertainty (Optional [float]): The minimum uncertainty
            associated with provided costs. Must be above zero to avoid fitting
            errors. Default `1e-8`.
        predict_global_minima_at_end (Optional [bool]): If `True` attempts to
            find the global minima when the learner is ended. Does not if
            `False`. Default `True`.

    Attributes:
        all_params (array): Array containing all parameters sent to learner.
        all_costs (array): Array containing all costs sent to learner.
        all_uncers (array): Array containing all uncertainties sent to learner.
        scaled_costs (array): Array contaning all the costs scaled to have zero
            mean and a standard deviation of 1. Needed for training the gaussian
            process.
        bad_run_indexs (list): list of indexes to all runs that were marked as
            bad.
        best_cost (float): Minimum received cost, updated during execution.
        best_params (array): Parameters of best run. (reference to element in
            params array).
        best_index (int): index of the best cost and params.
        worst_cost (float): Maximum received cost, updated during execution.
        worst_index (int): index to run with worst cost.
        cost_range (float): Difference between `worst_cost` and `best_cost`.
        generation_num (int): Number of sets of parameters to generate each
            generation. Set to `4`.
        length_scale_history (list): List of length scales found after each fit.
        noise_level_history (list): List of noise levels found after each fit.
        fit_count (int): Counter for the number of times the gaussian process
            has been fit.
        cost_count (int): Counter for the number of costs, parameters and
            uncertainties added to learner.
        params_count (int): Counter for the number of parameters asked to be
            evaluated by the learner.
        gaussian_process (GaussianProcessRegressor): Gaussian process that is
            fitted to data and used to make predictions
        cost_scaler (StandardScaler): Scaler used to normalize the provided
            costs.
        has_trust_region (bool): Whether the learner has a trust region.
    '''
    _ARCHIVE_TYPE = 'gaussian_process_learner'

    def __init__(self,
                 length_scale = None,
                 length_scale_bounds=None,
                 update_hyperparameters = True,
                 cost_has_noise=True,
                 noise_level=None,
                 noise_level_bounds=None,
                 gp_training_filename =None,
                 gp_training_file_type = None,
                 **kwargs):

        if gp_training_filename is not None:
            super(GaussianProcessLearner,self).__init__(
                training_filename=gp_training_filename,
                training_file_type=gp_training_file_type,
                **kwargs
            )

            # Maintain backwards compatability with archives generated by
            # previous versions of M-LOOP.
            if 'mloop_version' not in self.training_dict:
                # M-LOOP versions <= 3.1.1 didn't scale noise level and didn't
                # record the M-LOOP version. Mark that noise levels should be
                # unscaled later, which is necessary for plotting for archives
                # from older versions of M-LOOP.
                self._scale_deprecated_noise_levels = True
            else:
                self._scale_deprecated_noise_levels = False

        else:
            super(GaussianProcessLearner,self).__init__(**kwargs)

            # Maintain backwards compatability with archives generated by
            # previous versions of M-LOOP.
            self._scale_deprecated_noise_levels = False

        # Parameters that should only be loaded if a training archive was
        # provided and it has the same learner type.
        if self._learner_type_matches_training_archive:
            training_dict = self.training_dict
            # Storage variables, archived
            self.length_scale_history = list(
                training_dict['length_scale_history']
            )
            self.noise_level_history = mlu.safe_cast_to_list(
                training_dict['noise_level_history']
            )

            # Counters
            self.fit_count = int(training_dict['fit_count'])

            # Fit parameters that can be overriden by user keyword options.
            if length_scale is None:
                length_scale = mlu.safe_cast_to_array(training_dict['length_scale'])
            if noise_level is None:
                noise_level = float(training_dict['noise_level'])
        else:
            # Storage variables, archived
            self.length_scale_history = []
            self.noise_level_history = []

            # Counters
            self.fit_count = 0

        #Storage variables and counters
        self.scaled_uncers = None
        self.scaled_noise_level = None
        self.scaled_noise_level_bounds = None
        self.cost_bias = None
        self.uncer_bias = None

        #Internal variable for bias function
        self.bias_func_cycle = 4
        self.bias_func_cost_factor = [1.0,1.0,1.0,1.0]
        self.bias_func_uncer_factor =[0.0,1.0,2.0,3.0]
        self.generation_num = self.bias_func_cycle
        if self.generation_num < 3:
            self.log.error('Number in generation must be larger than 2.')
            raise ValueError

        #Constants, limits and tolerances
        self.hyperparameter_searches = max(10,self.num_params)

        #Optional user set variables
        self.cost_has_noise = bool(cost_has_noise)
        if length_scale is None:
            self.length_scale = np.ones((self.num_params,))
        else:
            self.length_scale = np.array(length_scale, dtype=float)
        if noise_level is None:
            # Temporarily change to NaN to mark that the default value
            # should be calcualted once training data is available. Using
            # NaN instead of None is necessary in case the archive is saved
            # in .mat format since it can handle NaN but not None.
            self.noise_level = float('nan')
        else:
            self.noise_level = float(noise_level)
        self.update_hyperparameters = bool(update_hyperparameters)
        if length_scale_bounds is None:
            self.length_scale_bounds = np.array([1e-5, 1e5])
        else:
            self.length_scale_bounds = mlu.safe_cast_to_array(length_scale_bounds)
        if noise_level_bounds is None:
            self.noise_level_bounds = float('nan')
        else:
            self.noise_level_bounds = mlu.safe_cast_to_array(noise_level_bounds)

        #Checks of variables
        if self.length_scale.size == 1:
            self.length_scale = float(self.length_scale)
        elif not self.check_num_params(self.length_scale):
            self.log.error('Correlation lengths not the right size and shape, must be one or the number of parameters:' + repr(self.length_scale))
            raise ValueError
        if not np.all(self.length_scale >0):
            self.log.error('Correlation lengths must all be positive numbers:' + repr(self.length_scale))
            raise ValueError
        self._check_length_scale_bounds()
        if self.noise_level < 0:
            self.log.error('noise_level must be greater or equal to zero:' +repr(self.noise_level))
            raise ValueError
        self._check_noise_level_bounds()
        if self.default_bad_uncertainty is not None:
            if self.default_bad_uncertainty < 0:
                self.log.error('Default bad uncertainty must be positive.')
                raise ValueError

        self.gaussian_process = None

        self.cost_scaler = skp.StandardScaler()

        self.archive_dict.update({'archive_type':self._ARCHIVE_TYPE,
                                  'cost_has_noise':self.cost_has_noise,
                                  'length_scale_history':self.length_scale_history,
                                  'length_scale_bounds':self.length_scale_bounds,
                                  'noise_level_history':self.noise_level_history,
                                  'noise_level_bounds':self.noise_level_bounds,
                                  'bias_func_cycle':self.bias_func_cycle,
                                  'bias_func_cost_factor':self.bias_func_cost_factor,
                                  'bias_func_uncer_factor':self.bias_func_uncer_factor,
                                  'generation_num':self.generation_num,
                                  'search_precision':self.search_precision,
                                  'parameter_searches':self.parameter_searches,
                                  'hyperparameter_searches':self.hyperparameter_searches,
                                  'bad_uncer_frac':self.bad_uncer_frac,
                                  'trust_region':self.trust_region,
                                  'has_trust_region':self.has_trust_region,
                                  'predict_global_minima_at_end':self.predict_global_minima_at_end})
        #Remove logger so gaussian process can be safely picked for multiprocessing on Windows
        self.log = None

    def _check_length_scale_bounds(self):
        '''
        Ensure self.length_scale_bounds has a valid value, otherwise raise a
        ValueError.
        '''
        bounds = self.length_scale_bounds
        # First ensure that all of the limits are positive numbers.
        if not np.all(bounds > 0):
            message = 'Correlation length bounds must all be positive numbers: ' + repr(self.length_scale_bounds)
            self.log.error(message)
            raise ValueError(message)
        dims_error_message = ('Length scale bounds must a single pair '
                              '(min, max) or a list of pairs [(min_0, max_0), '
                              '..., (min_N, max_N)] with one pair per '
                              'parameter: ' + repr(bounds))
        range_error_message = ('The length scale lower bound must be less than '
                               'or equal to the upper bound: ' + repr(bounds))
        if bounds.ndim == 1:
            # In this case, length_scale_bounds should be a single pair of
            # numbers, e.g. (1, 2).
            if bounds.shape[0] != 2:
                self.log.error(dims_error_message)
                raise ValueError(dims_error_message)
            # Ensure min <= max.
            if bounds[1] < bounds[0]:
                self.log.error(range_error_message)
                raise ValueError(range_error_message)
        elif bounds.ndim == 2:
            # In this case, length_scale_bounds should be a list of pairs of
            # numbers, with exactly one pair per parameter.
            if bounds.shape[0] != self.num_params:
                self.log.error(dims_error_message)
                raise ValueError(dims_error_message)
            elif bounds.shape[1] != 2:
                self.log.error(dims_error_message)
                raise ValueError(dims_error_message)
            # Ensure min <= max for all pairs.
            if np.any(bounds[:, 1] < bounds[:, 0]):
                self.log.error(range_error_message)
                raise ValueError(range_error_message)
        else:
            # Any number of dimensions other that 1 or 2 is definitely wrong.
            self.log.error(dims_error_message)
            raise ValueError(dims_error_message)

    def _check_noise_level_bounds(self):
        '''
        Ensure self.noise_level has a valid value, otherwise raise a ValueError.
        '''
        bounds = self.noise_level_bounds
        # If self.noise_level_bounds is set to NaN, then it's actual value will
        # be automatically set later once training data is available. In that
        # case there's no need to check anything.
        if np.any(np.isnan(bounds)):
            return
        # Ensure that all of the limits are positive numbers.
        if not np.all(bounds > 0):
            message = ('Noise level bounds must all be positive numbers: ' +
                       repr(bounds))
            self.log.error(message)
            raise ValueError(message)
        # Ensure that the dimensions are correct.
        if bounds.shape != (2,):
            message = ('Noise level bounds should have exactly two elements: ' +
                       repr(bounds))
            self.log.error(message)
            raise ValueError(message)
        # Ensure min <= max.
        if bounds[1] < bounds[0]:
            message = ('Noise level lower bound must be less than or equal to '
                       'upper bound' + repr(bounds))
            self.log.error(message)
            raise ValueError(message)

    def create_gaussian_process(self):
        '''
        Create a Gaussian process.
        '''
        gp_kernel = skk.RBF(
            length_scale=self.length_scale,
            length_scale_bounds=self.length_scale_bounds,
        )
        if self.cost_has_noise:
            white_kernel = skk.WhiteKernel(
                noise_level=self.scaled_noise_level,
                noise_level_bounds=self.scaled_noise_level_bounds,
            )
            gp_kernel = gp_kernel + white_kernel
        alpha = self.scaled_uncers**2
        if self.update_hyperparameters:
            self.gaussian_process = skg.GaussianProcessRegressor(alpha=alpha, kernel=gp_kernel,n_restarts_optimizer=self.hyperparameter_searches)
        else:
            self.gaussian_process = skg.GaussianProcessRegressor(alpha=alpha, kernel=gp_kernel,optimizer=None)

    def wait_for_new_params_event(self):
        '''
        Waits for a new parameters event and starts a new parameter generation cycle. Also checks end event and will break if it is triggered.
        '''
        while not self.end_event.is_set():
            if self.new_params_event.wait(timeout=self.learner_wait):
                self.new_params_event.clear()
                break
            else:
                continue
        else:
            self.log.debug('GaussianProcessLearner end signal received. Ending')
            raise LearnerInterrupt


    def get_params_and_costs(self):
        '''
        Get the parameters and costs from the queue and place in their appropriate all_[type] arrays. Also updates bad costs, best parameters, and search boundaries given trust region.
        '''
        if self.costs_in_queue.empty():
            if self.end_event.is_set():
                return
            else:
                self.log.error('Gaussian process asked for new parameters but no new costs were provided.')
                raise ValueError

        new_params = []
        new_costs = []
        new_uncers = []
        new_bads = []
        update_bads_flag = False

        while not self.costs_in_queue.empty():
            (param, cost, uncer, bad) = self.costs_in_queue.get_nowait()
            self.costs_count +=1

            if bad:
                new_bads.append(self.costs_count-1)
                if self.bad_defaults_set:
                    cost = self.default_bad_cost
                    uncer = self.default_bad_uncertainty
                else:
                    cost = self.worst_cost
                    uncer = self.cost_range*self.bad_uncer_frac

            message = (param, cost, uncer, bad)
            param, cost, uncer, bad = self._parse_cost_message(message)

            uncer = max(uncer, self.minimum_uncertainty)

            cost_change_flag = False
            if cost > self.worst_cost:
                self.worst_cost = cost
                self.worst_index = self.costs_count-1
                cost_change_flag = True
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_params = param
                self.best_index =  self.costs_count-1
                cost_change_flag = True
            if cost_change_flag:
                self.cost_range = self.worst_cost - self.best_cost
                if not self.bad_defaults_set:
                    update_bads_flag = True

            new_params.append(param)
            new_costs.append(cost)
            new_uncers.append(uncer)


        if self.all_params.size==0:
            self.all_params = np.array(new_params, dtype=float)
            self.all_costs = np.array(new_costs, dtype=float)
            self.all_uncers = np.array(new_uncers, dtype=float)
        else:
            self.all_params = np.concatenate((self.all_params, np.array(new_params, dtype=float)))
            self.all_costs = np.concatenate((self.all_costs, np.array(new_costs, dtype=float)))
            self.all_uncers = np.concatenate((self.all_uncers, np.array(new_uncers, dtype=float)))

        self.bad_run_indexs.append(new_bads)

        if self.all_params.shape != (self.costs_count,self.num_params):
            self.log('Saved GP params are the wrong size. THIS SHOULD NOT HAPPEN:' + repr(self.all_params))
        if self.all_costs.shape != (self.costs_count,):
            self.log('Saved GP costs are the wrong size. THIS SHOULD NOT HAPPEN:' + repr(self.all_costs))
        if self.all_uncers.shape != (self.costs_count,):
            self.log('Saved GP uncertainties are the wrong size. THIS SHOULD NOT HAPPEN:' + repr(self.all_uncers))

        if update_bads_flag:
            self.update_bads()

        self.update_search_region()

    def update_bads(self):
        '''
        Best and/or worst costs have changed, update the values associated with bad runs accordingly.
        '''
        for index in self.bad_run_indexs:
            self.all_costs[index] = self.worst_cost
            self.all_uncers[index] = self.cost_range*self.bad_uncer_frac

    def update_search_region(self):
        '''
        If trust boundaries is not none, updates the search boundaries based on the defined trust region.
        '''
        if self.has_trust_region:
            self.search_min = np.maximum(self.best_params - self.trust_region, self.min_boundary)
            self.search_max = np.minimum(self.best_params + self.trust_region, self.max_boundary)
            self.search_diff = self.search_max - self.search_min
            self.search_region = list(zip(self.search_min, self.search_max))

    def update_search_params(self):
        '''
        Update the list of parameters to use for the next search.
        '''
        self.search_params = []
        self.search_params.append(self.best_params)
        for _ in range(self.parameter_searches):
            self.search_params.append(self.search_min + nr.uniform(size=self.num_params) * self.search_diff)

    def update_archive(self):
        '''
        Update the archive.
        '''
        super(GaussianProcessLearner, self).update_archive()
        new_values_dict = {
            'fit_count':self.fit_count,
            'update_hyperparameters':self.update_hyperparameters,
            'length_scale':self.length_scale,
            'noise_level':self.noise_level,
        }
        self.archive_dict.update(new_values_dict)

    def fit_gaussian_process(self):
        '''
        Fit the Gaussian process to the current data
        '''
        self.log.debug('Fitting Gaussian process.')
        if self.all_params.size==0 or self.all_costs.size==0 or self.all_uncers.size==0:
            self.log.error('Asked to fit GP but no data is in all_costs, all_params or all_uncers.')
            raise ValueError
        self.scaled_costs = self.cost_scaler.fit_transform(self.all_costs[:,np.newaxis])[:,0]
        cost_scaling_factor = float(self.cost_scaler.scale_)
        self.scaled_uncers = self.all_uncers / cost_scaling_factor
        if self.cost_has_noise:
            # Ensure compatability with archives from M-LOOP versions <= 3.1.1.
            if self._scale_deprecated_noise_levels:
                self.noise_level = self.noise_level * cost_scaling_factor**2
                self.noise_level_history = [level * cost_scaling_factor**2 for level in self.noise_level_history]
                # Mark that scaling is done to avoid doing it multiple times.
                self._scale_deprecated_noise_levels = False
            if np.isnan(self.noise_level):
                # Set noise_level to its default value, which is the variance of
                # the training data, which is equal to the square of the cost
                # scaling factor. This will only happen on first iteration since
                # self.noise_level is overwritten.
                self.noise_level = cost_scaling_factor**2
            if np.any(np.isnan(self.noise_level_bounds)):
                self.noise_level_bounds = np.array([1e-5, 1e5]) * cost_scaling_factor**2
            # Cost variance's scaling factor is square of costs's scaling factor.
            self.scaled_noise_level = self.noise_level / cost_scaling_factor**2
            self.scaled_noise_level_bounds = self.noise_level_bounds / cost_scaling_factor**2

        self.create_gaussian_process()
        self.gaussian_process.fit(self.all_params,self.scaled_costs)

        if self.update_hyperparameters:

            self.fit_count += 1

            last_hyperparameters = self.gaussian_process.kernel_.get_params()

            if self.cost_has_noise:
                self.length_scale = last_hyperparameters['k1__length_scale']
                if isinstance(self.length_scale, float):
                    self.length_scale = np.array([self.length_scale])
                self.length_scale_history.append(self.length_scale)
                self.scaled_noise_level = last_hyperparameters['k2__noise_level']
                self.noise_level = self.scaled_noise_level * cost_scaling_factor**2
                self.noise_level_history.append(self.noise_level)
            else:
                self.length_scale = last_hyperparameters['length_scale']
                self.length_scale_history.append(self.length_scale)


    def update_bias_function(self):
        '''
        Set the constants for the cost bias function.
        '''
        self.cost_bias = self.bias_func_cost_factor[self.params_count%self.bias_func_cycle]
        self.uncer_bias = self.bias_func_uncer_factor[self.params_count%self.bias_func_cycle]

    def predict_biased_cost(self,params):
        '''
        Predicts the biased cost at the given parameters. The bias function is:
            biased_cost = cost_bias*pred_cost - uncer_bias*pred_uncer

        Returns:
            pred_bias_cost (float): Biased cost predicted at the given parameters
        '''
        (pred_cost, pred_uncer) = self.gaussian_process.predict(params[np.newaxis,:], return_std=True)
        return self.cost_bias*pred_cost - self.uncer_bias*pred_uncer

    def find_next_parameters(self):
        '''
        Returns next parameters to find. Increments counters and bias function appropriately.

        Return:
            next_params (array): Returns next parameters from biased cost search.
        '''
        self.params_count += 1
        self.update_bias_function()
        self.update_search_params()
        next_params = None
        next_cost = float('inf')
        for start_params in self.search_params:
            result = so.minimize(self.predict_biased_cost, start_params, bounds = self.search_region, tol=self.search_precision)
            if result.fun < next_cost:
                next_params = result.x
                next_cost = result.fun
        return next_params

    def run(self):
        '''
        Starts running the Gaussian process learner. When the new parameters event is triggered, reads the cost information provided and updates the Gaussian process with the information. Then searches the Gaussian process for new optimal parameters to test based on the biased cost. Parameters to test next are put on the output parameters queue.
        '''
        #logging to the main log file from a process (as apposed to a thread) in cpython is currently buggy on windows and/or python 2.7
        #current solution is to only log to the console for warning and above from a process
        self.log = mp.log_to_stderr(logging.WARNING)

        try:
            while not self.end_event.is_set():
                #self.log.debug('Learner waiting for new params event')
                self.save_archive()
                self.wait_for_new_params_event()
                #self.log.debug('Gaussian process learner reading costs')
                self.get_params_and_costs()
                self.fit_gaussian_process()
                for _ in range(self.generation_num):
                    self.log.debug('Gaussian process learner generating parameter:'+ str(self.params_count+1))
                    next_params = self.find_next_parameters()
                    self.params_out_queue.put(next_params)
                    if self.end_event.is_set():
                        raise LearnerInterrupt()
        except LearnerInterrupt:
            pass

        end_dict = {}
        if self.predict_global_minima_at_end:
            self.get_params_and_costs()
            self.fit_gaussian_process()
            self.find_global_minima()
            end_dict.update({'predicted_best_parameters':self.predicted_best_parameters,
                             'predicted_best_cost':self.predicted_best_cost,
                             'predicted_best_uncertainty':self.predicted_best_uncertainty})
        self.params_out_queue.put(end_dict)
        self._shut_down()
        self.log.debug('Ended Gaussian Process Learner')

    def predict_cost(self,params):
        '''
        Produces a prediction of cost from the gaussian process at params.

        Returns:
            float : Predicted cost at paramters
        '''
        return self.gaussian_process.predict(params[np.newaxis,:])

    def find_global_minima(self):
        '''
        Performs a quick search for the predicted global minima from the learner. Does not return any values, but creates the following attributes.

        Attributes:
            predicted_best_parameters (array): the parameters for the predicted global minima
            predicted_best_cost (float): the cost at the predicted global minima
            predicted_best_uncertainty (float): the uncertainty of the predicted global minima
        '''
        self.log.debug('Started search for predicted global minima.')

        self.predicted_best_parameters = None
        self.predicted_best_scaled_cost = float('inf')
        self.predicted_best_scaled_uncertainty = None

        search_params = []
        search_params.append(self.best_params)
        for _ in range(self.parameter_searches):
            search_params.append(self.min_boundary + nr.uniform(size=self.num_params) * self.diff_boundary)

        search_bounds = list(zip(self.min_boundary, self.max_boundary))
        for start_params in search_params:
            result = so.minimize(self.predict_cost, start_params, bounds = search_bounds, tol=self.search_precision)
            curr_best_params = result.x
            (curr_best_cost,curr_best_uncer) = self.gaussian_process.predict(curr_best_params[np.newaxis,:],return_std=True)
            if curr_best_cost<self.predicted_best_scaled_cost:
                self.predicted_best_parameters = curr_best_params
                self.predicted_best_scaled_cost = curr_best_cost
                self.predicted_best_scaled_uncertainty = curr_best_uncer

        self.predicted_best_cost = self.cost_scaler.inverse_transform(self.predicted_best_scaled_cost)
        self.predicted_best_uncertainty = self.predicted_best_scaled_uncertainty * self.cost_scaler.scale_

        self.archive_dict.update({'predicted_best_parameters':self.predicted_best_parameters,
                                  'predicted_best_scaled_cost':self.predicted_best_scaled_cost,
                                  'predicted_best_scaled_uncertainty':self.predicted_best_scaled_uncertainty,
                                  'predicted_best_cost':self.predicted_best_cost,
                                  'predicted_best_uncertainty':self.predicted_best_uncertainty})

        self.has_global_minima = True
        self.log.debug('Predicted global minima found.')


class NeuralNetLearner(MachineLearner, mp.Process):
    '''
    Learner that uses a neural network for function approximation.

    Args:
        params_out_queue (queue): Queue for parameters sent to controller.
        costs_in_queue (queue): Queue for costs.
        end_event (event): Event to trigger end of learner.

    Keyword Args:
        nn_training_filename (Optional [str]): The name of a learner archive
            from a previous optimization from which to extract past results for
            use in the current optimization. If `None`, no past results will be
            used. Default `None`.
        nn_training_file_type (Optional [str]): File type of the training
            archive. Can be `'txt'`, `'pkl'`, `'mat'`, or `None`. If set to
            `None`, then the file type will be determined automatically. This
            argument has no effect if `nn_training_filename` is set to `None`.
            Default `None`.
        trust_region (Optional [float or array]): The trust region defines the maximum distance the learner will travel from the current best set of parameters. If None, the learner will search everywhere. If a float, this number must be between 0 and 1 and defines maximum distance the learner will venture as a percentage of the boundaries. If it is an array, it must have the same size as the number of parameters and the numbers define the maximum absolute distance that can be moved along each direction.
        default_bad_cost (Optional [float]): If a run is reported as bad and default_bad_cost is provided, the cost for the bad run is set to this default value. If default_bad_cost is None, then the worst cost received is set to all the bad runs. Default None.
        default_bad_uncertainty (Optional [float]): If a run is reported as bad and default_bad_uncertainty is provided, the uncertainty for the bad run is set to this default value. If default_bad_uncertainty is None, then the uncertainty is set to a tenth of the best to worst cost range. Default None.
        minimum_uncertainty (Optional [float]): The minimum uncertainty associated with provided costs. Must be above zero to avoid fitting errors. Default 1e-8.
        predict_global_minima_at_end (Optional [bool]): If True finds the global minima when the learner is ended. Does not if False. Default True.

    Attributes:
        all_params (array): Array containing all parameters sent to learner.
        all_costs (array): Array containing all costs sent to learner.
        all_uncers (array): Array containing all uncertainties sent to learner.
        scaled_costs (array): Array contaning all the costs scaled to have zero mean and a standard deviation of 1.
        bad_run_indexs (list): list of indexes to all runs that were marked as bad.
        best_cost (float): Minimum received cost, updated during execution.
        best_params (array): Parameters of best run. (reference to element in params array).
        best_index (int): index of the best cost and params.
        worst_cost (float): Maximum received cost, updated during execution.
        worst_index (int): index to run with worst cost.
        cost_range (float): Difference between worst_cost and best_cost
        generation_num (int): Number of sets of parameters to generate each generation. Set to 5.
        noise_level_history (list): List of noise levels found after each fit.
        cost_count (int): Counter for the number of costs, parameters and uncertainties added to learner.
        params_count (int): Counter for the number of parameters asked to be evaluated by the learner.
        neural_net (NeuralNet): Neural net that is fitted to data and used to make predictions.
        cost_scaler (StandardScaler): Scaler used to normalize the provided costs.
        cost_scaler_init_index (int): The number of params to use to initialise cost_scaler.
        has_trust_region (bool): Whether the learner has a trust region.
    '''
    _ARCHIVE_TYPE = 'neural_net_learner'

    def __init__(self,
                 nn_training_filename =None,
                 nn_training_file_type =None,
                 **kwargs):

        if nn_training_filename is not None:
            super(NeuralNetLearner,self).__init__(
                training_filename=nn_training_filename,
                training_file_type=nn_training_file_type,
                **kwargs
            )
        else:
            super(NeuralNetLearner,self).__init__(**kwargs)

        # Parameters that should only be loaded if a training archive was
        # provided and it has the same learner type and min/max boundaries.
        same_learner_type = self._learner_type_matches_training_archive
        same_boundaries = self._boundaries_match_training_archive
        if same_learner_type and same_boundaries:
            pass  # Will load parameters here later.
        else:
            pass  # Will set default parameter values here later.

        # Set training file directory to None for now since no nets will be
        # loaded here. The NeuralNetVisualizer will overwrite this when it needs
        # to load the nets.
        self.nn_training_file_dir = None

        # The scaler will be initialised when we're ready to fit it
        self.cost_scaler = None
        self.cost_scaler_init_index = None

        #Constants, limits and tolerances
        self.num_nets = 3
        self.generation_num = 3

        self.archive_dict.update({'archive_type':self._ARCHIVE_TYPE,
                                  'generation_num':self.generation_num,
                                  'search_precision':self.search_precision,
                                  'parameter_searches':self.parameter_searches,
                                  'bad_uncer_frac':self.bad_uncer_frac,
                                  'trust_region':self.trust_region,
                                  'has_trust_region':self.has_trust_region,
                                  'predict_global_minima_at_end':self.predict_global_minima_at_end})

        #Remove logger so neural net can be safely picked for multiprocessing on Windows
        self.log = None

    def _construct_net(self):
        self.neural_net = [
            mlnn.NeuralNet(
                num_params=self.num_params,
                learner_archive_dir=self.learner_archive_dir,
                start_datetime=self.start_datetime)
            for _ in range(self.num_nets)
        ]

    def _init_cost_scaler(self):
        '''
        Initialises the cost scaler. cost_scaler_init_index must be set.
        '''
        self.cost_scaler = skp.StandardScaler(with_mean=False, with_std=False)
        self.cost_scaler.fit(self.all_costs[:self.cost_scaler_init_index,np.newaxis])

    def create_neural_net(self):
        '''
        Creates the neural net. Must be called from the same process as fit_neural_net, predict_cost and predict_costs_from_param_array.
        '''
        self._construct_net()
        for n in self.neural_net:
            n.init()

    def import_neural_net(self):
        '''
        Imports neural net parameters from the training dictionary provided at construction. Must be called from the same process as fit_neural_net, predict_cost and predict_costs_from_param_array. You must call exactly one of this and create_neural_net before calling other methods.
        '''
        if not self.training_dict:
            raise ValueError
        self._construct_net()
        for i, n in enumerate(self.neural_net):
            n.load(self.training_dict['net_' + str(i)],
                   extra_search_dirs=[self.nn_training_file_dir])

    def _fit_neural_net(self,index):
        '''
        Fits a neural net to the data.

        cost_scaler must have been fitted before calling this method.
        '''
        self.scaled_costs = self.cost_scaler.transform(self.all_costs[:,np.newaxis])[:,0]

        self.neural_net[index].fit_neural_net(self.all_params, self.scaled_costs)

    def predict_cost(self,params,net_index=None):
        '''
        Produces a prediction of cost from the neural net at params.

        Returns:
            float : Predicted cost at paramters
        '''
        if net_index is None:
            net_index = nr.randint(self.num_nets)
        return self.neural_net[net_index].predict_cost(params)

    def predict_cost_gradient(self,params,net_index=None):
        '''
        Produces a prediction of the gradient of the cost function at params.

        Returns:
            float : Predicted gradient at paramters
        '''
        if net_index is None:
            net_index = nr.randint(self.num_nets)
        # scipy.optimize.minimize doesn't seem to like a 32-bit Jacobian, so we convert to 64
        return self.neural_net[net_index].predict_cost_gradient(params).astype(np.float64)


    def predict_costs_from_param_array(self,params,net_index=None):
        '''
        Produces a prediction of costs from an array of params.

        Returns:
            float : Predicted cost at paramters
        '''
        # TODO: Can do this more efficiently.
        return [self.predict_cost(param,net_index) for param in params]


    def wait_for_new_params_event(self):
        '''
        Waits for a new parameters event and starts a new parameter generation cycle. Also checks end event and will break if it is triggered.
        '''
        while not self.end_event.is_set():
            if self.new_params_event.wait(timeout=self.learner_wait):
                self.new_params_event.clear()
                break
            else:
                continue
        else:
            self.log.debug('NeuralNetLearner end signal received. Ending')
            raise LearnerInterrupt


    def get_params_and_costs(self):
        '''
        Get the parameters and costs from the queue and place in their appropriate all_[type] arrays. Also updates bad costs, best parameters, and search boundaries given trust region.
        '''
        new_params = []
        new_costs = []
        new_uncers = []
        new_bads = []
        update_bads_flag = False

        first_dequeue = True
        while True:
            if first_dequeue:
                try:
                    # Block for 1s, because there might be a race with the event being set.
                    (param, cost, uncer, bad) = self.costs_in_queue.get(block=True, timeout=1)
                    first_dequeue = False
                except mlu.empty_exception:
                    self.log.error('Neural network asked for new parameters but no new costs were provided after 1s.')
                    raise ValueError
            else:
                try:
                    (param, cost, uncer, bad) = self.costs_in_queue.get_nowait()
                except mlu.empty_exception:
                    break

            self.costs_count +=1

            if bad:
                new_bads.append(self.costs_count-1)
                if self.bad_defaults_set:
                    cost = self.default_bad_cost
                    uncer = self.default_bad_uncertainty
                else:
                    cost = self.worst_cost
                    uncer = self.cost_range*self.bad_uncer_frac

            message = (param, cost, uncer, bad)
            param, cost, uncer, bad = self._parse_cost_message(message)

            uncer = max(uncer, self.minimum_uncertainty)

            cost_change_flag = False
            if cost > self.worst_cost:
                self.worst_cost = cost
                self.worst_index = self.costs_count-1
                cost_change_flag = True
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_params = param
                self.best_index =  self.costs_count-1
                cost_change_flag = True
            if cost_change_flag:
                self.cost_range = self.worst_cost - self.best_cost
                if not self.bad_defaults_set:
                    update_bads_flag = True

            new_params.append(param)
            new_costs.append(cost)
            new_uncers.append(uncer)


        if self.all_params.size==0:
            self.all_params = np.array(new_params, dtype=float)
            self.all_costs = np.array(new_costs, dtype=float)
            self.all_uncers = np.array(new_uncers, dtype=float)
        else:
            self.all_params = np.concatenate((self.all_params, np.array(new_params, dtype=float)))
            self.all_costs = np.concatenate((self.all_costs, np.array(new_costs, dtype=float)))
            self.all_uncers = np.concatenate((self.all_uncers, np.array(new_uncers, dtype=float)))

        self.bad_run_indexs.append(new_bads)

        if self.all_params.shape != (self.costs_count,self.num_params):
            self.log('Saved NN params are the wrong size. THIS SHOULD NOT HAPPEN:' + repr(self.all_params))
        if self.all_costs.shape != (self.costs_count,):
            self.log('Saved NN costs are the wrong size. THIS SHOULD NOT HAPPEN:' + repr(self.all_costs))
        if self.all_uncers.shape != (self.costs_count,):
            self.log('Saved NN uncertainties are the wrong size. THIS SHOULD NOT HAPPEN:' + repr(self.all_uncers))

        if update_bads_flag:
            self.update_bads()

        self.update_search_region()

    def update_bads(self):
        '''
        Best and/or worst costs have changed, update the values associated with bad runs accordingly.
        '''
        for index in self.bad_run_indexs:
            self.all_costs[index] = self.worst_cost
            self.all_uncers[index] = self.cost_range*self.bad_uncer_frac

    def update_search_region(self):
        '''
        If trust boundaries is not none, updates the search boundaries based on the defined trust region.
        '''
        if self.has_trust_region:
            self.search_min = np.maximum(self.best_params - self.trust_region, self.min_boundary)
            self.search_max = np.minimum(self.best_params + self.trust_region, self.max_boundary)
            self.search_diff = self.search_max - self.search_min
            self.search_region = list(zip(self.search_min, self.search_max))

    def update_search_params(self):
        '''
        Update the list of parameters to use for the next search.
        '''
        self.search_params = []
        self.search_params.append(self.best_params)
        for _ in range(self.parameter_searches):
            self.search_params.append(self.search_min + nr.uniform(size=self.num_params) * self.search_diff)

    def update_archive(self):
        '''
        Update the archive.
        '''
        super(NeuralNetLearner, self).update_archive()
        new_values_dict = {
            'cost_scaler_init_index':self.cost_scaler_init_index,
        }
        self.archive_dict.update(new_values_dict)
        if self.neural_net:
            for i,n in enumerate(self.neural_net):
                self.archive_dict.update({'net_'+str(i):n.save()})

    def find_next_parameters(self, net_index=None):
        '''
        Returns next parameters to find. Increments counters appropriately.

        Return:
            next_params (array): Returns next parameters from cost search.
        '''
        if net_index is None:
            net_index = nr.randint(self.num_nets)

        self.params_count += 1
        self.update_search_params()
        next_params = None
        next_cost = float('inf')
        self.neural_net[net_index].start_opt()
        for start_params in self.search_params:
            result = so.minimize(fun = lambda x: self.predict_cost(x, net_index),
                                 x0 = start_params,
                                 jac = lambda x: self.predict_cost_gradient(x, net_index),
                                 bounds = self.search_region,
                                 tol = self.search_precision)
            if result.fun < next_cost:
                next_params = result.x
                next_cost = result.fun
        self.neural_net[net_index].stop_opt()
        self.log.debug("Suggesting params " + str(next_params) + " with predicted cost: "
                + str(next_cost))
        return next_params

    def run(self):
        '''
        Starts running the neural network learner. When the new parameters event is triggered, reads the cost information provided and updates the neural network with the information. Then searches the neural network for new optimal parameters to test based on the biased cost. Parameters to test next are put on the output parameters queue.
        '''
        #logging to the main log file from a process (as apposed to a thread) in cpython is currently buggy on windows and/or python 2.7
        #current solution is to only log to the console for warning and above from a process
        self.log = mp.log_to_stderr(logging.WARNING)

        # The network needs to be created in the same process in which it runs
        self.create_neural_net()

        # We cycle through our different nets to generate each new set of params. This keeps track
        # of the current net.
        net_index = 0

        try:
            while not self.end_event.is_set():
                self.log.debug('Learner waiting for new params event')
                # TODO: Not doing this because it's slow. Is it necessary?
                #self.save_archive()
                self.wait_for_new_params_event()
                self.log.debug('NN learner reading costs')
                self.get_params_and_costs()
                if self.cost_scaler_init_index is None:
                    self.cost_scaler_init_index = len(self.all_costs)
                    self._init_cost_scaler()
                # Now we need to generate generation_num new param sets, by iterating over our
                # nets. We want to fire off new params as quickly as possible, so we don't train a
                # net until we actually need to use it. But we need to make sure that each net gets
                # trained exactly once, regardless of how many times it's used to generate new
                # params.
                num_nets_trained = 0
                for _ in range(self.generation_num):
                    if num_nets_trained < self.num_nets:
                        self._fit_neural_net(net_index)
                        num_nets_trained += 1

                    self.log.debug('Neural network learner generating parameter:'+ str(self.params_count+1))
                    next_params = self.find_next_parameters(net_index)
                    net_index = (net_index + 1) % self.num_nets
                    self.params_out_queue.put(next_params)
                    if self.end_event.is_set():
                        raise LearnerInterrupt()
                # Train any nets that haven't been trained yet.
                for i in range(self.num_nets - num_nets_trained):
                    self._fit_neural_net((net_index + i) % self.num_nets)

        except LearnerInterrupt:
            pass
        end_dict = {}
        if self.predict_global_minima_at_end:
            if not self.costs_in_queue.empty():
                # There are new parameters, get them.
                self.get_params_and_costs()
            # TODO: Somehow support predicting minima from all nets, rather than just net 0.
            self._fit_neural_net(0)
            self.find_global_minima(0)
            end_dict.update({'predicted_best_parameters':self.predicted_best_parameters,
                             'predicted_best_cost':self.predicted_best_cost})
        self.params_out_queue.put(end_dict)
        self._shut_down()
        for n in self.neural_net:
            n.destroy()
        self.log.debug('Ended neural network learner')

    def find_global_minima(self,net_index=None):
        '''
        Performs a quick search for the predicted global minima from the learner. Does not return any values, but creates the following attributes.

        Attributes:
            predicted_best_parameters (array): the parameters for the predicted global minima
            predicted_best_cost (float): the cost at the predicted global minima
        '''
        if net_index is None:
            net_index = nr.randint(self.num_nets)
        self.log.debug('Started search for predicted global minima.')

        self.predicted_best_parameters = None
        self.predicted_best_scaled_cost = float('inf')

        search_params = []
        search_params.append(self.best_params)
        for _ in range(self.parameter_searches):
            search_params.append(self.min_boundary + nr.uniform(size=self.num_params) * self.diff_boundary)

        search_bounds = list(zip(self.min_boundary, self.max_boundary))
        for start_params in search_params:
            result = so.minimize(fun = lambda x: self.predict_cost(x, net_index),
                                 x0 = start_params,
                                 jac = lambda x: self.predict_cost_gradient(x, net_index),
                                 bounds = search_bounds,
                                 tol = self.search_precision)
            curr_best_params = result.x
            curr_best_cost = result.fun
            if curr_best_cost<self.predicted_best_scaled_cost:
                self.predicted_best_parameters = curr_best_params
                self.predicted_best_scaled_cost = curr_best_cost

        self.predicted_best_cost = float(self.cost_scaler.inverse_transform([self.predicted_best_scaled_cost]))
        self.archive_dict.update({'predicted_best_parameters':self.predicted_best_parameters,
                                  'predicted_best_scaled_cost':self.predicted_best_scaled_cost,
                                  'predicted_best_cost':self.predicted_best_cost})

        self.has_global_minima = True
        self.log.debug('Predicted global minima found.')



    # Methods for debugging/analysis.

    def get_losses(self):
        all_losses = []
        for n in self.neural_net:
            all_losses.append(n.get_losses())
        return all_losses
