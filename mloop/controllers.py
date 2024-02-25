'''
Module of all the controllers used in M-LOOP. The controllers, as the name suggests, control the interface to the experiment and all the learners employed to find optimal parameters.
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import os
import datetime
from importlib import import_module
import logging
import traceback

import numpy as np

from mloop import __version__
import mloop.utilities as mlu
import mloop.learners as mll
import mloop.interfaces as mli

default_controller_archive_filename = 'controller_archive'
default_controller_archive_file_type = 'txt'

class ControllerInterrupt(Exception):
    '''
    Exception that is raised when the controlled is ended with the end flag or event.
    '''
    def __init__(self):
        super(ControllerInterrupt,self).__init__()

def create_controller(interface,
                      controller_type='gaussian_process',
                      **controller_config_dict):
    '''
    Start the controller with the options provided.
    Args:
        interface (interface): Interface with queues and events to be passed to controller
    Keyword Args:
        controller_type (Optional [str]): Defines the type of controller can be 'random', 'nelder', 'gaussian_process' or 'neural_net'. Alternatively, the controller can belong to an external module, in which case this parameter should be 'module_name:controller_name'. Defaults to 'gaussian_process'.
        **controller_config_dict : Options to be passed to controller.
    Returns:
        Controller : threadable object which must be started with start() to get the controller running.
    Raises:
        ValueError : if controller_type is an unrecognized string
    '''
    log = logging.getLogger(__name__)

    controller_type = str(controller_type)
    if controller_type=='gaussian_process':
        controller = GaussianProcessController(interface, **controller_config_dict)
    elif controller_type=='neural_net':
        controller = NeuralNetController(interface, **controller_config_dict)
    elif controller_type=='differential_evolution':
        controller = DifferentialEvolutionController(interface, **controller_config_dict)
    elif controller_type=='nelder_mead':
        controller = NelderMeadController(interface, **controller_config_dict)
    elif controller_type=='random':
        controller = RandomController(interface, **controller_config_dict)
    else:
        # If `controller_type` doesn't match any of the built-in controllers,
        # looks for an external controller with matching name.
        # The `controller_type` should be "module_name:controller_name".
        parts = controller_type.split(":")
        if (len(parts)==2) and all([len(part)>0 for part in parts]):
            try:
                module = import_module(parts[0])
                constructor = getattr(module, parts[1])
            except ModuleNotFoundError:
                log.error(f"Unknown module: {parts[0]}")
                raise ValueError
            except AttributeError:
                log.error(f"Unknown controller type: {parts[0]}.{parts[1]}")
                raise ValueError
            except:
                log.error('Unknown controller type:' + repr(controller_type))
                raise ValueError
            controller = constructor(interface, **controller_config_dict)
        else:
            log.error('Unknown controller type:' + repr(controller_type))
            raise ValueError

    return controller

class Controller():
    '''
    Abstract class for controllers.

    The controller controls the entire M-LOOP process. The controllers for each
    algorithm all inherit from this class. The class stores a variety of data
    which all algorithms use and also includes all of the archiving and saving
    features.
    
    In order to implement your own controller class the minimum requirement is
    to add a learner to the learner variable and implement the
    `next_parameters()` method where you provide the appropriate information to
    the learner and get the next parameters. See the `RandomController` for a
    simple implementation of a controller. Note the first three keywords are all
    possible halting conditions for the controller. If any of them are satisfied
    the controller will halt (meaning an OR condition is used). This base class
    also creates an empty attribute `self.learner`. The simplest way to make a
    working controller is to assign a learner of some kind to this variable, and
    add appropriate queues and events from it.

    Args:
        interface (interface): The interface process. It is run by the
            controller.

    Keyword Args:
        max_num_runs (Optional [float]): The number of runs before the
            controller stops. If set to `float('+inf')` the controller will run
            forever assuming no other halting conditions are met. Default
            `float('+inf')`, meaning the controller will run until another
            halting condition is met.
        target_cost (Optional [float]): The target cost for the run. If a run
            achieves a cost lower than the target, the controller is stopped.
            Default `float('-inf')`, meaning the controller will run until
            another halting condition is met.
        max_num_runs_without_better_params (Optional [float]): The optimization
            will halt if the number of consecutive runs without improving over
            the best measured value thus far exceeds this number. Default
            `float('+inf')`, meaning the controller will run until another
            halting condition is met.
        max_duration (Optional [float]): The maximum duration of the
            optimization, in seconds of wall time. The actual duration may
            exceed this value slightly, but no new iterations will start after
            `max_duration` seconds have elapsed since `start_datetime`. Default
            is `float('+inf')`, meaning the controller will run until another
            halting condition is met.
        controller_archive_filename (Optional [string]): Filename for archive.
            The archive contains costs, parameter history and other details
            depending on the controller type. Default
            `'controller_archive'`.
        controller_archive_file_type (Optional [string]): File type for archive.
            Can be either `'txt'` for a human readable text file, `'pkl'` for a
            python pickle file, `'mat'` for a matlab file, or `None` to forgo
            saving a controller archive. Default `'txt'`.
        archive_extra_dict (Optional [dict]): A dictionary with any extra
            variables that are to be saved to the archive. If `None`, nothing is
            added. Default `None`.
        start_datetime (Optional datetime): Datetime for when the controller was
            started.

    Attributes:
        params_out_queue (queue): Queue for parameters to next be run by the
            experiment.
        costs_in_queue (queue): Queue for costs (and other details) that have
            been returned by experiment.
        interface_error_queue (queue): Queue for returning errors encountered by
            the interface.
        end_interface (event): Event used to trigger the end of the interface.
        learner (None): The placeholder for the learner. Creating this variable
            is the minimum requirement to make a working controller class.
        learner_params_queue (queue): The parameters queue for the learner.
        learner_costs_queue (queue): The costs queue for the learner.
        end_learner (event): Event used to trigger the end of the learner.
        num_in_costs (int): Counter for the number of costs received.
        num_out_params (int): Counter for the number of parameters received.
        out_params (list): List of all parameters sent out by controller.
        out_extras (list): Any extras associated with the output parameters.
        in_costs (list): List of costs received by controller.
        in_uncers (list): List of uncertainties received by controller.
        best_cost (float): The lowest, and best, cost received by the learner.
        best_uncer (float): The uncertainty associated with the best cost.
        best_params (array): The best parameters received by the learner.
        best_index (float): The run number that produced the best cost.
        best_in_extras (dict): The extra entries returned in the cost dict from
            the best run.
    '''

    def __init__(
        self,
        interface,
        max_num_runs=float('+inf'),
        target_cost=float('-inf'),
        max_num_runs_without_better_params=float('+inf'),
        max_duration=float('+inf'),
        controller_archive_filename=default_controller_archive_filename,
        controller_archive_file_type=default_controller_archive_file_type,
        archive_extra_dict = None,
        start_datetime = None,
        **kwargs,
    ):

        # Make the logger.
        self.remaining_kwargs = mlu._config_logger(start_datetime=start_datetime, **kwargs)
        self.log = logging.getLogger(__name__)

        # Variables that are included in the controller archive.
        self.num_in_costs = 0
        self.num_out_params = 0
        self.out_params = []
        self.out_type = []
        self.out_extras = []
        self.in_costs = []
        self.in_uncers = []
        self.in_bads = []
        self.in_extras = []
        self.best_cost = float('inf')
        self.best_uncer = float('nan')
        self.best_index = float('nan')
        self.best_params = float('nan')
        self.best_in_extras = {}

        # Variables that are used internally.
        self.last_out_params = mlu.queue.Queue()
        self.curr_params = None
        self.curr_cost = None
        self.curr_uncer = None
        self.curr_bad = None
        self.curr_extras = None
        self.num_last_best_cost = 0
        self.halt_reasons = []

        # Constants.
        self.controller_wait = float(1)

        # Learner-related variables.
        self.learner_params_queue = None
        self.learner_costs_queue = None
        self.end_learner = None
        self.learner = None

        # Variables set by user.

        # Store the interface and extract important attributes from it.
        if isinstance(interface, mli.Interface):
            self.interface = interface
        else:
            self.log.error('interface is not a Interface as defined in the M-LOOP package.')
            raise TypeError

        self.params_out_queue = interface.params_out_queue
        self.costs_in_queue = interface.costs_in_queue
        self.interface_error_queue = interface.interface_error_queue
        self.end_interface = interface.end_event

        # Halting options.
        self.max_num_runs = float(max_num_runs)
        if self.max_num_runs <= 0:
            msg = f"max_num_runs must be greater than zero but was {max_num_runs}."
            self.log.error(msg)
            raise ValueError(msg)
        self.target_cost = float(target_cost)
        self.max_num_runs_without_better_params = float(max_num_runs_without_better_params)
        if self.max_num_runs_without_better_params <= 0:
            msg = f"max_num_runs_without_better_params must be greater than zero but was {max_num_runs_without_better_params}."
            self.log.error(msg)
            raise ValueError(msg)
        self.max_duration = float(max_duration)
        if self.max_duration <= 0:
            msg = f"max_duration must be greater than zero but was {max_duration}."
            self.log.error(msg)
            raise ValueError(msg)

        # Other options.
        if start_datetime is None:
            start_datetime = datetime.datetime.now()
        if isinstance(start_datetime, datetime.datetime):
            self.start_datetime = start_datetime
        else:
            msg = (
                "start_datetime must be of type datetime.datetime but was "
                f"{start_datetime} (type: {type(start_datetime)})."
            )
            self.log.error(msg)
            raise ValueError(msg)

        if mlu.check_file_type_supported(controller_archive_file_type):
            self.controller_archive_file_type = controller_archive_file_type
        else:
            self.log.error('File in type is not supported:' + repr(controller_archive_file_type))
            raise ValueError
        if controller_archive_filename is None:
            self.controller_archive_filename = None
        else:
            # Store self.controller_archive_filename without any path, but
            # include any path components in controller_archive_filename when
            # constructing the full path.
            controller_archive_filename = str(controller_archive_filename)
            self.controller_archive_filename = os.path.basename(controller_archive_filename)
            filename_suffix = mlu.generate_filename_suffix(
                self.controller_archive_file_type,
                file_datetime=self.start_datetime,
            )
            filename = controller_archive_filename + filename_suffix
            self.total_archive_filename = os.path.join(mlu.archive_foldername, filename)

            # Include any path info from controller_archive_filename when
            # creating directory for archive files.]
            archive_dir = os.path.dirname(self.total_archive_filename)
            if not os.path.exists(archive_dir):
                os.makedirs(archive_dir)

        self.archive_dict = {'mloop_version':__version__,
                             'archive_type':'controller',
                             'num_out_params':self.num_out_params,
                             'out_params':self.out_params,
                             'out_type':self.out_type,
                             'out_extras':self.out_extras,
                             'in_costs':self.in_costs,
                             'in_uncers':self.in_uncers,
                             'in_bads':self.in_bads,
                             'in_extras':self.in_extras,
                             'max_num_runs':self.max_num_runs,
                             'start_datetime':mlu.datetime_to_string(self.start_datetime)}

        if archive_extra_dict is not None:
            self.archive_dict.update(archive_extra_dict)

        self.log.debug('Controller init completed.')

    def check_end_conditions(self):
        '''
        Check whether any of the end contions have been met.

        In particular this method check for any of the following conditions:

        * If the number of iterations has reached `max_num_runs`.
        * If the `target_cost` has been reached.
        * If `max_num_runs_without_better_params` iterations in a row have
          occurred without any improvement.
        * If `max_duration` seconds or more has elapsed since `start_datetime`.

        Returns:
            bool: `True`, if the controller should continue or `False` if one or
                more halting conditions have been met and the controller should
                end.
        '''
        # Determine how long it has been since self.start_datetime.
        duration = datetime.datetime.now() - self.start_datetime
        duration = duration.total_seconds()  # Convert to seconds.

        # Check all of the halting conditions. Many if statements are used
        # instead of elif blocks so that we can mark if the optimization halted
        # for more than one reason.
        if self.num_in_costs >= self.max_num_runs:
            self.halt_reasons.append('Maximum number of runs reached.')
        if self.best_cost <= self.target_cost:
            self.halt_reasons.append('Target cost reached.')
        if self.num_last_best_cost >= self.max_num_runs_without_better_params:
            self.halt_reasons.append(
                'Maximum number of runs without better params reached.'
            )
        if duration > self.max_duration:
            self.halt_reasons.append('Maximum duration reached.')
        
        # The optimization should only continue if self.halt_reasons is empty.
        return not bool(self.halt_reasons)

    def _update_controller_with_learner_attributes(self):
        '''
        Update the controller with properties from the learner.
        '''
        self.learner_params_queue = self.learner.params_out_queue
        self.learner_costs_queue = self.learner.costs_in_queue
        self.end_learner = self.learner.end_event
        self.remaining_kwargs = self.learner.remaining_kwargs
        self.num_params = self.learner.num_params
        self.min_boundary = self.learner.min_boundary
        self.max_boundary = self.learner.max_boundary
        self.param_names = self.learner.param_names

        self.archive_dict.update(
            {
                'num_params': self.num_params,
                'min_boundary': self.min_boundary,
                'max_boundary': self.max_boundary,
                'param_names': self.param_names,
            }
        )


    def _put_params_and_out_dict(self, params, param_type=None, **kwargs):
        '''
        Send parameters to queue with optional additional keyword arguments.

        This method also saves sent variables in appropriate storage arrays.

        Args:
            params (array): Array of values to be experimentally tested.
            param_type (Optional, str): The learner type which generated the
                parameter values. Because some learners use other learners as
                trainers, the parameter type can be different for different
                iterations during a given optimization. This value will be
                stored in `self.out_type` and in the `out_type` list in the
                controller archive. If `None`, then it will be set to
                `self.learner.OUT_TYPE`. Default `None`.
        Keyword Args:
            **kwargs: Any additional keyword arguments will be stored in
                `self.out_extras` and in the `out_extras` list in the controller
                archive.
        '''
        # Set default values if needed.
        if param_type is None:
            param_type = self.learner.OUT_TYPE

        # Do one last check to ensure parameter values are within the allowed
        # limits before sending those values to the interface.
        params = self._enforce_boundaries(params)

        # Send the parameters to the interface and update various attributes.
        out_dict = {'params':params}
        out_dict.update(kwargs)
        self.params_out_queue.put(out_dict)
        self.num_out_params += 1
        self.last_out_params.put(params)
        self.out_params.append(params)
        self.out_extras.append(kwargs)
        self.out_type.append(param_type)
        self.log.info('params ' + str(params))
        #self.log.debug('Put params num:' + repr(self.num_out_params ))

    def _enforce_boundaries(self, params):
        '''
        Enforce the minimum and maximum parameter boundaries.

        If the values in params extend outside of the allowed boundaries set by
        `self.min_boundary` and `self.max_boundary` by a nontrivial amount, then
        this method will raise an `RuntimeError`.

        To avoid numerical precision problems, this method actually gently clips
        values which barely exceed the boundaries. This is because variables are
        internally scaled and thus may very slightly violate the boundaries
        after being unscaled. If a parameter's value only slightly exceeds the
        boundaries, then its value will be set equal to the boundary. If a value
        exceeds the boundary by a nontrivial amount, then a `RuntimeError` will
        be raised.

        Note that although this method is forgiving about input parameter values
        very slightly exceeding the boundaries, it is completely strict about
        returning parameter values which obey the boundaries. Thus it is safe to
        assume that the returned values are within the range set by
        `self.min_boundary` and `self.max_boundary` (inclusively).

        Args:
            params (array): Array of values to be experimentally tested.

        Raises:
            RuntimeError: A `RuntimeError` is raised if any value in `params`
                exceeds the parameter value boundaries by a nontrivial amount.

        Returns:
            array: The input `params`, except that any values which slightly
                exceed the boundaries will have been clipped to stay within the
                boundaries exactly.
        '''
        # Check for any values barely below the minimum boundary.
        is_below = (params < self.min_boundary)
        is_close = np.isclose(params, self.min_boundary)
        barely_below = np.logical_and(is_below, is_close)
        # The line below leaves most params entries untouched. It just takes the
        # ones which are barely below the boundary and moves them up to the
        # minimum boundary.
        params = np.where(barely_below, self.min_boundary, params)

        # Perform a similar procedure for the maximum boundary.
        is_above = (params > self.max_boundary)
        is_close = np.isclose(params, self.max_boundary)
        barely_above = np.logical_and(is_above, is_close)
        params = np.where(barely_above, self.max_boundary, params)

        # Ensure that all values are within the boundaries now. If a parameter
        # was well outside the boundaries then the above should not have changed
        # its value and the checks below will be violated.
        if not np.all(params >= self.min_boundary):
            msg = (
                f"Minimum boundary violated. Parameter values are {params}, "
                f"minimum boundary is {self.min_boundary}, and "
                f"(params - boundary) is {params - self.min_boundary}."
            )
            self.log.error(msg)
            raise RuntimeError(msg)
        if not np.all(params <= self.max_boundary):
            msg = (
                f"Maximum boundary violated. Parameter values are {params}, "
                f"maximum boundary is {self.max_boundary}, and "
                f"(boundary - params) is {self.max_boundary - params})."
            )
            self.log.error(msg)
            raise RuntimeError(msg)

        return params

    def _get_cost_and_in_dict(self):
        '''
        Get cost, uncertainty, parameters, bad and extra data from experiment.

        This method stores results in lists and also puts data in the
        appropriate 'current' variables. This method doesn't return anything and
        instead stores all of its results in the internal storage arrays and the
        'current' variables.
        
        If the interface encounters an error, it will pass the error to the
        controller here so that the error can be re-raised in the controller's
        thread (note that the interface runs in a separate thread).
        '''
        while True:
            try:
                in_dict = self.costs_in_queue.get(True, self.controller_wait)
            except mlu.empty_exception:
                # Check for an error from the interface.
                try:
                    err = self.interface_error_queue.get_nowait()
                except mlu.empty_exception:
                    # The interface didn't send an error, so go back to waiting
                    # for results.
                    continue
                else:
                    # Log and re-raise the error sent by the interface.
                    msg = 'The interface raised an error with traceback:\n'
                    msg = msg + '\n'.join(
                        traceback.format_tb(err.__traceback__),
                    )
                    self.log.error(msg)
                    raise err
            else:
                # Got a cost dict, so exit this while loop.
                break

        self.num_in_costs += 1
        self.num_last_best_cost += 1

        if not ('cost' in in_dict) and (not ('bad' in in_dict) or not in_dict['bad']):
            self.log.error('You must provide at least the key cost or the key bad with True.')
            raise ValueError
        try:
            self.curr_cost = float(in_dict.pop('cost',float('nan')))
            self.curr_uncer = float(in_dict.pop('uncer',0))
            self.curr_bad = bool(in_dict.pop('bad',False))
            self.curr_extras = in_dict
        except ValueError:
            self.log.error('One of the values you provided in the cost dict could not be converted into the right type.')
            raise
        if self.curr_bad and ('cost' in in_dict):
            self.log.warning('The cost provided with the bad run will be saved, but not used by the learners.')

        self.in_costs.append(self.curr_cost)
        self.in_uncers.append(self.curr_uncer)
        self.in_bads.append(self.curr_bad)
        self.in_extras.append(self.curr_extras)
        self.curr_params = self.last_out_params.get()
        if self.curr_cost < self.best_cost:
            self.best_cost = self.curr_cost
            self.best_uncer = self.curr_uncer
            self.best_index =  self.num_in_costs - 1  # -1 for zero-indexing.
            self.best_params = self.curr_params
            self.best_in_extras = self.curr_extras
            self.num_last_best_cost = 0
        if self.curr_bad:
            self.log.info('bad run')
        else:
            self.log.info('cost ' + str(self.curr_cost) + ' +/- ' + str(self.curr_uncer))

    def save_archive(self):
        '''
        Save the archive associated with the controller class. Only occurs if the filename for the archive is not None. Saves with the format previously set.
        '''
        if self.controller_archive_filename is not None:
            self.archive_dict.update(
                {
                    'num_in_costs': self.num_in_costs,
                    'num_out_params': self.num_out_params,
                    'best_cost': self.best_cost,
                    'best_uncer': self.best_uncer,
                    'best_params': self.best_params,
                    'best_index': self.best_index,
                    'best_in_extras': self.best_in_extras
                }
            )
            try:
                mlu.save_dict_to_file(self.archive_dict,self.total_archive_filename,self.controller_archive_file_type)
            except ValueError:
                self.log.error('Attempted to save with unknown archive file type, or some other value error.')
                raise
        else:
            self.log.debug('Did not save controller archive file.')

    def optimize(self):
        '''
        Optimize the experiment. This code learner and interface processes/threads are launched and appropriately ended.
        Starts both threads and catches kill signals and shuts down appropriately.
        '''
        log = logging.getLogger(__name__)

        try:
            log.info('Optimization started.')
            self._start_up()
            self._optimization_routine()
            log.info('Controller finished. Closing down M-LOOP. Please wait a moment...')
        except ControllerInterrupt:
            self.log.warning('Controller ended by interruption.')
        except (KeyboardInterrupt,SystemExit):
            log.warning('!!! Do not give the interrupt signal again !!! \n M-LOOP stopped with keyboard interupt or system exit. Please wait at least 1 minute for the threads to safely shut down. \n ')
            log.warning('Closing down controller.')
        except Exception:
            self.log.warning('Controller ended due to exception of some kind. Starting shut down...')
            self.halt_reasons.append('Error occurred.')
            self._shut_down()
            self.log.warning('Safely shut down. Below are results found before exception.')
            self.print_results()
            raise
        self._shut_down()
        self.print_results()
        self.log.info('M-LOOP Done.')

    def _start_up(self):
        '''
        Start the learner and interface threads/processes.
        '''
        self.learner.start()
        self.interface.start()

    def _shut_down(self):
        '''
        Shutdown and clean up resources of the controller. end the learners, queue_listener and make one last save of archive.
        '''
        self.log.debug('Learner end event set.')
        self.end_learner.set()
        self.log.debug('Interface end event set.')
        self.end_interface.set()
        #After 3 or 4 executions of mloop in same python environment, sometimes execution can be trapped here
        #Likely to be a bug with multiprocessing in python, but difficult to isolate.
        #current solution is to join with a timeout and kill if that fails
        self.learner.join()
        self.log.debug('Learner joined.')
        self.interface.join()
        self.log.debug('Interface joined.')
        self.save_archive()

    def print_results(self):
        '''
        Print results from optimization run to the logs
        '''
        self.log.info('Optimization ended because:')
        for reason in self.halt_reasons:
            self.log.info('\t* ' + reason)
        self.log.info('Results:')
        self.log.info('\t* Best parameters found: ' + str(self.best_params))
        self.log.info('\t* Best cost returned: ' + str(self.best_cost) + ' +/- ' + str(self.best_uncer))
        self.log.info('\t* Best run index: ' + str(self.best_index))
        if self.best_in_extras:
            self.log.info('\t* Best extras: ' + str(self.best_in_extras))

    def _optimization_routine(self):
        '''
        Runs controller main loop. Gives parameters to experiment and saves costs returned.
        '''
        self.log.debug('Start controller loop.')
        self.log.info('Run: ' + str(self.num_in_costs))
        next_params = self._first_params()
        self._put_params_and_out_dict(
            next_params,
            param_type=self.learner.OUT_TYPE,
        )
        self.save_archive()
        self._get_cost_and_in_dict()
        while self.check_end_conditions():
            self.log.info('Run: ' + str(self.num_in_costs))
            next_params = self._next_params()
            self._put_params_and_out_dict(
                next_params,
                param_type=self.learner.OUT_TYPE,
            )
            self.save_archive()
            self._get_cost_and_in_dict()
        self.log.debug('End controller loop.')
        # Send result of last run to learner to make sure that it makes it to
        # the learner archive.
        self._send_to_learner()

    def _first_params(self):
        '''
        Checks queue to get the first parameters.

        Returns:
            Parameters for first experiment
        '''
        return self.learner_params_queue.get()

    def _send_to_learner(self):
        '''
        Send the latest cost info the the learner.
        '''
        if self.curr_bad:
            cost = float('inf')
        else:
            cost = self.curr_cost
        message = (
            self.curr_params,
            cost,
            self.curr_uncer,
            self.curr_bad,
        )
        self.learner_costs_queue.put(message)

    def _next_params(self):
        '''
        Send latest cost info and get next parameters from the learner.
        '''
        self._send_to_learner()
        return self.learner_params_queue.get()

class RandomController(Controller):
    '''
    Controller that simply returns random variables for the next parameters. Costs are stored but do not influence future points picked.
    Args:
        params_out_queue (queue): Queue for parameters to next be run by experiment.
        costs_in_queue (queue): Queue for costs (and other details) that have been returned by experiment.
        **kwargs (Optional [dict]): Dictionary of options to be passed to Controller and Random Learner.
    '''
    def __init__(self, interface,**kwargs):

        super(RandomController,self).__init__(interface, **kwargs)
        self.learner = mll.RandomLearner(start_datetime = self.start_datetime,
                                         **self.remaining_kwargs)

        self._update_controller_with_learner_attributes()

        self.log.debug('Random controller init completed.')


class NelderMeadController(Controller):
    '''
    Controller for the Nelder–Mead solver. Suggests new parameters based on the Nelder–Mead algorithm. Can take no boundaries or hard boundaries. More details for the Nelder–Mead options are in the learners section.
    Args:
        params_out_queue (queue): Queue for parameters to next be run by experiment.
        costs_in_queue (queue): Queue for costs (and other details) that have been returned by experiment.
        **kwargs (Optional [dict]): Dictionary of options to be passed to Controller parent class and Nelder–Mead learner.
    '''
    def __init__(self, interface,
                **kwargs):
        super(NelderMeadController,self).__init__(interface, **kwargs)

        self.learner = mll.NelderMeadLearner(start_datetime = self.start_datetime,
                                             **self.remaining_kwargs)

        self._update_controller_with_learner_attributes()


class DifferentialEvolutionController(Controller):
    '''
    Controller for the differential evolution learner.
    Args:
        params_out_queue (queue): Queue for parameters to next be run by experiment.
        costs_in_queue (queue): Queue for costs (and other details) that have been returned by experiment.
        **kwargs (Optional [dict]): Dictionary of options to be passed to Controller parent class and differential evolution learner.
    '''
    def __init__(self, interface,
                **kwargs):
        super(DifferentialEvolutionController,self).__init__(interface, **kwargs)

        self.learner = mll.DifferentialEvolutionLearner(start_datetime = self.start_datetime,
                                                        **self.remaining_kwargs)

        self._update_controller_with_learner_attributes()


class MachineLearnerController(Controller):
    '''
    Abstract Controller class for the machine learning based solvers.
    Args:
        interface (Interface): The interface to the experiment under optimization.
        **kwargs (Optional [dict]): Dictionary of options to be passed to Controller parent class and initial training learner.
    Keyword Args:
        training_type (Optional [string]): The type for the initial training source can be 'random' for the random learner, 'nelder_mead' for the Nelder–Mead learner or 'differential_evolution' for the Differential Evolution learner. This learner is also called if the machine learning learner is too slow and a new point is needed. Default 'differential_evolution'.
        num_training_runs (Optional [int]): The number of training runs to before starting the learner. If None, will be ten or double the number of parameters, whatever is larger.
        no_delay (Optional [bool]): If True, there is never any delay between a returned cost and the next parameters to run for the experiment. In practice, this means if the machine learning learner has not prepared the next parameters in time the learner defined by the initial training source is used instead. If false, the controller will wait for the machine learning learner to predict the next parameters and there may be a delay between runs.
    '''

    def __init__(self, interface,
                 training_type='differential_evolution',
                 num_training_runs=None,
                 no_delay=True,
                 num_params=None,
                 min_boundary=None,
                 max_boundary=None,
                 trust_region=None,
                 learner_archive_filename = mll.default_learner_archive_filename,
                 learner_archive_file_type = mll.default_learner_archive_file_type,
                 param_names=None,
                 **kwargs):

        super(MachineLearnerController,self).__init__(interface, **kwargs)

        if num_training_runs is None:
            if num_params is None:
                self.num_training_runs = 10
            else:
                self.num_training_runs = max(10, 2*int(num_params))
        else:
            self.num_training_runs = int(num_training_runs)
        if self.num_training_runs<=0:
            self.log.error('Number of training runs must be larger than zero:'+repr(self.num_training_runs))
            raise ValueError
        self.no_delay = bool(no_delay)

        self.training_type = str(training_type)
        if self.training_type == 'random':
            self.learner = mll.RandomLearner(start_datetime=self.start_datetime,
                                             num_params=num_params,
                                             min_boundary=min_boundary,
                                             max_boundary=max_boundary,
                                             trust_region=trust_region,
                                             learner_archive_filename=None,
                                             learner_archive_file_type=learner_archive_file_type,
                                             param_names=param_names,
                                             **self.remaining_kwargs)

        elif self.training_type == 'nelder_mead':
            self.learner = mll.NelderMeadLearner(start_datetime=self.start_datetime,
                                                 num_params=num_params,
                                                 min_boundary=min_boundary,
                                                 max_boundary=max_boundary,
                                                 learner_archive_filename=None,
                                                 learner_archive_file_type=learner_archive_file_type,
                                                 param_names=param_names,
                                                 **self.remaining_kwargs)

        elif self.training_type == 'differential_evolution':
            self.learner = mll.DifferentialEvolutionLearner(start_datetime=self.start_datetime,
                                                            num_params=num_params,
                                                            min_boundary=min_boundary,
                                                            max_boundary=max_boundary,
                                                            trust_region=trust_region,
                                                            evolution_strategy='rand2',
                                                            learner_archive_filename=None,
                                                            learner_archive_file_type=learner_archive_file_type,
                                                            param_names=param_names,
                                                            **self.remaining_kwargs)

        else:
            self.log.error('Unknown training type provided to machine learning controller:' + repr(training_type))

        self.archive_dict.update({'training_type':self.training_type})
        self._update_controller_with_learner_attributes()


    def _update_controller_with_machine_learner_attributes(self):

        self.ml_learner_params_queue = self.ml_learner.params_out_queue
        self.ml_learner_costs_queue = self.ml_learner.costs_in_queue
        self.end_ml_learner = self.ml_learner.end_event
        self.new_params_event = self.ml_learner.new_params_event
        self.remaining_kwargs = self.ml_learner.remaining_kwargs
        self.generation_num = self.ml_learner.generation_num

    def _get_cost_and_in_dict(self):
        '''
        Get cost, uncertainty, parameters, bad, and extra data from experiment.
        
        This method calls `_get_cost_and_in_dict()` of the parent `Controller`
        class and additionally sends the results to machine learning learner.
        '''
        super(MachineLearnerController,self)._get_cost_and_in_dict()
        self.ml_learner_costs_queue.put((self.curr_params,
                                         self.curr_cost,
                                         self.curr_uncer,
                                         self.curr_bad))

    def _start_up(self):
        '''
        Runs pararent method and also starts training_learner.
        '''
        super(MachineLearnerController,self)._start_up()
        self.log.debug('ML learner started.')
        self.ml_learner.start()

    def _optimization_routine(self):
        '''
        Overrides _optimization_routine. Uses the parent routine for the training runs. Implements a customized _optimization_routine when running the machine learning learner.
        '''
        #Run the training runs using the standard optimization routine.
        self.log.debug('Starting training optimization.')
        self.log.info('Run: ' + str(self.num_in_costs) + ' (training)')
        next_params = self._first_params()
        self._put_params_and_out_dict(
            next_params,
            param_type=self.learner.OUT_TYPE,
        )
        self.save_archive()
        self._get_cost_and_in_dict()

        while (self.num_in_costs < self.num_training_runs) and self.check_end_conditions():
            self.log.info('Run: ' + str(self.num_in_costs) + ' (training)')
            next_params = self._next_params()
            self._put_params_and_out_dict(
                next_params,
                param_type=self.learner.OUT_TYPE,
            )
            self.save_archive()
            self._get_cost_and_in_dict()

        if self.check_end_conditions():
            #Start last training run
            self.log.info('Run: ' + str(self.num_in_costs) + ' (training)')
            next_params = self._next_params()
            self._put_params_and_out_dict(
                next_params,
                param_type=self.learner.OUT_TYPE,
            )

            self.log.debug('Starting ML optimization.')
            # This may be a race. Although the cost etc. is put in the queue to
            # the learner before the new_params_event is set, it's not clear if
            # python guarantees that the other process will see the item in the
            # queue before the event is set. To work around this,
            # learners.MachineLearner.get_params_and_costs() blocks with a
            # timeout while waiting for an item in the queue.
            self._get_cost_and_in_dict()
            self.save_archive()
            self.new_params_event.set()
            self.log.debug('End training runs.')

            ml_consec = 0
            ml_count = 0

        while self.check_end_conditions():
            run_num = self.num_in_costs
            if ml_consec==self.generation_num or (self.no_delay and self.ml_learner_params_queue.empty()):
                self.log.info('Run: ' + str(run_num) + ' (trainer)')
                next_params = self._next_params()
                self._put_params_and_out_dict(
                    next_params,
                    param_type=self.learner.OUT_TYPE,
                )
                ml_consec = 0
            else:
                self.log.info('Run: ' + str(run_num) + ' (machine learner)')
                next_params = self.ml_learner_params_queue.get()
                self._put_params_and_out_dict(
                    next_params,
                    param_type=self.ml_learner.OUT_TYPE,
                )
                ml_consec += 1
                ml_count += 1

            self.save_archive()
            self._get_cost_and_in_dict()

            if ml_count==self.generation_num:
                self.new_params_event.set()
                ml_count = 0

    def _shut_down(self):
        '''
        Shutdown and clean up resources of the machine learning controller.
        '''
        self.log.debug('ML learner end set.')
        self.end_ml_learner.set()
        self.ml_learner.join()

        self.log.debug('ML learner joined')
        last_dict = None
        while not self.ml_learner_params_queue.empty():
            last_dict = self.ml_learner_params_queue.get_nowait()
        if isinstance(last_dict, dict):
            try:
                self.predicted_best_parameters = last_dict['predicted_best_parameters']
                self.predicted_best_cost = last_dict['predicted_best_cost']
                self.predicted_best_uncertainty = last_dict['predicted_best_uncertainty']
            except KeyError:
                pass
            try:
                self.number_of_local_minima = last_dict['number_of_local_minima']
                self.local_minima_parameters = last_dict['local_minima_parameters']
                self.local_minima_costs = last_dict['local_minima_costs']
                self.local_minima_uncers = last_dict['local_minima_uncers']
            except KeyError:
                pass
            self.archive_dict.update(last_dict)
        else:
            if self.ml_learner.predict_global_minima_at_end or self.ml_learner.predict_local_minima_at_end:
                self.log.info('Machine learning learner did not provide best and/or all minima.')
        super(MachineLearnerController,self)._shut_down()

    def print_results(self):
        '''
        Adds some additional output to the results specific to controller.
        '''
        super(MachineLearnerController,self).print_results()
        try:
            self.log.info('\t* Predicted best parameters:' + str(self.predicted_best_parameters))
            try:
                errorstring = ' +/- ' + str(self.predicted_best_uncertainty)
            except AttributeError:
                errorstring = ''
            self.log.info('\t* Predicted best cost:' + str(self.predicted_best_cost) + errorstring)
        except AttributeError:
            pass
        try:
            self.log.info('\t* Predicted number of local minima:' + str(self.number_of_local_minima))
        except AttributeError:
            pass

class GaussianProcessController(MachineLearnerController):
    '''
    Controller for the Gaussian Process solver. Primarily suggests new points from the Gaussian Process learner. However, during the initial few runs it must rely on a different optimization algorithm to get some points to seed the learner.
    Args:
        interface (Interface): The interface to the experiment under optimization.
        **kwargs (Optional [dict]): Dictionary of options to be passed to MachineLearnerController parent class and Gaussian Process learner.
    Keyword Args:
    '''

    def __init__(
        self,
        interface,
        num_params=None,
        min_boundary=None,
        max_boundary=None,
        trust_region=None,
        learner_archive_filename = mll.default_learner_archive_filename,
        learner_archive_file_type = mll.default_learner_archive_file_type,
        param_names=None,
        **kwargs,
    ):

        super(GaussianProcessController,self).__init__(
            interface,
            num_params=num_params,
            min_boundary=min_boundary,
            max_boundary=max_boundary,
            trust_region=trust_region,
            learner_archive_filename=learner_archive_filename,
            learner_archive_file_type=learner_archive_file_type,
            param_names=param_names,
            **kwargs,
        )

        self.ml_learner = mll.GaussianProcessLearner(
            start_datetime=self.start_datetime,
            num_params=num_params,
            min_boundary=min_boundary,
            max_boundary=max_boundary,
            trust_region=trust_region,
            learner_archive_filename=learner_archive_filename,
            learner_archive_file_type=learner_archive_file_type,
            param_names=param_names,
            **self.remaining_kwargs,
        )

        self._update_controller_with_machine_learner_attributes()

class NeuralNetController(MachineLearnerController):
    '''
    Controller for the Neural Net solver. Primarily suggests new points from the Neural Net learner. However, during the initial few runs it must rely on a different optimization algorithm to get some points to seed the learner.
    Args:
        interface (Interface): The interface to the experiment under optimization.
        **kwargs (Optional [dict]): Dictionary of options to be passed to MachineLearnerController parent class and Neural Net learner.
    Keyword Args:
    '''

    def __init__(
        self,
        interface,
        num_params=None,
        min_boundary=None,
        max_boundary=None,
        trust_region=None,
        learner_archive_filename = mll.default_learner_archive_filename,
        learner_archive_file_type = mll.default_learner_archive_file_type,
        param_names=None,
        **kwargs,
    ):

        super(NeuralNetController,self).__init__(
            interface,
            num_params=num_params,
            min_boundary=min_boundary,
            max_boundary=max_boundary,
            trust_region=trust_region,
            learner_archive_filename=learner_archive_filename,
            learner_archive_file_type=learner_archive_file_type,
            param_names=param_names,
            **kwargs,
        )

        self.ml_learner = mll.NeuralNetLearner(
            start_datetime=self.start_datetime,
            num_params=num_params,
            min_boundary=min_boundary,
            max_boundary=max_boundary,
            trust_region=trust_region,
            learner_archive_filename=learner_archive_filename,
            learner_archive_file_type=learner_archive_file_type,
            param_names=param_names,
            **self.remaining_kwargs,
        )

        self._update_controller_with_machine_learner_attributes()
