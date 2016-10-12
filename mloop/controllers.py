'''
Module of all the controllers used in M-LOOP. The controllers, as the name suggests, control the interface to the experiment and all the learners employed to find optimal parameters.
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import datetime
import mloop.utilities as mlu
import mloop.learners as mll
import mloop.interfaces as mli
import logging
import os

controller_dict = {'random':1,'nelder_mead':2,'gaussian_process':3,'differential_evolution':4}
number_of_controllers = 4
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
        controller_type (Optional [str]): Defines the type of controller can be 'random', 'nelder' or 'gaussian_process'. Defaults to 'gaussian_process'.
        **controller_config_dict : Options to be passed to controller.
        
    Returns:
        Controller : threadible object which must be started with start() to get the controller running.
        
    Raises:
        ValueError : if controller_type is an unrecognized string
    '''
    log = logging.getLogger(__name__)
    
    controller_type = str(controller_type)
    if controller_type=='gaussian_process':
        controller = GaussianProcessController(interface, **controller_config_dict)
    elif controller_type=='differential_evolution':
        controller = DifferentialEvolutionController(interface, **controller_config_dict)
    elif controller_type=='nelder_mead':
        controller = NelderMeadController(interface, **controller_config_dict)
    elif controller_type=='random':
        controller = RandomController(interface, **controller_config_dict)
    else:
        log.error('Unknown controller type:' + repr(controller_type))
        raise ValueError
    
    return controller

class Controller():
    '''
    Abstract class for controllers. The controller controls the entire M-LOOP process. The controller for each algorithm all inherit from this class. The class stores a variety of data which all algorithms use and also all of the achiving and saving features.
    
    In order to implement your own controller class the minimum requirement is to add a learner to the learner variable. And implement the next_parameters method, where you provide the appropriate information to the learner and get the next parameters.
    
    See the RandomController for a simple implementation of a controller.
    
    Note the first three keywords are all possible halting conditions for the controller. If any of them are satisfied the controller will halt (meaning an and condition is used).
    
    Also creates an empty variable learner. The simplest way to make a working controller is to assign a learner of some kind to this variable, and add appropriate queues and events from it.
    
    Args:
        interface (interface): The interface process. Is run by learner.
        
    Keyword Args:
        max_num_runs (Optional [float]): The number of runs before the controller stops. If set to float('+inf') the controller will run forever. Default float('inf'), meaning the controller will run until another condition is met.
        target_cost (Optional [float]): The target cost for the run. If a run achieves a cost lower than the target, the controller is stopped. Default float('-inf'), meaning the controller will run until another condition is met.
        max_num_runs_without_better_params (Otional [float]): Puts a limit on the number of runs are allowed before a new better set of parameters is found. Default float('inf'), meaning the controller will run until another condition is met. 
        controller_archive_filename (Optional [string]): Filename for archive. Contains costs, parameter history and other details depending on the controller type. Default 'ControllerArchive.mat'
        controller_archive_file_type (Optional [string]): File type for archive. Can be either 'txt' a human readable text file, 'pkl' a python dill file, 'mat' a matlab file or None if there is no archive. Default 'mat'.
        archive_extra_dict (Optional [dict]): A dictionary with any extra variables that are to be saved to the archive. If None, nothing is added. Default None.
        start_datetime (Optional datetime): Datetime for when controller was started.
        
    Attributes:
        params_out_queue (queue): Queue for parameters to next be run by experiment.
        costs_in_queue (queue): Queue for costs (and other details) that have been returned by experiment.
        end_interface (event): Event used to trigger the end of the interface
        learner (None): The placeholder for the learner, creating this variable is the minimum requirement to make a working controller class.
        learner_params_queue (queue): The parameters queue for the learner
        learner_costs_queue (queue): The costs queue for the learner
        end_learner (event): Event used to trigger the end of the learner
        num_in_costs (int): Counter for the number of costs received.
        num_out_params (int): Counter for the number of parameters received. 
        out_params (list): List of all parameters sent out by controller.
        out_extras (list): Any extras associated with the output parameters.
        in_costs (list): List of costs received by controller.
        in_uncers (list): List of uncertainties receieved by controller.
        best_cost (float): The lowest, and best, cost received by the learner.
        best_uncer (float): The uncertainty associated with the best cost.
        best_params (array): The best parameters recieved by the learner.
        best_index (float): The run number that produced the best cost.  
    
    '''
    
    def __init__(self, interface,
                 max_num_runs = float('+inf'),
                 target_cost = float('-inf'),
                 max_num_runs_without_better_params = float('+inf'),
                 controller_archive_filename=default_controller_archive_filename,
                 controller_archive_file_type=default_controller_archive_file_type,
                 archive_extra_dict = None,
                 start_datetime = None,
                 **kwargs):
        
        #Make logger
        self.remaining_kwargs = mlu._config_logger(**kwargs)
        self.log = logging.getLogger(__name__)
        
        #Variable that are included in archive
        self.num_in_costs = 0
        self.num_out_params = 0
        self.num_last_best_cost = 0
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
        
        #Variables that used internally
        self.last_out_params = None
        self.curr_params = None
        self.curr_cost = None
        self.curr_uncer = None
        self.curr_bad = None
        self.curr_extras = None
        
        #Constants
        self.controller_wait = float(1)
        
        #Learner related variables
        self.learner_params_queue = None
        self.learner_costs_queue = None
        self.end_learner = None
        self.learner = None
        
        #Variables set by user
        
        #save interface and extract important variables
        if isinstance(interface, mli.Interface):
            self.interface = interface
        else:
            self.log.error('interface is not a Interface as defined in the MLOOP package.')
            raise TypeError
        
        self.params_out_queue = interface.params_out_queue
        self.costs_in_queue = interface.costs_in_queue
        self.end_interface = interface.end_event
        
        #Other options
        if start_datetime is None:
            self.start_datetime = datetime.datetime.now()
        else:
            self.start_datetime = datetime.datetime(start_datetime)
        self.max_num_runs = float(max_num_runs)
        if self.max_num_runs<=0:
            self.log.error('Number of runs must be greater than zero. max_num_runs:'+repr(self.max_num_run))
            raise ValueError
        self.target_cost = float(target_cost)
        self.max_num_runs_without_better_params = float(max_num_runs_without_better_params)
        if self.max_num_runs_without_better_params<=0:
            self.log.error('Max number of repeats must be greater than zero. max_num_runs:'+repr(max_num_runs_without_better_params))
            raise ValueError
        
        if mlu.check_file_type_supported(controller_archive_file_type):
            self.controller_archive_file_type = controller_archive_file_type
        else:
            self.log.error('File in type is not supported:' + repr(controller_archive_file_type))
            raise ValueError
        if controller_archive_filename is None:
            self.controller_archive_filename = None
        else:
            if not os.path.exists(mlu.archive_foldername):
                os.makedirs(mlu.archive_foldername)
            self.controller_archive_filename =str(controller_archive_filename)
            self.total_archive_filename = mlu.archive_foldername + self.controller_archive_filename + '_' + mlu.datetime_to_string(self.start_datetime) + '.' + self.controller_archive_file_type
        
        self.archive_dict = {'archive_type':'controller',
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
        Check whether either of the three end contions have been met: number_of_runs, target_cost or max_num_runs_without_better_params.
        
        Returns:
            bool : True, if the controlled should continue, False if the controller should end. 
        '''
        return (self.num_in_costs < self.max_num_runs) and (self.best_cost > self.target_cost) and (self.num_last_best_cost < self.max_num_runs_without_better_params)
    
    def _update_controller_with_learner_attributes(self):
        '''
        Update the controller with properties from the learner.
        '''
        self.learner_params_queue = self.learner.params_out_queue
        self.learner_costs_queue = self.learner.costs_in_queue
        self.end_learner = self.learner.end_event
        self.remaining_kwargs = self.learner.remaining_kwargs
        
        self.archive_dict.update({'num_params':self.learner.num_params,
                                  'min_boundary':self.learner.min_boundary,
                                  'max_boundary':self.learner.max_boundary})
        
    
    def _put_params_and_out_dict(self, params,  param_type=None, **kwargs):
        '''
        Send parameters to queue and whatever additional keywords. Saves sent variables in appropriate storage arrays. 
        
        Args:
            params (array) : array of values to be sent to file
        
        Keyword Args:
            **kwargs: any additional data to be attached to file sent to experiment
        '''
        out_dict = {'params':params}
        out_dict.update(kwargs)
        self.params_out_queue.put(out_dict)
        self.num_out_params += 1
        self.last_out_params = params
        self.out_params.append(params)
        self.out_extras.append(kwargs)
        if param_type is not None:
            self.out_type.append(param_type)
        self.log.info('params ' + str(params))
        #self.log.debug('Put params num:' + repr(self.num_out_params ))
        
    def _get_cost_and_in_dict(self):
        '''
        Get cost, uncertainty, parameters, bad and extra data from experiment. Stores in a list of history and also puts variables in their appropriate 'current' variables
        
        Note returns nothing, stores everything in the internal storage arrays and the curr_variables
        '''
        while True:
            try:
                in_dict = self.costs_in_queue.get(True, self.controller_wait)
            except mlu.empty_exception:
                continue
            else:
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
        if self.curr_bad and 'cost' in dict:
            self.log.warning('The cost provided with the bad run will be saved, but not used by the learners.')
        
        self.in_costs.append(self.curr_cost)
        self.in_uncers.append(self.curr_uncer)
        self.in_bads.append(self.curr_bad)
        self.in_extras.append(self.curr_extras)
        self.curr_params = self.last_out_params
        if self.curr_cost < self.best_cost:
            self.best_cost = self.curr_cost
            self.best_uncer = self.curr_uncer
            self.best_index =  self.num_in_costs
            self.best_params = self.curr_params
            self.num_last_best_cost = 0
        if self.curr_bad:
            self.log.info('bad run')
        else:
            self.log.info('cost ' + str(self.curr_cost) + ' +/- ' + str(self.curr_uncer))
        #self.log.debug('Got cost num:' + repr(self.num_in_costs))
    
    def save_archive(self):
        '''
        Save the archive associated with the controller class. Only occurs if the filename for the archive is not None. Saves with the format previously set.
        '''
        if self.controller_archive_filename is not None:
            self.archive_dict.update({'num_in_costs':self.num_in_costs,
                                      'num_out_params':self.num_out_params,
                                      'best_cost':self.best_cost,
                                      'best_uncer':self.best_uncer,
                                      'best_params':self.best_params,
                                      'best_index':self.best_index})
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
        except (KeyboardInterrupt,SystemExit):
            log.warning('!!! Do not give the interrupt signal again !!! \n M-LOOP stopped with keyboard interupt or system exit. Please wait at least 1 minute for the threads to safely shut down. \n ')
            log.warning('Closing down controller.')
        except Exception:
            self.log.warning('Controller ended due to exception of some kind. Starting shut down...')
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
        #After 3 or 4 executions of mloop in same python environment, sometimes excution can be trapped here
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
        self.log.info('Optimization ended because:-')
        if self.num_in_costs >= self.max_num_runs:
            self.log.info('Maximum number of runs reached.')
        if self.best_cost <= self.target_cost:
            self.log.info('Target cost reached.')
        if self.num_last_best_cost >= self.max_num_runs_without_better_params:
            self.log.info('Maximum number of runs without better params reached.')
        self.log.info('Results:-')
        self.log.info('Best parameters found:' + str(self.best_params))
        self.log.info('Best cost returned:' + str(self.best_cost) + ' +/- ' + str(self.best_uncer))
        self.log.info('Best run number:' + str(self.best_index))

    def _optimization_routine(self):
        '''
        Runs controller main loop. Gives parameters to experiment and saves costs returned. 
        '''
        self.log.debug('Start controller loop.')
        try:
            self.log.info('Run:' + str(self.num_in_costs +1))
            next_params = self._first_params()
            self._put_params_and_out_dict(next_params)
            self.save_archive()
            self._get_cost_and_in_dict()
            while self.check_end_conditions():
                self.log.info('Run:' + str(self.num_in_costs +1))
                next_params = self._next_params()
                self._put_params_and_out_dict(next_params)
                self.save_archive()
                self._get_cost_and_in_dict()
            self.log.debug('End controller loop.')
        except ControllerInterrupt:
            self.log.warning('Controller ended by interruption.')
    
    def _first_params(self):
        '''
        Checks queue to get first  parameters. 
        
        Returns:
            Parameters for first experiment
        '''
        return self.learner_params_queue.get()
    
    def _next_params(self):
        '''
        Abstract method.
        
        When implemented should send appropriate information to learner and get next parameters.
        
        Returns:
            Parameters for next experiment.
        '''
        pass
    
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
                                         learner_archive_filename=None,
                                         **self.remaining_kwargs)
        
        self._update_controller_with_learner_attributes()
        self.out_type.append('random')
        
        self.log.debug('Random controller init completed.')    
        
    def _next_params(self):
        '''
        Sends cost uncer and bad tuple to learner then gets next parameters.
        
        Returns:
            Parameters for next experiment.
        '''
        self.learner_costs_queue.put(self.best_params)
        return self.learner_params_queue.get()      
     

class NelderMeadController(Controller):
    '''
    Controller for the Nelder-Mead solver. Suggests new parameters based on the Nelder-Mead algorithm. Can take no boundaries or hard boundaries. More details for the Nelder-Mead options are in the learners section.
    
    Args:
        params_out_queue (queue): Queue for parameters to next be run by experiment.
        costs_in_queue (queue): Queue for costs (and other details) that have been returned by experiment.
        **kwargs (Optional [dict]): Dictionary of options to be passed to Controller parent class and Nelder-Mead learner.
    '''
    def __init__(self, interface,
                **kwargs):
        super(NelderMeadController,self).__init__(interface, **kwargs)    
        
        self.learner = mll.NelderMeadLearner(start_datetime = self.start_datetime,
                                             **self.remaining_kwargs)
        
        self._update_controller_with_learner_attributes()
        self.out_type.append('nelder_mead')
    
    def _next_params(self):
        '''
        Gets next parameters from Nelder-Mead learner.
        '''
        if self.curr_bad:
            cost = float('inf')
        else:
            cost = self.curr_cost       
        self.learner_costs_queue.put(cost)
        return self.learner_params_queue.get()

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
        self.out_type.append('differential_evolution')
    
    def _next_params(self):
        '''
        Gets next parameters from differential evolution learner.
        '''
        if self.curr_bad:
            cost = float('inf')
        else:
            cost = self.curr_cost       
        self.learner_costs_queue.put(cost)
        return self.learner_params_queue.get()




class GaussianProcessController(Controller):
    '''
    Controller for the Gaussian Process solver. Primarily suggests new points from the Gaussian Process learner. However, during the initial few runs it must rely on a different optimization algorithm to get some points to seed the learner. 
    
    Args:
        interface (Interface): The interface to the experiment under optimization.
        **kwargs (Optional [dict]): Dictionary of options to be passed to Controller parent class, initial training learner and Gaussian Process learner.
    
    Keyword Args:
        initial_training_source (Optional [string]): The type for the initial training source can be 'random' for the random learner or 'nelder_mead' for the Nelder-Mead learner. This leaner is also called if the Gaussian process learner is too slow and a new point is needed. Default 'random'.
        num_training_runs (Optional [int]): The number of training runs to before starting the learner. If None, will by ten or double the number of parameters, whatever is larger. 
        no_delay (Optional [bool]): If True, there is never any delay between a returned cost and the next parameters to run for the experiment. In practice, this means if the gaussian process has not prepared the next parameters in time the learner defined by the initial training source is used instead. If false, the controller will wait for the gaussian process to predict the next parameters and there may be a delay between runs. 
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
                 **kwargs):
        super(GaussianProcessController,self).__init__(interface, **kwargs)   
        
        self.last_training_cost = None
        self.last_training_bad = None
        self.last_training_run_flag = False
        
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
                                             **self.remaining_kwargs)

        elif self.training_type == 'nelder_mead':
            self.learner = mll.NelderMeadLearner(start_datetime=self.start_datetime,
                                                 num_params=num_params,
                                                 min_boundary=min_boundary,
                                                 max_boundary=max_boundary,
                                                 learner_archive_filename=None,
                                                 learner_archive_file_type=learner_archive_file_type,
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
                                                            **self.remaining_kwargs)    
        
        else:
            self.log.error('Unknown training type provided to Gaussian process controller:' + repr(training_type))
        
        self.archive_dict.update({'training_type':self.training_type})
        self._update_controller_with_learner_attributes()
        
        self.gp_learner = mll.GaussianProcessLearner(start_datetime=self.start_datetime,
                                                  num_params=num_params,
                                                  min_boundary=min_boundary,
                                                  max_boundary=max_boundary,
                                                  trust_region=trust_region,
                                                  learner_archive_filename=learner_archive_filename,
                                                  learner_archive_file_type=learner_archive_file_type,
                                                  **self.remaining_kwargs)
        
        self.gp_learner_params_queue = self.gp_learner.params_out_queue
        self.gp_learner_costs_queue = self.gp_learner.costs_in_queue
        self.end_gp_learner = self.gp_learner.end_event
        self.new_params_event = self.gp_learner.new_params_event
        self.remaining_kwargs = self.gp_learner.remaining_kwargs
        self.generation_num = self.gp_learner.generation_num
        
    def _put_params_and_out_dict(self, params):
        '''
        Override _put_params_and_out_dict function, used when the training learner creates parameters. Makes the defualt param_type the training type and sets last_training_run_flag.
        '''
        super(GaussianProcessController,self)._put_params_and_out_dict(params, param_type=self.training_type)
        self.last_training_run_flag = True 
    
    def _get_cost_and_in_dict(self):
        '''
        Call _get_cost_and_in_dict() of parent Controller class. But also sends cost to Gaussian process learner and saves the cost if the parameters came from a trainer. 
        
        '''
        super(GaussianProcessController,self)._get_cost_and_in_dict()
        if self.last_training_run_flag:
            self.last_training_cost = self.curr_cost
            self.last_training_bad = self.curr_bad
            self.last_training_run_flag = False
        self.gp_learner_costs_queue.put((self.curr_params,
                                         self.curr_cost,
                                         self.curr_uncer,
                                         self.curr_bad))
    
    def _next_params(self):
        '''
        Gets next parameters from training learner.
        '''
        if self.training_type == 'differential_evolution' or self.training_type == 'nelder_mead':
            #Copied from NelderMeadController
            if self.last_training_bad:
                cost = float('inf')
            else:
                cost = self.last_training_cost       
            self.learner_costs_queue.put(cost)
            temp = self.learner_params_queue.get()
            
        elif self.training_type == 'random':
            #Copied from RandomController
            self.learner_costs_queue.put(self.best_params)
            temp = self.learner_params_queue.get()  
            
        else:
            self.log.error('Unknown training type called. THIS SHOULD NOT HAPPEN')
        return temp
    
    def _start_up(self):
        '''
        Runs pararent method and also starts training_learner.
        '''
        super(GaussianProcessController,self)._start_up()
        self.log.debug('GP learner started.')
        self.gp_learner.start()

    def _optimization_routine(self):
        '''
        Overrides _optimization_routine. Uses the parent routine for the training runs. Implements a customized _optimization_rountine when running the Gaussian Process learner. 
        '''
        #Run the training runs using the standard optimization routine. Adjust the number of max_runs
        save_max_num_runs = self.max_num_runs
        self.max_num_runs = self.num_training_runs - 1
        self.log.debug('Starting training optimization.')
        super(GaussianProcessController,self)._optimization_routine()
        
        #Start last training run
        self.log.info('Run:' + str(self.num_in_costs +1))
        next_params = self._next_params()
        self._put_params_and_out_dict(next_params)
        
        #Begin GP optimization routine
        self.max_num_runs = save_max_num_runs
        
        self.log.debug('Starting GP optimization.')
        self.new_params_event.set()
        self.save_archive()
        self._get_cost_and_in_dict()
        
        gp_consec = 0
        gp_count = 0
        while self.check_end_conditions():
            self.log.info('Run:' + str(self.num_in_costs +1))
            if gp_consec==self.generation_num or (self.no_delay and self.gp_learner_params_queue.empty()):
                next_params = self._next_params()
                self._put_params_and_out_dict(next_params)
                gp_consec = 0
            else:
                next_params = self.gp_learner_params_queue.get()
                super(GaussianProcessController,self)._put_params_and_out_dict(next_params, param_type='gaussian_process')
                gp_consec += 1
                gp_count += 1
            
            if gp_count%self.generation_num == 2:
                self.new_params_event.set()
            
            self.save_archive()
            self._get_cost_and_in_dict()
        

    def _shut_down(self):
        '''
        Shutdown and clean up resources of the Gaussian process controller.
        '''
        self.log.debug('GP learner end set.')
        self.end_gp_learner.set()
        self.gp_learner.join()
        #self.gp_learner.join(self.gp_learner.learner_wait*3)
        '''
        if self.gp_learner.is_alive():
            self.log.warning('GP Learner did not join in time had to terminate.')
            self.gp_learner.terminate()
        '''
        self.log.debug('GP learner joined')   
        last_dict = None
        while not self.gp_learner_params_queue.empty():
            last_dict = self.gp_learner_params_queue.get_nowait()
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
            if self.gp_learner.predict_global_minima_at_end or self.gp_learner.predict_local_minima_at_end:
                self.log.warning('GP Learner may not have closed properly unable to get best and/or all minima.')
        super(GaussianProcessController,self)._shut_down()
        
    def print_results(self):
        '''
        Adds some additional output to the results specific to controller. 
        '''
        super(GaussianProcessController,self).print_results()
        try:
            self.log.info('Predicted best parameters:' + str(self.predicted_best_parameters))
            self.log.info('Predicted best cost:' + str(self.predicted_best_cost) + ' +/- ' + str(self.predicted_best_uncertainty))
            
        except AttributeError:
            pass
        try:
            self.log.info('Predicted number of local minima:' + str(self.number_of_local_minima))
        except AttributeError:
            pass


    

        