'''
Module of learners used to determine what parameters to try next given previous cost evaluations. 

Each learner is created and controlled by a controller.
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import threading
import numpy as np
import numpy.random as nr
import scipy.optimize as so
import logging
import datetime
import os
import mloop.utilities as mlu
import sklearn.gaussian_process as skg
import sklearn.gaussian_process.kernels as skk
import sklearn.preprocessing as skp
import multiprocessing as mp

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
        min_boundary (Optional [array]): Array with minimimum values allowed for each parameter. Note if certain values have no minimum value you can set them to -inf for example [-1, 2, float('-inf')] is a valid min_boundary. If None sets all the boundaries to '-1'. Default None.
        max_boundary (Optional [array]): Array with maximum values allowed for each parameter. Note if certain values have no maximum value you can set them to +inf for example [0, float('inf'),3,-12] is a valid max_boundary. If None sets all the boundaries to '1'. Default None.
        learner_archive_filename (Optional [string]): Name for python archive of the learners current state. If None, no archive is saved. Default None. But this is typically overloaded by the child class.
        learner_archive_file_type (Optional [string]):  File type for archive. Can be either 'txt' a human readable text file, 'pkl' a python dill file, 'mat' a matlab file or None if there is no archive. Default 'mat'.
        log_level (Optional [int]): Level for the learners logger. If None, set to warning. Default None.
        start_datetime (Optional [datetime]): Start date time, if None, is automatically generated.
        
    Attributes:
        params_out_queue (queue): Queue for parameters created by learner.
        costs_in_queue (queue): Queue for costs to be used by learner.
        end_event (event): Event to trigger end of learner.
    '''
    
    def __init__(self, 
                 num_params=None,
                 min_boundary=None, 
                 max_boundary=None, 
                 learner_archive_filename=default_learner_archive_filename,
                 learner_archive_file_type=default_learner_archive_file_type,
                 start_datetime=None,
                 **kwargs):

        super(Learner,self).__init__()
        
        global learner_thread_count
        learner_thread_count += 1        
        self.log = logging.getLogger(__name__ + '.' + str(learner_thread_count))
        
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
            if not os.path.exists(mlu.archive_foldername):
                os.makedirs(mlu.archive_foldername)
            self.learner_archive_filename =str(learner_archive_filename)
            self.total_archive_filename = mlu.archive_foldername + self.learner_archive_filename + '_' + mlu.datetime_to_string(self.start_datetime) + '.' + self.learner_archive_file_type
        
        self.archive_dict = {'archive_type':'learner',
                             'num_params':self.num_params,
                             'min_boundary':self.min_boundary,
                             'max_boundary':self.max_boundary,
                             'start_datetime':mlu.datetime_to_string(self.start_datetime)}
        
        self.log.debug('Learner init completed.')   
        
    def check_num_params(self,param):
        '''
        Check the number of parameters is right.
        '''
        return param.shape == (self.num_params,)
    
    def check_in_boundary(self,param):
        '''
        Check give parameters are within stored boundaries
        
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
        Check given distances are less than the boundaries
        
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
        Send parameters to queue and whatever additional keywords. Saves sent variables in appropriate storage arrays. 
        
        Args:
            params (array) : array of values to be sent to file
        
        Returns:
            cost from the cost queue
        '''
        #self.log.debug('Learner params='+repr(params))
        if not self.check_num_params(params):
            self.log.error('Incorrect number of parameters sent to queue.Params' + repr(params))
            raise ValueError
        if not self.check_in_boundary(params):
            self.log.warning('Parameters sent to queue are not within boundaries. Params:' + repr(params))
        #self.log.debug('Learner puts params.')
        self.params_out_queue.put(params)
        #self.log.debug('Learner waiting for costs.')
        self.save_archive()
        while not self.end_event.is_set():
            try:
                cost = self.costs_in_queue.get(True, self.learner_wait)
            except mlu.empty_exception:
                continue
            else:
                break
        else:
            self.log.debug('Learner end signal received. Ending')
            raise LearnerInterrupt
        #self.log.debug('Learner cost='+repr(cost))
        return cost
    
    def save_archive(self):
        '''
        Save the archive associated with the learner class. Only occurs if the filename for the archive is not None. Saves with the format previously set.
        '''
        self.update_archive()
        if self.learner_archive_filename is not None:
            mlu.save_dict_to_file(self.archive_dict, self.total_archive_filename, self.learner_archive_file_type)
    
    def update_archive(self):
        '''
        Abstract method for update to the archive. To be implemented by child class.
        '''
        pass
    
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
            self.log.error('Minimum and/or maximum boundaries are NaN or inf. Must both be finite for random learner. Min boundary:' + repr(self.min_boundary) +'. Max bounday:' + repr(self.max_boundary))
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
                centre_params = self.put_params_and_get_cost(next_params)
            except LearnerInterrupt:
                break
            else:
                if self.has_trust_region:
                    temp_min = np.maximum(self.min_boundary,centre_params - self.trust_region)
                    temp_max = np.minimum(self.max_boundary,centre_params + self.trust_region)
                    next_params = temp_min + nr.rand(self.num_params) * (temp_max - temp_min)
                else:
                    next_params =  self.min_boundary + nr.rand(self.num_params) * self.diff_boundary
                
        self._shut_down()
        self.log.debug('Ended Random Learner')
    
    
class NelderMeadLearner(Learner, threading.Thread):
    '''
    Nelder-Mead learner. Executes the Nelder-Mead learner algorithm and stores the needed simplex to estimate the next points.
    
    Args:
        params_out_queue (queue): Queue for parameters from controller.
        costs_in_queue (queue): Queue for costs for nelder learner. The queue should be populated with cost (float) corresponding to the last parameter sent from the Nelder-Mead Learner. Can be a float('inf') if it was a bad run.
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
        
        self.log.debug('Nelder-Mead learner init completed.')
        
    def run(self):
        '''
        Runs Nelder-Mead algorithm to produce new parameters given costs, until end signal is given. 
        '''
        
        self.log.info('Starting Nelder Mead Learner')
        
        N = int(self.num_params)
        
        one2np1 = list(range(1, N + 1))
    
        self.simplex_params[0] = self.init_simplex_corner
        
        try:
            self.simplex_costs[0] = self.put_params_and_get_cost(self.init_simplex_corner)
        except ValueError:
            self.log.error('Outside of boundary on initial condition. THIS SHOULD NOT HAPPEN')
            raise
        except LearnerInterrupt:
            self.log.info('Ended Nelder-Mead before end of simplex')
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
                self.log.info('Ended Nelder-Mead before end of simplex')
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
        self.log.info('Ended Nelder-Mead')

def update_archive(self):
        '''
        Update the archive.
        '''
        self.archive_dict.update({'archive_type':'nelder_mead_learner',
                                  'simplex_parameters':self.simplex_params,
                                  'simplex_costs':self.simplex_costs})


class GaussianProcessLearner(Learner, mp.Process):
    '''
    Gaussian process learner. Generats new parameters based on a gaussian process fitted to all previous data. 
    
    Args:
        params_out_queue (queue): Queue for parameters sent to controller.
        costs_in_queue (queue): Queue for costs for gaussian process. This must be tuple
        end_event (event): Event to trigger end of learner.
        
    Keyword Args:
        length_scale (Optional [array]): The initial guess for length scale(s) of the gaussian process. The array can either of size one or the number of parameters or None. If it is size one, it is assumed all the correlation lengths are the same. If it is the number of the parameters then all the parameters have their own independent length scale. If it is None, it is assumed all the length scales should be independent and they are all given an initial value of 1. Default None.
        cost_has_noise (Optional [bool]): If true the learner assumes there is common additive white noise that corrupts the costs provided. This noise is assumed to be on top of the uncertainty in the costs (if it is provided). If false, it is assumed that there is no noise in the cost (or if uncertainties are provided no extra noise beyond the uncertainty). Default True. 
        noise_level (Optional [float]): The initial guess for the noise level in the costs, is only used if cost_has_noise is true. Default 1.0.
        update_hyperparameters (Optional [bool]): Whether the length scales and noise estimate should be updated when new data is provided. Is set to true by default.
        trust_region (Optional [float or array]): The trust region defines the maximum distance the learner will travel from the current best set of parameters. If None, the learner will search everywhere. If a float, this number must be between 0 and 1 and defines maximum distance the learner will venture as a percentage of the boundaries. If it is an array, it must have the same size as the number of parameters and the numbers define the maximum absolute distance that can be moved along each direction. 
        default_bad_cost (Optional [float]): If a run is reported as bad and default_bad_cost is provided, the cost for the bad run is set to this default value. If default_bad_cost is None, then the worst cost received is set to all the bad runs. Default None.
        default_bad_uncertainty (Optional [float]): If a run is reported as bad and default_bad_uncertainty is provided, the uncertainty for the bad run is set to this default value. If default_bad_uncertainty is None, then the uncertainty is set to a tenth of the best to worst cost range. Default None.
        minimum_uncertainty (Optional [float]): The minimum uncertainty associated with provided costs. Must be above zero to avoid fitting errors. Default 1e-8.
        predict_global_minima_at_end (Optional [bool]): If True finds the global minima when the learner is ended. Does not if False. Default True.
        predict_local_minima_at_end (Optional [bool]): If True finds the all minima when the learner is ended. Does not if False. Default False.
        
    Attributes:
        all_params (array): Array containing all parameters sent to learner.
        all_costs (array): Array containing all costs sent to learner.
        all_uncers (array): Array containing all uncertainties sent to learner.
        scaled_costs (array): Array contaning all the costs scaled to have zero mean and a standard deviation of 1. Needed for training the gaussian process. 
        bad_run_indexs (list): list of indexes to all runs that were marked as bad.
        best_cost (float): Minimum received cost, updated during execution.
        best_params (array): Parameters of best run. (reference to element in params array).
        best_index (int): index of the best cost and params. 
        worst_cost (float): Maximum received cost, updated during execution.
        worst_index (int): index to run with worst cost.
        cost_range (float): Difference between worst_cost and best_cost
        generation_num (int): Number of sets of parameters to generate each generation. Set to 5.
        length_scale_history (list): List of length scales found after each fit.
        noise_level_history (list): List of noise levels found after each fit. 
        fit_count (int): Counter for the number of times the gaussian process has been fit.
        cost_count (int): Counter for the number of costs, parameters and uncertainties added to learner.
        params_count (int): Counter for the number of parameters asked to be evaluated by the learner.  
        gaussian_process (GaussianProcessRegressor): Gaussian process that is fitted to data and used to make predictions
        cost_scaler (StandardScaler): Scaler used to normalize the provided costs. 
        has_trust_region (bool): Whether the learner has a trust region. 
    ''' 
    
    def __init__(self, 
                 length_scale = None,
                 update_hyperparameters = True,
                 cost_has_noise=True,
                 noise_level=1.0,
                 trust_region=None,
                 default_bad_cost = None,
                 default_bad_uncertainty = None,
                 minimum_uncertainty = 1e-8,
                 gp_training_filename =None,
                 gp_training_file_type ='txt',
                 predict_global_minima_at_end = True,
                 predict_local_minima_at_end = False,
                 **kwargs):
        
        if gp_training_filename is not None:
            
            gp_training_filename = str(gp_training_filename)
            gp_training_file_type = str(gp_training_file_type)
            if not mlu.check_file_type_supported(gp_training_file_type):
                self.log.error('GP training file type not supported' + repr(gp_training_file_type))
            
            self.training_dict = mlu.get_dict_from_file(gp_training_filename, gp_training_file_type)
            
            #Basic optimization settings
            num_params = int(self.training_dict['num_params'])
            min_boundary = np.squeeze(np.array(self.training_dict['min_boundary'], dtype=float))
            max_boundary = np.squeeze(np.array(self.training_dict['max_boundary'], dtype=float))
            
            #Configuration of the learner
            self.cost_has_noise = bool(self.training_dict['cost_has_noise'])
            self.length_scale = np.squeeze(np.array(self.training_dict['length_scale']))
            self.length_scale_history = list(self.training_dict['length_scale_history'])
            self.noise_level = float(self.training_dict['noise_level'])
            if isinstance(self.training_dict['noise_level_history'], np.ndarray):
                self.noise_level_history = list(np.squeeze(self.training_dict['noise_level_history']))
            else:
                self.noise_level_history = list( self.training_dict['noise_level_history'])
            
            #Counters
            self.costs_count = int(self.training_dict['costs_count'])
            self.fit_count = int(self.training_dict['fit_count'])
            self.params_count = int(self.training_dict['params_count'])
            
            #Data from previous experiment
            self.all_params = np.array(self.training_dict['all_params'], dtype=float)
            self.all_costs = np.squeeze(np.array(self.training_dict['all_costs'], dtype=float))
            self.all_uncers = np.squeeze(np.array(self.training_dict['all_uncers'], dtype=float))
            
            if isinstance(self.training_dict['bad_run_indexs'], np.ndarray):
                self.bad_run_indexs = list(np.squeeze(self.training_dict['bad_run_indexs']))
            else:
                self.bad_run_indexs = list(self.training_dict['bad_run_indexs'])
            
            
            #Derived properties
            self.best_cost = float(self.training_dict['best_cost'])
            self.best_params = np.squeeze(np.array(self.training_dict['best_params'], dtype=float))
            self.best_index = int(self.training_dict['best_index'])
            self.worst_cost = float(self.training_dict['worst_cost'])
            self.worst_index = int(self.training_dict['worst_index'])
            self.cost_range = float(self.training_dict['cost_range'])
            try:
                self.predicted_best_parameters = np.squeeze(np.array(self.training_dict['predicted_best_parameters']))
                self.predicted_best_cost = float(self.training_dict['predicted_best_cost'])
                self.predicted_best_uncertainty = float(self.training_dict['predicted_best_uncertainty'])
                self.has_global_minima = True
            except KeyError:
                self.has_global_minima = False
            try:
                self.local_minima_parameters = list(self.training_dict['local_minima_parameters'])
                
                if isinstance(self.training_dict['local_minima_costs'], np.ndarray):
                    self.local_minima_costs = list(np.squeeze(self.training_dict['local_minima_costs']))
                else:
                    self.local_minima_costs = list(self.training_dict['local_minima_costs'])
                if isinstance(self.training_dict['local_minima_uncers'], np.ndarray):
                    self.local_minima_uncers = list(np.squeeze(self.training_dict['local_minima_uncers']))
                else:
                    self.local_minima_uncers = list(self.training_dict['local_minima_uncers'])
                
                self.has_local_minima = True
            except KeyError:
                self.has_local_minima = False
            
        
            super(GaussianProcessLearner,self).__init__(num_params=num_params,
                             min_boundary=min_boundary, 
                             max_boundary=max_boundary, 
                             **kwargs)
            
        else:
            
            super(GaussianProcessLearner,self).__init__(**kwargs)
            
            #Storage variables, archived
            self.all_params = np.array([], dtype=float)
            self.all_costs = np.array([], dtype=float)
            self.all_uncers = np.array([], dtype=float)
            self.bad_run_indexs = []
            self.best_cost = float('inf')
            self.best_params = float('nan')
            self.best_index = 0
            self.worst_cost = float('-inf')
            self.worst_index = 0
            self.cost_range = float('inf')
            self.length_scale_history = []
            self.noise_level_history = []
            
            self.costs_count = 0
            self.fit_count = 0
            self.params_count = 0
            
            self.has_local_minima = False
            self.has_global_minima = False
            
            #Optional user set variables
            if length_scale is None:
                self.length_scale = np.ones((self.num_params,))
            else:
                self.length_scale = np.array(length_scale, dtype=float)
            self.noise_level = float(noise_level)
            self.cost_has_noise = bool(cost_has_noise)
            
            
        #Multiprocessor controls
        self.new_params_event = mp.Event()
        
        #Storage variables and counters
        self.search_params = []
        self.scaled_costs = None
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
        self.search_precision = 1.0e-6
        self.parameter_searches = max(10,self.num_params)
        self.hyperparameter_searches = max(10,self.num_params)
        self.bad_uncer_frac = 0.1 #Fraction of cost range to set a bad run uncertainty 
        
        #Optional user set variables
        self.update_hyperparameters = bool(update_hyperparameters)
        self.predict_global_minima_at_end = bool(predict_global_minima_at_end)
        self.predict_local_minima_at_end = bool(predict_local_minima_at_end)
        if default_bad_cost is not None:
            self.default_bad_cost = float(default_bad_cost)
        else:
            self.default_bad_cost = None
        if default_bad_uncertainty is not None:
            self.default_bad_uncertainty = float(default_bad_uncertainty)
        else:
            self.default_bad_uncertainty = None
        self.minimum_uncertainty = float(minimum_uncertainty)
        self._set_trust_region(trust_region)
        
        #Checks of variables
        if self.length_scale.size == 1:
            self.length_scale = float(self.length_scale)
        elif not self.check_num_params(self.length_scale):
            self.log.error('Correlation lengths not the right size and shape, must be one or the number of parameters:' + repr(self.length_scale))
            raise ValueError
        if not np.all(self.length_scale >0):
            self.log.error('Correlation lengths must all be positive numbers:' + repr(self.length_scale))
            raise ValueError
        if self.noise_level < 0:
            self.log.error('noise_level must be greater or equal to zero:' +repr(self.noise_level))
            raise ValueError
        if self.default_bad_uncertainty is not None:
            if self.default_bad_uncertainty < 0:
                self.log.error('Default bad uncertainty must be positive.')
                raise ValueError
        if (self.default_bad_cost is None) and (self.default_bad_cost is None):
            self.bad_defaults_set = False
        elif (self.default_bad_cost is not None) and (self.default_bad_cost is not None):
            self.bad_defaults_set = True
        else:
            self.log.error('Both the default cost and uncertainty must be set for a bad run or they must both be set to None.')
            raise ValueError
        if self.minimum_uncertainty <= 0:
            self.log.error('Minimum uncertainty must be larger than zero for the learner.')
            raise ValueError
                
        self.create_gaussian_process()
        
        #Search bounds
        self.search_min = self.min_boundary
        self.search_max = self.max_boundary
        self.search_diff = self.search_max - self.search_min
        self.search_region = list(zip(self.search_min, self.search_max))
        
        self.cost_scaler = skp.StandardScaler()
        
        self.archive_dict.update({'archive_type':'gaussian_process_learner',
                                  'cost_has_noise':self.cost_has_noise,
                                  'length_scale_history':self.length_scale_history,
                                  'noise_level_history':self.noise_level_history,
                                  'bad_run_indexs':self.bad_run_indexs,
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
                                  'predict_global_minima_at_end':self.predict_global_minima_at_end,
                                  'predict_local_minima_at_end':self.predict_local_minima_at_end})
        
        #Remove logger so gaussian process can be safely picked for multiprocessing on Windows
        self.log = None
            
        
    def create_gaussian_process(self):
        '''
        Create the initial Gaussian process.
        '''
        if self.cost_has_noise:
            gp_kernel = skk.RBF(length_scale=self.length_scale) + skk.WhiteKernel(noise_level=self.noise_level)
        else:
            gp_kernel = skk.RBF(length_scale=self.length_scale)
        if self.update_hyperparameters:
            self.gaussian_process = skg.GaussianProcessRegressor(kernel=gp_kernel,n_restarts_optimizer=self.hyperparameter_searches)
        else:
            self.gaussian_process = skg.GaussianProcessRegressor(kernel=gp_kernel,optimizer=None)
    
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
            self.log.error('Gaussian process asked for new parameters but no new costs were provided.')
            raise ValueError
        
        new_params = []
        new_costs = []
        new_uncers = []
        new_bads = []
        new_costs_count = 0
        update_bads_flag = False
        
        while not self.costs_in_queue.empty():
            (param, cost, uncer, bad) = self.costs_in_queue.get_nowait()
            if bad:
                new_bads.append(self.data_count)
                if self.bad_defaults_set:
                    cost = self.default_bad_cost
                    uncer = self.default_bad_uncertainty
                else:
                    cost = self.worst_cost
                    uncer = self.cost_range*self.bad_uncer_frac
                    
            param = np.array(param, dtype=float)
            if not self.check_num_params(param):
                self.log.error('Incorrect number of parameters provided to Gaussian process learner:' + repr(param) + '. Number of parameters:' + str(self.num_params))
                raise ValueError
            if not self.check_in_boundary(param):
                self.log.warning('Parameters provided to Gaussian process learner not in boundaries:' + repr(param))
            cost = float(cost)
            if uncer < 0:
                self.log.error('Provided uncertainty must be larger or equal to zero:' + repr(uncer))
                uncer = max(float(uncer), self.minimum_uncertainty)
            
            new_costs_count += 1
            self.costs_count +=1
            
            cost_change_flag = False
            if cost > self.worst_cost:
                self.worst_cost = cost
                self.worst_index = self.costs_count
                cost_change_flag = True
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_params = param
                self.best_index =  self.costs_count
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
        self.archive_dict.update({'all_params':self.all_params,
                                  'all_costs':self.all_costs,
                                  'all_uncers':self.all_uncers,
                                  'best_cost':self.best_cost,
                                  'best_params':self.best_params,
                                  'best_index':self.best_index,
                                  'worst_cost':self.worst_cost,
                                  'worst_index':self.worst_index,
                                  'cost_range':self.cost_range,
                                  'fit_count':self.fit_count,
                                  'costs_count':self.costs_count,
                                  'params_count':self.params_count,
                                  'update_hyperparameters':self.update_hyperparameters,
                                  'length_scale':self.length_scale,
                                  'noise_level':self.noise_level})    
        

    
    def fit_gaussian_process(self):
        '''
        Fit the Gaussian process to the current data
        '''
        self.log.debug('Fitting Gaussian process.')
        if self.all_params.size==0 or self.all_costs.size==0 or self.all_uncers.size==0:
            self.log.error('Asked to fit GP but no data is in all_costs, all_params or all_uncers.')
            raise ValueError
        self.scaled_costs = self.cost_scaler.fit_transform(self.all_costs[:,np.newaxis])[:,0]
        self.scaled_uncers = self.all_uncers * self.cost_scaler.scale_
        self.gaussian_process.alpha_ = self.scaled_uncers
        self.gaussian_process.fit(self.all_params,self.scaled_costs)
        
        if self.update_hyperparameters:
            
            self.fit_count += 1
            self.gaussian_process.kernel = self.gaussian_process.kernel_
        
            last_hyperparameters = self.gaussian_process.kernel.get_params()
            
            if self.cost_has_noise:
                self.length_scale = last_hyperparameters['k1__length_scale']
                if isinstance(self.length_scale, float):
                    self.length_scale = np.array([self.length_scale])
                self.length_scale_history.append(self.length_scale)
                self.noise_level = last_hyperparameters['k2__noise_level']
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
            self.find_global_minima()
            end_dict.update({'predicted_best_parameters':self.predicted_best_parameters,
                             'predicted_best_cost':self.predicted_best_cost,
                             'predicted_best_uncertainty':self.predicted_best_uncertainty})
        if self.predict_local_minima_at_end:
            self.find_local_minima()
            end_dict.update({'local_minima_parameters':self.local_minima_parameters,
                             'local_minima_costs':self.local_minima_costs,
                             'local_minima_uncers':self.local_minima_uncers})
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
        self.predicted_best_uncertainty = self.predicted_best_scaled_uncertainty / self.cost_scaler.scale_
        
        self.archive_dict.update({'predicted_best_parameters':self.predicted_best_parameters,
                                  'predicted_best_scaled_cost':self.predicted_best_scaled_cost,
                                  'predicted_best_scaled_uncertainty':self.predicted_best_scaled_uncertainty,
                                  'predicted_best_cost':self.predicted_best_cost,
                                  'predicted_best_uncertainty':self.predicted_best_uncertainty})
        
        self.has_global_minima = True
        self.log.debug('Predicted global minima found.')   
    
    def find_local_minima(self):
        '''
        Performs a comprehensive search of the learner for all predicted local minima (and hence the global as well) in the landscape. Note, this can take a very long time. 
        
        Attributes:
            local_minima_parameters (list): list of all the parameters for local minima.
            local_minima_costs (list): list of all the costs at local minima.
            local_minima_uncers (list): list of all the uncertainties at local minima.
             
        '''
        self.log.info('Searching for all minima.')
        
        self.minima_tolerance = 10*self.search_precision
        
        self.number_of_local_minima = 0
        self.local_minima_parameters = []
        self.local_minima_costs = []
        self.local_minima_uncers = []
        
        search_bounds = list(zip(self.min_boundary, self.max_boundary))
        for start_params in self.all_params:
            result = so.minimize(self.predict_cost, start_params, bounds = search_bounds, tol=self.search_precision)
            curr_minima_params = result.x
            (curr_minima_cost,curr_minima_uncer) = self.gaussian_process.predict(curr_minima_params[np.newaxis,:],return_std=True)
            if all( not np.all( np.abs(params - curr_minima_params) < self.minima_tolerance ) for params in self.local_minima_parameters):    
                #Non duplicate point so add to the list
                self.number_of_local_minima += 1
                self.local_minima_parameters.append(curr_minima_params)
                self.local_minima_costs.append(curr_minima_cost)
                self.local_minima_uncers.append(curr_minima_uncer)
        
        self.archive_dict.update({'number_of_local_minima':self.number_of_local_minima,
                                  'local_minima_parameters':self.local_minima_parameters,
                                  'local_minima_costs':self.local_minima_costs,
                                  'local_minima_uncers':self.local_minima_uncers})
        
        self.has_local_minima = True
        self.log.info('Search completed')

    
class DifferentialEvolutionLearner(Learner, threading.Thread):

    """
    Adaption of the differential evolution algorithm in scipy. 
    
    Args:
        params_out_queue (queue): Queue for parameters sent to controller.
        costs_in_queue (queue): Queue for costs for gaussian process. This must be tuple
        end_event (event): Event to trigger end of learner.
        
    Keyword Args:
        evolution_strategy (Optional [string]): the differential evolution strategy to use, options are 'best1', 'best1', 'rand1' and 'rand2'. The default is 'rand2'.
        population_size (Optional [int]): multiplier proportional to the number of parameters in a generation. The generation population is set to population_size * parameter_num. Default 15.
        mutation_scale (Optional [tuple]): The mutation scale when picking new points. Otherwise known as differential weight. When provided as a tuple (min,max) a mutation constant is picked randomly in the interval. Default (0.5,1.0).
        crossover_probability (Optional [float]): The recombination constand or crossover probability, the probability a new points will be added to the population.
        restart_tolerance (Optional [float]): when the current population have a spread less than the initial tolerance, namely stdev(curr_pop) < restart_tolerance stdev(init_pop), it is likely the population is now in a minima, and so the search is started again.
        trust_region (Optional [float or array]): The trust region defines the maximum distance the learner will travel from the current best set of parameters. If None, the learner will search everywhere. If a float, this number must be between 0 and 1 and defines maximum distance the learner will venture as a percentage of the boundaries. If it is an array, it must have the same size as the number of parameters and the numbers define the maximum absolute distance that can be moved along each direction. 
        
    Attributes:
        has_trust_region (bool): Whether the learner has a trust region. 
        
    """

    # Dispatch of mutation strategy method (binomial or exponential).
    
    def __init__(self, 
                 strategy='rand2', 
                 population_size=15,
                 restart_tolerance=0.01, 
                 mutation=(0.5, 1), 
                 recombination=0.7, 
                 **kwargs):
        
        super(NelderMeadLearner,self).__init__(**kwargs)
        
        if strategy == 'best1':
            self.mutation_func = self._best1
        elif strategy == 'best2':
            self.mutation_func = self._best2
        elif strategy == 'rand1':
            self.mutation_func = self._rand1
        elif strategy == 'rand2':
            self.mutation_func = self._rand2
        else:
            self.log.error("Please select a valid mutation strategy")
            raise ValueError
        
        self.strategy = strategy
        self.restart_tolerance = restart_tolerance

        # Mutation constant should be in [0, 2). If specified as a sequence
        # then dithering is performed.
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            raise ValueError('The mutation constant must be a float in '
                             'U[0, 2), or specified as a tuple(min, max)'
                             ' where min < max and min, max are in U[0, 2).')

        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        self.cross_over_probability = recombination

        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or not
                np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        if maxiter is None:  # the default used to be None
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:  # the default used to be None
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = _make_random_gen(seed)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        self.num_population_members = popsize * self.parameter_count

        self.population_shape = (self.num_population_members,
                                 self.parameter_count)

        self._nfev = 0
        if init == 'latinhypercube':
            self.init_population_lhs()
        elif init == 'random':
            self.init_population_random()
        else:
            raise ValueError("The population initialization method must be one"
                             "of 'latinhypercube' or 'random'")

        self.disp = disp

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Each parameter range needs to be sampled uniformly. The scaled
        # parameter range ([0, 1)) needs to be split into
        # `self.num_population_members` segments, each of which has the following
        # size:
        segsize = 1.0 / self.num_population_members

        # Within each segment we sample from a uniform random distribution.
        # We need to do this sampling for each parameter.
        samples = (segsize * rng.random_sample(self.population_shape)

        # Offset each segment to cover the entire parameter range [0, 1)
                   + np.linspace(0., 1., self.num_population_members,
                                 endpoint=False)[:, np.newaxis])

        # Create an array for population of candidate solutions.
        self.population = np.zeros_like(samples)

        # Initialize population of candidate solutions by permutation of the
        # random samples.
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)

        # reset population energies
        self.population_energies = (np.ones(self.num_population_members) *
                                    np.inf)

        # reset number of function evaluations counter
        self._nfev = 0

    @property
    def x(self):
        """
        The best solution from the solver

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        """
        return self._scale_parameters(self.population[0])

    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        return (np.std(self.population_energies) /
                np.abs(np.mean(self.population_energies) + _MACHEPS))

    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.

        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nit, warning_flag = 0, False
        status_message = _status_message['success']

        # The population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies.
        # Although this is also done in the evolve generator it's possible
        # that someone can set maxiter=0, at which point we still want the
        # initial energies to be calculated (the following loop isn't run).
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        # do the optimisation.
        for nit in range(1, self.maxiter + 1):
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                status_message = _status_message['maxfev']
                break

            if self.disp:
                print("differential_evolution step %d: f(x)= %g"
                      % (nit,
                         self.population_energies[0]))

            # stop when the fractional s.d. of the population is less than tol
            # of the mean energy
            convergence = self.convergence

            if (self.callback and
                    self.callback(self._scale_parameters(self.population[0]),
                                  convergence=self.tol / convergence) is True):

                warning_flag = True
                status_message = ('callback function requested stop early '
                                  'by returning True')
                break

            if convergence < self.tol or warning_flag:
                break

        else:
            status_message = _status_message['maxiter']
            warning_flag = True

        DE_result = OptimizeResult(
            x=self.x,
            fun=self.population_energies[0],
            nfev=self._nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is not True))

        if self.polish:
            result = minimize(self.func,
                              np.copy(DE_result.x),
                              method='L-BFGS-B',
                              bounds=self.limits.T,
                              args=self.args)

            self._nfev += result.nfev
            DE_result.nfev = self._nfev

            if result.fun < DE_result.fun:
                DE_result.fun = result.fun
                DE_result.x = result.x
                DE_result.jac = result.jac
                # to keep internal state consistent
                self.population_energies[0] = result.fun
                self.population[0] = self._unscale_parameters(result.x)

        return DE_result

    def _calculate_population_energies(self):
        """
        Calculate the energies of all the population members at the same time.
        Puts the best member in first place. Useful if the population has just
        been initialised.
        """
        for index, candidate in enumerate(self.population):
            if self._nfev > self.maxfun:
                break

            parameters = self._scale_parameters(candidate)
            self.population_energies[index] = self.func(parameters,
                                                        *self.args)
            self._nfev += 1

        minval = np.argmin(self.population_energies)

        # put the lowest energy into the best solution position.
        lowest_energy = self.population_energies[minval]
        self.population_energies[minval] = self.population_energies[0]
        self.population_energies[0] = lowest_energy

        self.population[[0, minval], :] = self.population[[minval, 0], :]

    def __iter__(self):
        return self

    def __next__(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # the population may have just been initialized (all entries are
        # np.inf). If it has you have to calculate the initial energies
        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        if self.dither is not None:
            self.scale = (self.random_number_generator.rand()
                          * (self.dither[1] - self.dither[0]) + self.dither[0])

        for candidate in range(self.num_population_members):
            if self._nfev > self.maxfun:
                raise StopIteration

            # create a trial solution
            trial = self._mutate(candidate)

            # ensuring that it's in the range [0, 1)
            self._ensure_constraint(trial)

            # scale from [0, 1) to the actual parameter value
            parameters = self._scale_parameters(trial)

            # determine the energy of the objective function
            energy = self.func(parameters, *self.args)
            self._nfev += 1

            # if the energy of the trial candidate is lower than the
            # original population member then replace it
            if energy < self.population_energies[candidate]:
                self.population[candidate] = trial
                self.population_energies[candidate] = energy

                # if the trial candidate also has a lower energy than the
                # best solution then replace that as well
                if energy < self.population_energies[0]:
                    self.population_energies[0] = energy
                    self.population[0] = trial

        return self.x, self.population_energies[0]

    def next(self):
        """
        Evolve the population by a single generation

        Returns
        -------
        x : ndarray
            The best solution from the solver.
        fun : float
            Value of objective function obtained from the best solution.
        """
        # next() is required for compatibility with Python2.7.
        return self.__next__()

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

    def _ensure_constraint(self, trial):
        """
        make sure the parameters lie between the limits
        """
        for index, param in enumerate(trial):
            if param > 1 or param < 0:
                trial[index] = self.random_number_generator.rand()

    def _mutate(self, candidate):
        """
        create a trial vector based on a mutation strategy
        """
        trial = np.copy(self.population[candidate])

        rng = self.random_number_generator

        fill_point = rng.randint(0, self.parameter_count)

        if (self.strategy == 'randtobest1exp' or
                self.strategy == 'randtobest1bin'):
            bprime = self.mutation_func(candidate,
                                        self._select_samples(candidate, 5))
        else:
            bprime = self.mutation_func(self._select_samples(candidate, 5))

        if self.strategy in self._binomial:
            crossovers = rng.rand(self.parameter_count)
            crossovers = crossovers < self.cross_over_probability
            # the last one is always from the bprime vector for binomial
            # If you fill in modulo with a loop you have to set the last one to
            # true. If you don't use a loop then you can have any random entry
            # be True.
            crossovers[fill_point] = True
            trial = np.where(crossovers, bprime, trial)
            return trial

        elif self.strategy in self._exponential:
            i = 0
            while (i < self.parameter_count and
                   rng.rand() < self.cross_over_probability):

                trial[fill_point] = bprime[fill_point]
                fill_point = (fill_point + 1) % self.parameter_count
                i += 1

            return trial

    def _best1(self, samples):
        """
        best1bin, best1exp
        """
        r0, r1 = samples[:2]
        return (self.population[0] + self.scale *
                (self.population[r0] - self.population[r1]))

    def _rand1(self, samples):
        """
        rand1bin, rand1exp
        """
        r0, r1, r2 = samples[:3]
        return (self.population[r0] + self.scale *
                (self.population[r1] - self.population[r2]))

    def _best2(self, samples):
        """
        best2bin, best2exp
        """
        r0, r1, r2, r3 = samples[:4]
        bprime = (self.population[0] + self.scale *
                  (self.population[r0] + self.population[r1] -
                   self.population[r2] - self.population[r3]))

        return bprime

    def _rand2(self, samples):
        """
        rand2bin, rand2exp
        """
        r0, r1, r2, r3, r4 = samples
        bprime = (self.population[r0] + self.scale *
                  (self.population[r1] + self.population[r2] -
                   self.population[r3] - self.population[r4]))

        return bprime

    def _select_samples(self, candidate, number_samples):
        """
        obtain random integers from range(self.num_population_members),
        without replacement.  You can't have the original candidate either.
        """
        idxs = list(range(self.num_population_members))
        idxs.remove(candidate)
        self.random_number_generator.shuffle(idxs)
        idxs = idxs[:number_samples]
        return idxs


def _make_random_gen(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)



