'''
Module of classes used to test M-LOOP.
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import numpy as np
import threading
import mloop.utilities as mlu
import numpy.random as nr
import logging
import os
import time

class TestLandscape():
    '''
    Produces fake landscape data for testing, default functions are set for each of the methods which can then be over ridden.
        
    Keyword Args:
        num_parameters (Optional [int]): Number of parameters for landscape, defaults to 1.
    '''
    
    def __init__(self, num_params = 1):
        self.log = logging.getLogger(__name__)
        self.test_eval_num = 0
        self.num_params = num_params
        self.set_default_landscape()
    
    def set_default_landscape(self):
        '''
        Set landscape functions to their defaults
        '''
        self.log.debug('Setting default landscapes')
        self.cost_function = lambda p: np.sum(np.square(p))
        self.noise_function = lambda p,c: (c,0)
        self.bad_function = lambda p,c,u: (c,u,False)
        
        self.expected_minima = np.zeros((self.num_params,))
        
    
    def set_random_quadratic_landscape(self, min_region, max_region, random_scale=True, min_scale=0, max_scale=3):
        '''
        Make a quadratic function with a minimum randomly placed in a region with random scales
        
        Args:
            min_region (array): minimum region boundary
            max_region (array): maximum region boundary
        
        Keyword Args:
            random_scale (Optional bool): If true randomize the scales of the parameters. Default True.
            min_scale (float): Natural log of minimum scale factor. Default 0.
            max_scale (float): Natural log of maximum scale factor. Default 3.
        '''
        mini = min_region + nr.rand(self.num_params) * (max_region - min_region)
        scal = np.exp(min_scale + nr.rand(self.num_params) * (max_scale - min_scale))
        self.set_quadratic_landscape(minimum = mini,scale = scal)
    
    def set_quadratic_landscape(self, minimum = None, scale = None):
        '''
        Set deterministic part of landscape to be a quadratic.
        
        with the formula::
        
            c(x) = \sum_i a_i * (x_i - x_0,i)^2
        
        where x_i are the parameters, x_0,i is the location of the minimum and a_i are the scaling factors.
        
        Keyword Args:
            minimum (Optional [array]): Location of the minimum. If set to None is at the origin. Default None.
            scales (Optional [array]): scaling of quadratic along the dimention specified. If set to None the scaling is one.
        '''
        if minimum is None:
            minimum = np.zeros((self.num_params,))
        if scale is None:
            scale = 1
        self.cost_minimum = minimum
        self.cost_scale = scale
        self.cost_function = lambda p : np.sum(self.cost_scale*np.square(p - self.cost_minimum))
        self.expected_minima = self.cost_minimum
        self.log.debug('Test Minimum at:' + repr(self.cost_minimum))
        self.log.debug('Test Scales are:' + repr(self.cost_scale))
        self.log.debug('Test Cost minimum:' + repr(self.cost_function(p=self.cost_minimum)))
    
    def set_noise_function(self, proportional=0.0, absolute=0.0):
        '''
        Adds noise to the function.
        
        with the formula::
        
            c'(c,x) = c (1 + s_p p) + s_a a
        
        where s_i are gaussian random variables, p is the proportional noise factor and a is the absolute noise factor, and c is the cost before noise is added
        
        the uncertainty is then::
        
            u = sqrt((cp)^2 + a^2)

        Keyword Args:
            proportional (Optional [float]): the proportional factor. Defaults to 0
            absolute (Optional [float]): the absolute factor. Defaults to 0
        '''
        
        self.noise_prop = proportional
        self.noise_abs = absolute
        self.noise_function = lambda p,c,u : (c *(1 + nr.normal()*self.noise_prop) + nr.normal()*self.noise_abs,np.sqrt((c*self.noise_prop)**2 + (self.noise_abs)**2))
    
    def set_bad_region(self, min_boundary, max_boundary, bad_cost=None, bad_uncer=None):
        '''
        Adds a region to landscape that is reported as bad.
        
        Args:
            min_boundary (array): mininum boundary for bad region
            max_boundary (array): maximum boundary for bad region
        
        
        '''
        self.bad_min_boundary = min_boundary
        self.bad_max_boundary = max_boundary
        if bad_cost is None and bad_uncer is None:
            self.bad_function = lambda p,c,u : (c, u, np.all(p >= self.bad_min_boundary)&np.all(self.bad_max_boundary >= p))
        elif bad_cost is not None and bad_uncer is not None:
            self.bad_cost = bad_cost
            self.bad_uncer = bad_uncer
            self.bad_function  = lambda p,c,u : (self.bad_cost,self.bad_uncer, True) if np.all(p >= self.bad_min_boundary)&np.all(self.bad_max_boundary >= p) else (c,u,False)
        else:
            self.log.error('When defining bad region behavoir bad_cost and bad_uncer must both be None or neither of them are none. bad_cost:' + repr(bad_cost) +'. bad_uncer' + repr(bad_uncer) )
            raise ValueError
    
    def get_cost_dict(self,params):
        '''
        Return cost from fake landscape given parameters.
        
        Args:
            params (array): Parameters to evaluate cost. 
        '''
        self.test_eval_num +=1
        self.log.debug('Test function called, num:' + repr(self.test_eval_num))
        mean_cost = self.cost_function(p=params)
        (noise_cost, noise_uncer) = self.noise_function(c=mean_cost,p=params)
        (final_cost, final_uncer, final_bad) = self.bad_function(c=noise_cost,u=noise_uncer,p=params)
        
        return_dict = {}
        return_dict['cost']=final_cost
        return_dict['uncer']=final_uncer
        return_dict['bad']=final_bad
        return_dict['mean_cost']=mean_cost
        return_dict['noise_cost']=noise_cost
        return_dict['noise_uncer']=noise_uncer
        return_dict['test_eval_num']=self.test_eval_num
        return return_dict

class FakeExperiment(threading.Thread):
    '''
    Pretends to be an experiment and reads files and prints files based on the costs provided by a TestLandscape. Executes as a thread.
    
    Keyword Args:
        test_landscape (Optional TestLandscape): landscape to generate costs from.
        experiment_file_type (Optional [string]): currently supports: 'txt' where the output is a text file with the parameters as a list of numbers, and 'mat' a matlab file with variable parameters with the next_parameters. Default is 'txt'. 
        
    Attributes
        self.end_event (Event): Used to trigger end of experiment. 
    '''
    
    def __init__(self,
                 test_landscape = None,
                 experiment_file_type=mlu.default_interface_file_type,
                 exp_wait = 0,
                 poll_wait = 1,
                 **kwargs):
        
        super(FakeExperiment,self).__init__()
        
        if test_landscape is None:
            self.test_landscape = TestLandscape()
        else:
            self.test_landscape = test_landscape
        
        self.log = logging.getLogger(__name__)
        self.exp_wait = float(exp_wait)
        self.poll_wait = float(poll_wait)
        self.out_file_type = str(experiment_file_type)
        self.in_file_type = str(experiment_file_type)
        
        self.total_out_filename = mlu.default_interface_in_filename + '.' + self.out_file_type
        self.total_in_filename = mlu.default_interface_out_filename + '.' + self.in_file_type
        self.end_event = threading.Event()
        self.test_count =0
    
    def set_landscape(self,test_landscape):
        '''
        Set new test landscape.
        
        Args:
            test_landscape (TestLandscape): Landscape to generate costs from.
        '''
        self.test_landscape = test_landscape
    
    def run(self):
        '''
        Implementation of file read in and out. Put parameters into a file and wait for a cost file to be returned.
        '''
        
        self.log.debug('Entering FakeExperiment loop')
        while not self.end_event.is_set():
            if os.path.isfile(self.total_in_filename):
                time.sleep(mlu.filewrite_wait) #wait for file to be written
                try:
                    in_dict = mlu.get_dict_from_file(self.total_in_filename, self.in_file_type)
                except IOError:
                    self.log.warning('Unable to open ' + self.total_in_filename + '. Trying again.')
                    continue
                except (ValueError,SyntaxError):
                    self.log.error('There is something wrong with the syntax or type of your file:' + self.in_filename + '.' + self.in_file_type)
                    raise
                
                os.remove(self.total_in_filename)
                self.test_count +=1
                self.log.debug('Test exp evaluating cost. Num:' + repr(self.test_count))
                try:
                    params = in_dict['params']
                except KeyError as e:
                    self.log.error('You are missing ' + repr(e.args[0]) + ' from the in params dict you provided through the queue.')
                    raise
                cost_dict = self.test_landscape.get_cost_dict(params)
                time.sleep(self.exp_wait)
                mlu.save_dict_to_file(cost_dict, self.total_out_filename, self.out_file_type)
                
            else:
                time.sleep(self.poll_wait)
        self.log.debug('Ended FakeExperiment')
