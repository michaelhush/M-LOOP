'''
Module of the interfaces used to connect the controller to the experiment. 
'''

import time
import os
import queue
import multiprocessing as mp
import mloop.utilities as mlu
import mloop.testing as mlt
import logging

def create_interface(interface_type='file', 
                      **interface_config_dict):
    '''
    Start a new interface with the options provided.
    
    Args:
        interface_type (Optional [str]): Defines the type of interface, currently the only option is 'file'. Default 'file'.
        **interface_config_dict : Options to be passed to interface.
        
    Returns:
        interface : An interface as defined by the keywords
    '''
    log = logging.getLogger(__name__)
    
    if interface_type=='file':
        file_interface = FileInterface(**interface_config_dict)
        log.info('Using the file interface with the experiment.')
    else:
        log.error('Unknown interface type:' + repr(interface_type))
        raise ValueError
    
    return file_interface

class InterfaceInterrupt(Exception):
    '''
    Exception that is raised when the interface is ended with the end event, or some other interruption.  
    '''
    def __init__(self):
        super().__init__()
    

class Interface(mp.Process):
    '''
    A abstract class for interfaces which populate the costs_in_queue and read from the params_out_queue. Inherits from Thread
    
    Args:
        interface_wait (Optional [float]): Time between polling when needed in interface.
        
    Keyword Args: 
        interface_wait (float): Wait time when polling for files or queues is needed.
        
    Arguments:
        params_out_queue (queue): Queue for parameters to next be run by experiment.
        costs_in_queue (queue): Queue for costs (and other details) that have been returned by experiment.
        end_event (event): Event which triggers the end of the interface. 
    
    
    '''

    def __init__(self,
                 interface_wait = 1, 
                 **kwargs):
        
        super().__init__()
        self.log = logging.getLogger(__name__)
        self.log.debug('Creating interface.')
        
        self.params_out_queue = mp.Queue()
        self.costs_in_queue = mp.Queue()
        self.end_event = mp.Event()
        
        self.interface_wait = float(interface_wait)
        if self.interface_wait<=0:
            self.log.error('Interface wait time must be a positive number.')
            raise ValueError
        
        self.remaining_kwargs = kwargs
    
    def add_mp_safe_log(self,log_queue):
        '''
        Add a multiprocess safe log based using a queue (which is presumed to be listened to by a QueueListener).
        '''
        self.log = logging.getLogger(__name__)
        que_handler = logging.handlers.QueueHandler(log_queue)
        self.log.addHandler(que_handler)
        self.log.propagate = False
    
    def run(self):
        '''
        The run sequence for the interface. This method does not need to be overloaded create a working interface. 
        
        '''
        self.log.debug('Entering main loop of interface.')
        try:
            while not self.end_event.is_set():
                try:
                    params_dict = self.params_out_queue.get(True, self.interface_wait)
                except queue.Empty:
                    continue
                else:
                    cost_dict = self._get_next_cost_dict(params_dict)
                    self.costs_in_queue.put(cost_dict)
        except InterfaceInterrupt:
            pass
        self.log.debug('Interface ended')
        #self.log = None
        
    def _get_next_cost_dict(self,params_dict):
        '''
        Abstract method. This is the only method that needs to be implemented to make a working interface. Given the parameters the interface must then produce a new cost. This may occur by running an experiment or program. If you wish to abruptly end this interface for whatever rease please raise the exception InterfaceInterrupt, which will then be safely caught.
        
        Args:
            params_dict (dictionary): A dictionary containing the parameters. Use params_dict['params'] to access them.
        
        Returns:
            cost_dict (dictionary): The cost and other properties derived from the experiment when it was run with the parameters. If just a cost was produced provide {'cost':[float]}, if you also have an uncertainty provide {'cost':[float],'uncer':[float]}. If the run was bad you can simply provide {'bad':True}. For completeness you can always provide all three using {'cost':[float],'uncer':[float],'bad':[bool]}. Providing any extra keys will also be saved byt he controller.
        '''
        pass
    
class FileInterface(Interface):
    '''
    Interfaces between the files produced by the experiment and the queues accessed by the controllers. 
    
    Args:
        params_out_queue (queue): Queue for parameters to next be run by experiment.
        costs_in_queue (queue): Queue for costs (and other details) that have been returned by experiment.
        
    Keyword Args:
        interface_out_filename (Optional [string]): filename for file written with parameters.
        interface_in_filename (Optional [string]): filename for file written with parameters.
        interface_file_type (Optional [string]): file type to be written either 'mat' for matlab or 'txt' for readible text file. Defaults to 'txt'.
    '''
    
    def __init__(self,
                 interface_out_filename=mlu.default_interface_out_filename, 
                 interface_in_filename=mlu.default_interface_in_filename,
                 interface_file_type=mlu.default_interface_file_type,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.out_file_count = 0
        self.in_file_count = 0
        
        if mlu.check_file_type_supported(interface_file_type):
            self.out_file_type = str(interface_file_type)
            self.in_file_type = str(interface_file_type)
        else:
            self.log.error('File out type is not supported:' + interface_file_type)
        self.out_filename = str(interface_out_filename)
        self.total_out_filename = self.out_filename + '.' + self.out_file_type
        self.in_filename = str(interface_in_filename)
        self.total_in_filename = self.in_filename + '.' + self.in_file_type

    def _get_next_cost_dict(self,params_dict):
        '''
        Implementation of file read in and out. Put parameters into a file and wait for a cost file to be returned.
        '''
        self.out_file_count += 1
        self.log.debug('Writing out_params to file. Count:' + repr(self.out_file_count))
        self.last_params_dict = params_dict
        mlu.save_dict_to_file(self.last_params_dict,self.total_out_filename,self.out_file_type)
        while not self.end_event.is_set():
            if os.path.isfile(self.total_in_filename):
                time.sleep(mlu.filewrite_wait) #wait for file to be written to disk
                try:
                    in_dict = mlu.get_dict_from_file(self.total_in_filename, self.in_file_type)
                except IOError:
                    self.log.warning('Unable to open ' + self.total_in_filename + '. Trying again.')
                    continue
                except (ValueError,SyntaxError):
                    self.log.error('There is something wrong with the syntax or type of your file:' + self.in_filename + '.' + self.in_file_type)
                    raise
                os.remove(self.total_in_filename)
                self.in_file_count += 1
                self.log.debug('Putting dict from file onto in queue. Count:' + repr(self.in_file_count))
                break
            else:
                time.sleep(self.interface_wait)
        else:
            raise InterfaceInterrupt
        return in_dict
        
    
class TestInterface(Interface):
    '''
    Interface for testing. Returns fake landscape data directly to learner.
    
    Args:
        params_out_queue (queue): Parameters to be used to evaluate fake landscape.
        costs_in_queue (queue): Queue for costs (and other details) that have been calculated from fake landscape.
        
    Keyword Args:
        test_landscape (Optional [TestLandscape]): Landscape that can be given a set of parameters and a cost and other values. If None creates a the default landscape. Default None 
        out_queue_wait (Optional [float]): Time in seconds to wait for queue before checking end flag.
    
    '''
    def __init__(self, 
                 test_landscape=None,
                 **kwargs):
        
        super().__init__(**kwargs)
        if test_landscape is None:
            self.test_landscape = mlt.TestLandscape()
        else:
            self.test_landscape = test_landscape
        self.test_count = 0
    
    def add_mp_safe_log(self,log_queue):
        super().add_mp_safe_log(log_queue)
        self.test_landscape.add_mp_safe_log(log_queue)
    
    def _get_next_cost_dict(self, params_dict):
        '''
        Test implementation. Gets the next cost from the test_landscape.
        '''
        
        self.test_count +=1
        self.log.debug('Test interface evaluating cost. Num:' + repr(self.test_count))
        try:
            params = params_dict['params']
        except KeyError as e:
            self.log.error('You are missing ' + repr(e.args[0]) + ' from the in params dict you provided through the queue.')
            raise
        cost_dict = self.test_landscape.get_cost_dict(params)
        return cost_dict
        
        
        
        
    