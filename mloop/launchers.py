'''
Modules of launchers used to start M-LOOP.  
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import logging
import mloop.utilities as mlu
import mloop.controllers as mlc
import mloop.interfaces as mli
import mloop.visualizations as mlv

def launch_from_file(config_filename, 
                     **kwargs):
    '''
    Launch M-LOOP using a configuration file. See configuration file documentation.
    
    Args:
        config_filename (str): Filename of configuration file 
        **kwargs : keywords that override the keywords in the file.
       
    Returns:
        controller (Controller): Controller for optimization.
    '''
    try:
        file_kwargs = mlu.get_dict_from_file(config_filename,'txt')
    except (IOError, OSError):
        print('Unable to open M-LOOP configuration file:' + repr(config_filename))
        raise
    
    file_kwargs.update(kwargs)
    #Main run sequence
    #Create interface and extract unused keywords
    interface = mli.create_interface(**file_kwargs)
    file_kwargs = interface.remaining_kwargs
    #Create controller and extract unused keywords
    controller = mlc.create_controller(interface, **file_kwargs)
    file_kwargs = controller.remaining_kwargs
    #Extract keywords for post processing extras, and raise an error if any keywords were unused. 
    extras_kwargs = _pop_extras_kwargs(file_kwargs)
    if file_kwargs:
        logging.getLogger(__name__).error('Unused extra options provided:' + repr(file_kwargs))
        raise ValueError
    #Run the actual optimization
    controller.optimize()
    #Launch post processing extras
    launch_extras(controller, **extras_kwargs)  
    
    return controller 

def launch_extras(controller,visualizations=True, **kwargs):
    '''
    Launch post optimization extras. Including visualizations.
    
    Keyword Args:
        visualizations (Optional [bool]): If true run default visualizations for the controller. Default false. 
    '''
    if visualizations:
        mlv.show_all_default_visualizations(controller)
    
def _pop_extras_kwargs(kwargs):
    '''
    Remove the keywords used in the extras section (if present), and return them.
    
    Returns:
        tuple made of (extras_kwargs, kwargs), where extras_kwargs are keywords for the extras and kwargs are the others that were provided. 
        
    '''
    extras_kwargs={}
    if 'visualizations' in kwargs:
        extras_kwargs['visualizations'] = kwargs.pop('visualizations')
    return extras_kwargs
    