'''
Created on 15Jun.,2016

@author: michaelhush
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import mloop.controllers as mlc
import mloop.interfaces as mli
import mloop.testing as mlt
import mloop.visualizations as mlv
import mloop.utilities as mlu
import numpy as np
import logging
import matplotlib.pyplot as plt

def main():
    
    mlu.config_logger(file_log_level=logging.DEBUG,
                      console_log_level=logging.DEBUG)
    
    log = logging.getLogger('mloop.main')
    
    log.info("Making queues")
    
    tnum_params = 10
    
    tmin_boundary=[-10.] * tnum_params
    tmax_boundary=[ 10.] * tnum_params
    
    tmax_num_runs = 40
    tcost = -1.
    
    log.info("Making landscape")
    
    tlandscape = mlt.TestLandscape(num_params = tnum_params)
    
    tlandscape.set_random_quadratic_landscape(np.array(tmin_boundary),np.array(tmax_boundary),random_scale=False)

    which = 4
    if which == 0:
        log.info("Running random controller")
        interface = mli.TestInterface(test_landscape=tlandscape)
        random_controller = mlc.create_controller(interface, 
                                                  controller_type='random', 
                                                  max_num_runs= tmax_num_runs, 
                                                  num_params=tnum_params, 
                                                  min_boundary=tmin_boundary, 
                                                  max_boundary=tmax_boundary,
                                                  trust_region=0.1)
        random_controller.optimize()
        mlv.show_all_default_visualizations(random_controller, show_plots=False)
        log.info("Random controller ended")
    elif which == 1:
        log.info("Running Nelder-Mead controller")
        interface = mli.TestInterface(test_landscape=tlandscape)
        nelder_mead_controller = mlc.create_controller(interface, 
                                                       controller_type='nelder_mead', 
                                                       max_num_runs= tmax_num_runs, 
                                                       num_params=tnum_params, 
                                                       min_boundary=tmin_boundary,
                                                       max_boundary=tmax_boundary)            
        nelder_mead_controller.optimize()
        mlv.show_all_default_visualizations(nelder_mead_controller, show_plots=False)
        log.info("Running Nelder-Mead controller")
    elif which == 2:
        log.info("Running differential evolution controller")
        interface = mli.TestInterface(test_landscape=tlandscape)
        diff_evo_controller = mlc.create_controller(interface, 
                                                       controller_type='differential_evolution', 
                                                       evolution_strategy='rand2',
                                                       max_num_runs= tmax_num_runs, 
                                                       num_params=tnum_params, 
                                                       min_boundary=tmin_boundary,
                                                       max_boundary=tmax_boundary)            
        diff_evo_controller.optimize()
        mlv.show_all_default_visualizations(diff_evo_controller, show_plots=False)
        log.info("Running differential evolution controller")
    elif which == 3:
        log.info("Running Gaussian process controller")
        interface = mli.TestInterface(test_landscape=tlandscape)
        gp_controller = mlc.create_controller(interface, 
                                              controller_type='gaussian_process', 
                                              no_delay=False, 
                                              max_num_runs= tmax_num_runs, 
                                              target_cost = tcost,
                                              num_params=tnum_params, 
                                              min_boundary=tmin_boundary, 
                                              max_boundary=tmax_boundary)
        #length_scale = 1.)            
        gp_controller.optimize()
        mlv.show_all_default_visualizations(gp_controller, show_plots=False)
        log.info("Gaussian process controller ended")
    elif which == 4:
        log.info("Running Neural net controller")
        interface = mli.TestInterface(test_landscape=tlandscape)
        nn_controller = mlc.create_controller(interface, 
                                              controller_type='neural_net', 
                                              no_delay=False, 
                                              max_num_runs= tmax_num_runs, 
                                              target_cost = tcost,
                                              num_params=tnum_params, 
                                              min_boundary=tmin_boundary, 
                                              max_boundary=tmax_boundary)            
        nn_controller.optimize()
        mlv.show_all_default_visualizations(nn_controller, show_plots=False)
        log.info("Neural net process controller ended")
    else:
        raise ValueError
    
    log.info("True minimum:" + str(tlandscape.expected_minima))
    log.info("True minimum value:" + str(tlandscape.cost_function(p=tlandscape.expected_minima)))
    
    log.info("Visualizations started.")
    
    plt.show()
    
    log.info("MLOOP Quick Test ended")


if __name__ == '__main__':
    main()
