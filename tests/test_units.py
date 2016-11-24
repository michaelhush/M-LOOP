'''
Unit test for all of the example scripts provided in the examples folder.
'''
from __future__ import absolute_import, division, print_function

import os
import unittest
import math
import mloop.interfaces as mli
import mloop.controllers as mlc
import numpy as np
import multiprocessing as mp

class CostListInterface(mli.Interface):
    def __init__(self, cost_list):
        super(CostListInterface,self).__init__()
        self.call_count = 0
        self.cost_list = cost_list
    def get_next_cost_dict(self,params_dict):
        if np.isfinite(self.cost_list[self.call_count]):
            cost_dict = {'cost': self.cost_list[self.call_count]}
        else:
            cost_dict = {'bad': True}
        self.call_count += 1
        return cost_dict

class TestUnits(unittest.TestCase):
    
    def test_max_num_runs(self):
        cost_list = [5.,4.,3.,2.,1.]
        interface = CostListInterface(cost_list)
        controller = mlc.create_controller(interface, 
                                           max_num_runs = 5, 
                                           target_cost = -1,
                                           max_num_runs_without_better_params = 10)
        controller.optimize()
        self.assertTrue(controller.best_cost == 1.)
        self.assertTrue(np.array_equiv(np.array(controller.in_costs),
                                        np.array(cost_list)))
        
        
    def test_max_num_runs_without_better_params(self):
        cost_list = [1.,2.,3.,4.,5.]
        interface = CostListInterface(cost_list)
        controller = mlc.create_controller(interface, 
                                           max_num_runs = 10, 
                                           target_cost = -1,
                                           max_num_runs_without_better_params = 4)
        controller.optimize()
        self.assertTrue(controller.best_cost == 1.)
        self.assertTrue(np.array_equiv(np.array(controller.in_costs),
                                        np.array(cost_list)))
        
    def test_target_cost(self):
        cost_list = [1.,2.,-1.]
        interface = CostListInterface(cost_list)
        controller = mlc.create_controller(interface, 
                                           max_num_runs = 10, 
                                           target_cost = -1,
                                           max_num_runs_without_better_params = 4)
        controller.optimize()
        self.assertTrue(controller.best_cost == -1.)
        self.assertTrue(np.array_equiv(np.array(controller.in_costs),
                                        np.array(cost_list)))
    
    def test_bad(self):
        cost_list = [1., float('nan'),2.,float('nan'),-1.]
        interface = CostListInterface(cost_list)
        controller = mlc.create_controller(interface, 
                                           max_num_runs = 10, 
                                           target_cost = -1,
                                           max_num_runs_without_better_params = 4)
        controller.optimize()
        self.assertTrue(controller.best_cost == -1.)
        for x,y in zip(controller.in_costs,cost_list):
            self.assertTrue(x==y or (math.isnan(x) and math.isnan(y)))
    
if __name__ == "__main__":
    mp.freeze_support()
    unittest.main()