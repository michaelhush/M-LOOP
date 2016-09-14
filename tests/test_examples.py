'''
Unit test for all of the example scripts provided in the examples folder.
'''
from __future__ import absolute_import, division, print_function

import os
import unittest
import mloop.testing as mlt
import mloop.launchers as mll
import mloop.utilities as mlu
import logging
import numpy as np
import multiprocessing

class TestExamples(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        os.chdir(mlu.mloop_path + '/../tests')
        cls.override_dict = {'file_log_level':logging.DEBUG,'console_log_level':logging.DEBUG,'visualizations':False}
        cls.fake_experiment = mlt.FakeExperiment()
        cls.fake_experiment.start()
    
    @classmethod
    def tearDownClass(cls):
        cls.fake_experiment.end_event.set()
        cls.fake_experiment.join()
    
    def test_controller_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/controller_config.txt', 
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
    
    def test_extras_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/extras_config.txt', 
                                          num_params=1,
                                          target_cost = 0.1,
                                          **self.override_dict) 
        self.asserts_for_cost_and_params(controller)
    
    def test_logging_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/logging_config.txt',
                                          num_params=1,
                                          target_cost = 0.1,
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
        
    def test_random_simple_config(self):
        _ = mll.launch_from_file(mlu.mloop_path+'/../examples/random_simple_config.txt', 
                                 **self.override_dict)
        
    def test_random_complete_config(self):
        _ = mll.launch_from_file(mlu.mloop_path+'/../examples/random_complete_config.txt', 
                                 **self.override_dict)
        
    def test_nelder_mead_simple_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/nelder_mead_simple_config.txt',
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
    
    def test_nelder_mead_complete_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/nelder_mead_complete_config.txt',
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
    
    def test_gaussian_process_simple_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/gaussian_process_simple_config.txt',
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
     
    def test_gaussian_process_complete_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/gaussian_process_complete_config.txt',
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
    
    def test_tutorial_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/tutorial_config.txt',
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
    
    def asserts_for_cost_and_params(self,controller):
        self.assertTrue(controller.best_cost<=controller.target_cost)
        self.assertTrue(np.sum(np.square(controller.best_params))<=controller.target_cost)
    
    
if __name__ == "__main__":
    mp.freeze_support()
    unittest.main()