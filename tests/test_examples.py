'''
Unit test for all of the example scripts provided in the examples folder.
'''

import os
import unittest
import mloop.testing as mlt
import mloop.launchers as mll
import mloop.utilities as mlu
import logging
import numpy as np

class TestExamples(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        os.chdir(mlu.mloop_path + '/../tests')
        cls.override_dict = {'file_log_level':logging.DEBUG,'console_log_level':logging.WARNING,'visualizations':False}
        cls.fake_experiment = mlt.FakeExperiment()
        cls.fake_experiment.start()
    
    @classmethod
    def tearDownClass(cls):
        cls.fake_experiment.end_event.set()
        cls.fake_experiment.join()
    
    def test_complete_controller_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/complete_controller_config.txt', 
                                          num_params=1,
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
    
    def test_complete_extras_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/complete_extras_config.txt', 
                                          num_params=1,
                                          target_cost = 0.1,
                                          **self.override_dict) 
        self.asserts_for_cost_and_params(controller)
    
    def test_complete_logging_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/complete_logging_config.txt',
                                          num_params=1,
                                          target_cost = 0.1,
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
        
    def test_simple_random_config(self):
        _ = mll.launch_from_file(mlu.mloop_path+'/../examples/simple_random_config.txt', 
                                 **self.override_dict)
        
    def test_complete_random_config(self):
        _ = mll.launch_from_file(mlu.mloop_path+'/../examples/complete_random_config.txt', 
                                 **self.override_dict)
        
    def test_simple_nelder_mead_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/simple_nelder_mead_config.txt',
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
        
    def test_complete_nelder_mead_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/complete_nelder_mead_config.txt',
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
    
    def test_simple_gaussian_process_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/simple_gaussian_process_config.txt',
                                          **self.override_dict)
        self.asserts_for_cost_and_params(controller)
        
    def test_complete_gaussian_process_config(self):
        controller = mll.launch_from_file(mlu.mloop_path+'/../examples/complete_gaussian_process_config.txt',
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
    unittest.main()