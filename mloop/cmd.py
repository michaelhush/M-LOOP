'''
Module of command line tools that can be used to execute mloop.
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import sys
import argparse
import mloop as ml
import mloop.launchers as mll
import multiprocessing as mp

def run_mloop():
    '''
    M-LOOP Launcher
    
    Starts an instance of M-LOOP configured using a configuration file. 
    
    Takes the following command line options
    
    -c filename for configuration file
    
    -h display help
    
    the default name for the configuration is "ExpConfig.txt"
    '''
    
    parser = argparse.ArgumentParser(description='M-LOOP Launcher \n Version:' + ml.__version__+'\n \n Starts a new instance of M-LOOP based a on configuration file.', 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c','--configFile',default='exp_config.txt',help='Filename of configuration file.')
    parser.add_argument('-v','--version', action='version', version=ml.__version__)
    
    args = parser.parse_args()
    
    config_filename = args.configFile
    
    _ = mll.launch_from_file(config_filename)

        