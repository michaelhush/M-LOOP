'''
M-LOOP: Machine-Learning Online Optimization Packaage

Python package for performing automated, online optimization of scientific experiments or anything that can be computer controlled. The package employs machine learning algorithms to rapidly find optimal parameters for systems under control. 

If you use this package please cite the article http://www.nature.com/articles/srep25890.

To contribute to the project or report a bug visit the project's github https://github.com/michaelhush/M-LOOP.
'''

import os

__version__= "2.0.2"
__all__ = ['controllers','interfaces','launchers','learners','testing','utilities','visualizations']

#Add a null handler in case the user does not run config_logger() before running the optimization
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())