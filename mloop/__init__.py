'''
M-LOOP: Machine-Learning Online Optimization Packaage

Python package for performing automated, online optimization of scientific experiments or anything that can be computer controlled. The package employs machine learning algorithms to rapidly find optimal parameters for systems under control. 

If you use this package please cite the article http://www.nature.com/articles/srep25890.

To contribute to the project or report a bug visit the project's github https://github.com/michaelhush/M-LOOP.
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import os

__version__= "2.1.0"
__all__ = ['controllers','interfaces','launchers','learners','testing','utilities','visualizations','cmd']