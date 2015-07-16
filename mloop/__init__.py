'''
Created on 16 Jul 2015

M-LOOP
------

Machine-Learning Online Optimization Packaage

Python package for performing online optimization based on machine learning with quantum experiments.

@author: michaelhush
'''

import sys
import numpy as nm
import numpy.random as nr
import numpy.linalg as nl
import scipy.io as si
import scipy.optimize as so
import time
import os
import dill
import math

from sklearn import gaussian_process as slgp

__all__ = ["expcontroller","expinterface","learnercontrollers","neldercontroller","randomcontroller"]
