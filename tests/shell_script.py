#! /usr/bin/env python

from __future__ import absolute_import, division, print_function
__metaclass__ = type

import sys
import mloop.testing as mlt
import numpy as np

def main(argv):
    
    params = np.array([float(v) for v in argv])
    tester = mlt.TestLandscape()
    cost_dict = tester.get_cost_dict(params)
    
    print('M-LOOP_start')
    print('cost = '+str(cost_dict['cost']))
    print('M-LOOP_end')
    
if __name__ == '__main__':
    main(sys.argv[1:])