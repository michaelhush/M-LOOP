#!/usr/bin/env python
import argparse

parser = argparse.ArgumentParser(description='Plot cross sections of the predicted landscape. Must be run from the same directory as M-LOOP was run.')
parser.add_argument("controller_filename")
parser.add_argument("learner_filename")
args = parser.parse_args()

import mloop.visualizations as mlv
import mloop.utilities as mlu
import logging

mlu.config_logger(log_filename=None, console_log_level=logging.DEBUG)

mlv.show_all_default_visualizations_from_archive(args.controller_filename, args.learner_filename)
