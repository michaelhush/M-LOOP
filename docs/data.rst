.. _sec-data:

====
Data
====

M-LOOP saves all data produced by the experiment in archives which are saved to disk during and after the optimization run. The archives also contain information derived from the data, including the machine learning model for how the experiment works. Here we explain how to interpret the file archives. 

File Formats
============

M-LOOP currently supports three file formats for all file input and output. 

- 'txt' text files: Human readable text files. This is the default file format for all outputs. The advantage of text files is they are easy to read, and there will be no format compatibility issues in the future. However, there will be some loss of precision in your data. To ensure you keep all significant figure you may want to use 'pkl' or 'mat'.
- 'mat' MATLAB files: Matlab files that can be opened and written with MATLAB or `numpy <http://www.numpy.org/>`_.
- 'pkl' pickle files: a serialization of a python dictionary made with `pickle <https://docs.python.org/3/library/pickle.html>`. Your data can be retrieved from this dictionary using the appropriate keywords. 

File Keywords
=============

The archives contain a set of keywords/variable names with associated data. The quickest way to understand what the values mean for a particular keyword is to :ref:`search` the documentation for a description. 

For a comprehensive list of all the keywords looks at the attributes described in the API. 

For the controller archive see :ref:`api-controllers`.

For the learner archive see :ref:`api-learners`. The generic keywords are described in the class Learner, with learner specific options described in the derived classes, for example GaussianProcessLearner.

Converting files
================

If for whatever reason you want to convert files between the formats you can do so using the utilities module of M-LOOP. For example the following python code will convert the file controller_archive_2016-08-18_12-18.pkl from a 'pkl' file to a 'mat' file::

   import mloop.utilities as mlu

   saved_dict = mlu.get_dict_from_file('./M-LOOP_archives/controller_archive_2016-08-18_12-18.pkl') 
   mlu.save_dict_to_file(saved_dict,'./M-LOOP_archives/controller_archive_2016-08-18_12-18.mat')
