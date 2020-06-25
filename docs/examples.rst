.. _sec-examples:

========
Examples
========

M-LOOP includes a series of example configuration files for each of the controllers and interfaces. The examples can be found in examples folder.  For some controllers there are two files, ones ending with *_basic_config* which includes the standard configuration options and *_complete_config* which include a comprehensive list of all the configuration options available.

The options available are also comprehensively documented in the :ref:`sec-api` as keywords for each of the classes. However, the quickest and easiest way to learn what options are available, if you are not familiar with python, is to just look at the provided examples.

Each of the example files is used when running tests of M-LOOP. So please copy and modify them elsewhere if you use them as a starting point for your configuration file. 

Interfaces
==========

There are currently two interfaces supported: 'file' and 'shell'. You can specify which interface you want with the option::

   interface_type = [name]

The default will be 'file'. The specific options for each of the interfaces are described below.

File Interface
--------------

The file interface exchanges information with the experiment by writing files to disk. You can change the names of the files used for the file interface and their type. The file interface options are described in *file_interface_config.txt*.

.. include:: ../examples/file_interface_config.txt
   :literal:

Shell Interface
---------------

The shell interface is for experiments that can be run through a command executed in a shell. Information is then piped between M-LOOP and the experiment through the shell. You can change the command to run the experiment and the way the parameters are formatted. The shell interface options are described in *shell_interface_config.txt*

.. include:: ../examples/shell_interface_config.txt
   :literal:

   
Controllers
===========

There are currently five controller types supported: 'gaussian_process', 'neural_net', 'differential_evolution', 'nelder_mead', and 'random'.
The default is 'gaussian_process'.
You can set which interface you want to use with the option::

   controller_type = [name]

Each of the controllers and their specific options are described below. There is also a set of common options shared by all controllers which is described in *controller_config.txt*. The common options include the parameter settings and the halting conditions.

.. include:: ../examples/controller_config.txt
   :literal:
   
Gaussian process
----------------

The Gaussian process controller is the default controller.
It uses a `Gaussian process <http://scikit-learn.org/dev/modules/gaussian_process.html>`_ to develop a model for how the parameters relate to the measured cost, effectively creating a model for how the experiment operates.
This model is then used when picking new points to test. 

There are two example files for the Gaussian-process controller: *gaussian_process_simple_config.txt* which contains the basic options.

.. include:: ../examples/gaussian_process_simple_config.txt
   :literal:
   
*gaussian_process_complete_config.txt* which contains a comprehensive list of options.

.. include:: ../examples/gaussian_process_complete_config.txt
   :literal:
   
Neural net
----------------

The neural net controller also uses a machine-learning-based algorithm.
It is similar to the Gaussian process controller in that it constructs a model of how the parameters relate to the cost and then uses that model for the optimization.
However instead of modeling with a Gaussian process, it works by modeling with a sampled neural net.

The neural net models aren't always as robust and reliable as the Gaussian process.
However, the time required to fit a Gaussian process scales as the cube of the number of data points, while the time to train a neural net only scales linearly.
Often the Gaussian process fitting can be prohibitively slow for long optimizations with many parameters, while the neural net training remains relatively fast.
That makes the neural net controller a good choice for high-dimensional optimizations.

There are two example files for the neural net controller: *neural_net_simple_config.txt* which contains the basic options.

.. include:: ../examples/neural_net_simple_config.txt
   :literal:
   
*neural_net_complete_config.txt* which contains a comprehensive list of options.

.. include:: ../examples/neural_net_complete_config.txt
   :literal:

Differential evolution
----------------------

The differential evolution (DE) controller uses a `DE algorithm <https://en.wikipedia.org/wiki/Differential_evolution>`_ for optimization. DE is a type of evolutionary algorithm, and is historically the most commonly used in automated optimization. DE will eventually find a global solution, however it can take many experiments before it does so. 

There are two example files for the differential evolution controller: *differential_evolution_simple_config.txt* which contains the basic options.

.. include:: ../examples/differential_evolution_simple_config.txt
   :literal:
   
*differential_evolution_complete_config.txt* which contains a comprehensive list of options.

.. include:: ../examples/differential_evolution_complete_config.txt
   :literal:
   
   
Nelder Mead
-----------

The Nelder Mead controller implements the `Nelder-Mead method <https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method>`_ for optimization. You can control the starting point and size of the initial simplex of the method with the configuration file.

There are two example files for the Nelder-Mead controller: *nelder_mead_simple_config.txt* which contains the basic options.

.. include:: ../examples/nelder_mead_simple_config.txt
   :literal:
   
*nelder_mead_complete_config.txt* which contains a comprehensive list of options.

.. include:: ../examples/nelder_mead_complete_config.txt
   :literal:
   
Random
------

The random optimization algorithm picks parameters randomly from a uniform distribution from within the parameter bounds or trust region. 

There are two example files for the random controller: *random_simple_config.txt* which contains the basic options.

.. include:: ../examples/random_simple_config.txt
   :literal:
   
*random_complete_config.txt* which contains a comprehensive list of options.

.. include:: ../examples/random_complete_config.txt
   :literal:
   
Logging
=======

You can control the filename of the logs and also the level which is reported to the file and the console. For more information see `logging levels <https://docs.python.org/3.6/library/logging.html#levels>`_. The logging options are described in *logging_config.txt*.

.. include:: ../examples/logging_config.txt
   :literal:

Extras
======

Extras refers to options related to post processing your data once the optimization is complete. Currently the only extra option is for visualizations. The extra options are described in *extras_config.txt*.

.. include:: ../examples/extras_config.txt
   :literal:


