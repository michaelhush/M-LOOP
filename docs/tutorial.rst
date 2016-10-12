.. _sec-tutorial:

Tutorials
=========

Here we provide some tutorials on how to use M-LOOP. M-LOOP is flexible and can be customized with a variety of :ref:`options <sec-examples>` and :ref:`sec-interfaces`. Here we provide some basic tutorials to get you up and started as quick as possible.

There are two different approaches to using M-LOOP:

1. You can execute M-LOOP from a command line (or shell) and configure it using a text file. 
2. You can use M-LOOP as a :ref:`python API <sec-api>`.

If you have a standard experiment, that is operated by LabVIEW, Simulink or some other method, then your should use option 1 and follow the :ref:` first tutorial <sec-standard-experiment>`. If your experiment is operated using python, you should consider using option 2 as it will give you more flexibility. In which case, look at the :ref:`second tutorial <sec-python-experiment>`.


.. _sec-standard-experiment:

Standard experiment
===================

Overview
--------

The basic operation of M-LOOP is sketched below.

.. _fig-mloop-diag:

.. figure:: _static/M-LOOP_diagram.png
   :alt: M-LOOP in a loop with an experiment sending parameters and receiving costs.
   
There are three stages: 

1. M-LOOP is started with the command::

      M-LOOP 

   M-LOOP first looks for the configuration file *exp_input.txt*, which contains options like the number of parameters and their limits, in the folder it is executed, then starts the optimization process. 

2. M-LOOP controls and optimizes the experiment by exchanging files written to disk. M-LOOP produces a file called *exp_input.txt* which contains a variable params with the next parameters to be run by the experiment. The experiment is expected to run an experiment with these parameters and measure the resultant cost. The experiment should then write the file *exp_output.txt* which contains at least the variable cost which quantifies the performance of that experimental run, and optionally, the variables uncer (for uncertainty) and bad (if the run failed). This process is repeated many times until the halting condition is met.

3. Once the optimization process is complete, M-LOOP prints to the console the parameters and cost of the best run performed during the experiment, and a prediction of what the optimal parameters (with the corresponding predicted cost and uncertainty). M-LOOP also produces a set of plots that allow the user to visualize the optimization process and cost landscape. During operation and at the end M-LOOP write three files to disk: 

   - *M-LOOP_[datetime].log* a log of the console output and other debugging information during the run.
   - *controller_archive_[datetime].txt* an archive of all the experimental data recorded and the results.
   - *learner_archive_[datetime].txt* an archive of the model created by the machine learner of the experiment.

In what follows we will unpack this process and give details on how to configure and run M-LOOP.

Launching M-LOOP
----------------

Launching M-LOOP is performed by executing the command M-LOOP on the console. You can also provide the name of your configuration file if you do not want to use the default with the command::

   M-LOOP -c [config_filename]

Configuration File
------------------

The configuration file contains a list of options and settings for the optimization run. Each option must be started on a new line and formatted as::

   [keyword] = [value]
   
You can add comments to your file using #, everything past # will be ignored. Examples of relevant keywords and syntax for the values is provided in :ref:`sec-examples` and a comprehensive list of options is described in :ref:`sec-examples`. The values should be formatted with python syntax, strings should be surrounded with single or double quotes and arrays of values can be surrounded with square brackets/parentheses with numbers separated with commas. In this tutorial we will examine the example file *tutoral_config.txt*::

   #Tutorial Config
   #---------------

   #Parameter settings
   num_params = 2                #number of parameters
   min_boundary = [-1,-1]        #minimum boundary
   max_boundary = [1,1]          #maximum boundary

   #Halting conditions
   max_num_runs = 1000                       #maximum number of runs
   max_num_runs_without_better_params = 50   #maximum number of runs without finding better parameters
   target_cost = 0.01                        #optimization halts when a cost below this target is found 

   #Learner specific options
   first_params = [0.5,0.5]   #first parameters to try
   trust_region = 0.4         #maximum % move distance from best params

   #File format options
   interface_file_type = 'txt'            #file types of *exp_input.mat* and *exp_output.mat*
   controller_archive_file_type = 'mat'   #file type of the controller archive
   learner_archive_file_type = 'pkl'      #file type of the learner archive

   #Visualizations
   visualizations = True

We will now explain the options in each of their groups. In almost all cases you will only need to the parameters settings and halting conditions, but we have also describe a few of the most commonly used extra options. 

Parameter settings
~~~~~~~~~~~~~~~~~~

The number of parameters and their limits is defined with three keywords::

   num_params = 2
   min_boundary = [-1,-1]
   max_boundary = [1,1] 

num_params defines the number of parameters, min_boundary defines the minimum value each of the parameters can take and max_boundary defines the maximum value each parameter can take. Here there are two value which each must be between -1 and 1.

Halting conditions
~~~~~~~~~~~~~~~~~~

The halting conditions define when the simulation will stop. We present three options here::

   max_num_runs = 100                        
   max_num_runs_without_better_params = 10   
   target_cost = 0.1

max_num_runs is the maximum number of runs that the optimization algorithm is allowed to run. max_num_runs_without_better_params is the maximum number of runs allowed before a lower cost and better parameters is found. Finally, when target_cost is set, if a run produces a cost that is less than this value the optimization process will stop.

When multiple halting conditions are set, the optimization process will halt when any one of them is met. 

If you do not have any prior knowledge of the problem use only the keyword max_num_runs and set it to the highest value you can wait for. If you have some knowledge about what the minimum attainable cost is or there is some cost threshold you need to achieve, you might want to set the target_cost. max_num_runs_without_better_params is useful if you want to let the optimization algorithm run as long as it needs until there is a good chance the global optimum has been found. 

If you do not want one of the halting conditions, simply delete it from your file. For example if you just wanted the algorithm to search as long as it can until it found a global minimum you could set::

   max_num_runs_without_better_params = 10 


Learner specific options
~~~~~~~~~~~~~~~~~~~~~~~~

There are many learner specific options (and different learner algorithms) described in :ref:`sec-examples`. Here we consider just a couple of the most commonly used ones. M-LOOP has been designed to find an optimum quickly with no custom configuration as long as the experiment is able to provide a cost for every parameter it provides.

However if your experiment will fail to work if there are sudden and significant changes to your parameters you may need to set the following options::

   first_parameters = [0.5,0.5]      
   trust_region = 0.4            

first_parameters defines the first parameters the learner will try. trust_region defines the maximum change allowed in the parameters from the best parameters found so far. In the current example the region size is 2 by 2, with a trust region of 40% thus the maximum allowed change for the second run will be [0 +/- 0.8, 0 +/- 0.8].

If you experiment reliably produces costs for any parameter set you will not need these settings and you can just delete them.

File format options
~~~~~~~~~~~~~~~~~~~

You can set the file formats for the archives produced at the end and the files exchanged with the experiment with the options::

   interface_file_type = 'txt'          
   controller_archive_file_type = 'mat'  
   learner_archive_file_type = 'pkl' 

interface_file_type controls the file format for the files exchanged with the experiment. controller_archive_file_type and learner_archive_file_type control the format of the respective archives.  

There are three file formats currently available: 'mat' is for MATLAB readable files, 'pkl' if for python binary archives created using the `pickle package <https://docs.python.org/3/library/pickle.html>`_, and 'txt' human readable text files. For more details on these formats see :ref:`sec-data`.

Visualization
~~~~~~~~~~~~~

By default M-LOOP will display a set of plots that allow the user to visualize the optimization process and the cost landscape. To change this behavior use the option::

   visualizations = True
   
Set it to false to turn the visualizations off. For more details see :ref:`sec-visualizations`.

Interface
---------

There are many options of how to connect M-LOOP to your experiment. We consider the most generic method, writing and reading files to disk. For other options see :ref:`sec-interfaces`. If you design a bespoke interface for your experiment please consider :ref:`sec-contributing` to the project by sharing your method with other users.

The file interface works under the assumption that you experiment follows the following algorithm.

1. Wait for the file *exp_input.txt* to be made on the disk in the same folder M-LOOP is run.
2. Read the parameters for the next experiment from the file (named params).
3. Delete the file  *exp_input.txt*.
4. Run the experiment with the parameters provided and calculate a cost, and optionally the uncertainty.
5. Write the cost to the file *exp_output.txt*. Go back to step 1.

It is important you delete the file *exp_input.txt* after reading it, since it is used to as an indicator for the next experiment to run.

When writing the file *exp_output.txt* there are three keywords and values you can include in your file, for example after the first run your experiment may produce the following::

   cost = 0.5
   uncer = 0.01
   bad = false

cost refers to the cost calculated from the experimental data. uncer, is optional, and refers to the uncertainty in the cost measurement made. Note, M-LOOP by default assumes there is some noise corrupting costs, which is fitted and compensated for. Hence, if there is some noise in your costs which you are unable to predict from a single measurement, do not worry, you do not have to estimate uncer, you can just leave it out. Lastly bad can be used to indicate an experiment failed and was not able to produce a cost. If the experiment worked set bad = false and if it failed set bad = true.

Note you do not have to include all of the keywords, you must provide at least a cost or the bad keyword set to false. For example a successful run can simply be::

   cost = 0.3
   
and failed experiment can be as simple as::

   bad = True
   
Once the *exp_output.txt* has been written to disk, M-LOOP will read it and delete it. 
   
Parameters and cost function
----------------------------

Choosing the right parameterization of your experiment and cost function will be an important part of getting great results. 

If you have time dependent functions in your experiment you will need to choose a parametrization of these function before interfacing them with M-LOOP. M-LOOP will take more time and experiments to find an optimum, given more parameters. But if you provide too few parameters, you may not be able to achieve your cost target.

Fortunately, the visualizations provided after the optimization will help you determine which parameters contributed the most to the optimization process. Try with whatever parameterization is convenient to start and use the data produced afterwards to guide you on how to better improve the parametrization of your experiment. 

Picking the right cost function from experimental observables will also be important. M-LOOP will always find a global optimal as quick as it can, but if you have a poorly chosen cost function, the global optimal may not what you really wanted to optimize. Make sure you pick a cost function that will uniquely produce the result you want. Again, do not be afraid to experiment and use the data produced by the optimization runs to improve the cost function you are using.

Have a look at our `paper <http://www.nature.com/articles/srep25890>`_ on using M-LOOP to create a Bose-Einstein Condensate for an example of choosing a parametrization and cost function for an experiment.

Results
-------

Once M-LOOP has completed the optimization, it will output results in several ways.

M-LOOP will print results to the console. It will give the parameters of the experimental run that produced the lowest cost. It will also provide a set of parameters which are predicted to be produce the lowest average cost. If there is no noise in the costs your experiment produced, then the best parameters and predicted best parameters will be the same. If there was some noise your costs then it is possible that there will be a difference between the two. This is because the noise might have resulted with a set of experimental parameters that produced a lower cost due to a random fluke. The real optimal parameters that correspond to the minimum average cost are the predicted best parameters. In general, use the predicted best parameters (when provided) as the final result of the experiment. 

M-LOOP will produce an archive for the controller and machine learner. The controller archive contains all the data gathered during the experimental run and also other configuration details set by the user. By default it will be a 'txt' file which is human readable. If the meaning of a keyword and its associated data in the file is unclear, just :ref:`search` the documentation with the keyword to find a description. The learner archive contains a model of the experiment produced by the machine learner algorithm, which is currently a gaussian process. By default it will also be a 'txt' file. For more detail on these files see :ref:`sec-data`.

M-LOOP, by default, will produce a set of visualizations. These plots show the optimizations process over time and also predictions made by the learner of the cost landscape. For more details on these visualizations and their interpretation see :ref:`sec-visualizations`.

.. _sec-python-experiment:

Python controlled experiment 
============================


