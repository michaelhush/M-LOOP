.. _sec-tutorial:

=========
Tutorials
=========

Here we provide some tutorials on how to use M-LOOP.
M-LOOP is flexible and can be customized with a variety of :ref:`options <sec-examples>` and :ref:`interfaces <sec-interfaces>`.
Here we provide some basic tutorials to get you started as quickly as possible.

There are two different approaches to using M-LOOP:

1. You can execute M-LOOP from a command line (or shell) and configure it using a text file. 
2. You can use M-LOOP as a :ref:`python API <sec-api>`.

If you have a standard experiment that is operated by LabVIEW, Simulink or some other method, then you should use option 1 and follow the :ref:`first tutorial <sec-standard-experiment>`.
If your experiment is operated using python, you should consider using option 2 as it will give you more flexibility and control, in which case, look at the :ref:`second tutorial <sec-python-experiment>`.

.. _sec-standard-experiment:

Standard experiment
===================

The basic operation of M-LOOP is sketched below.

.. _fig-mloop-diag:

.. figure:: _static/M-LOOP_diagram.png
   :alt: M-LOOP in a loop with an experiment sending parameters and receiving costs.
   
There are three stages: 

1. M-LOOP is started with the command::

      M-LOOP 

   M-LOOP first looks for the configuration file *exp_config.txt*, which contains options like the number of parameters and their limits, in the folder in which it is executed.
   Then it starts the optimization process. 

2. M-LOOP controls and optimizes the experiment by exchanging files written to disk.
   M-LOOP produces a file called *exp_input.txt* which contains a variable params with the next parameters to be run by the experiment. The experiment is expected to run an experiment with these parameters and measure the resultant cost.
   The experiment should then write the file *exp_output.txt* which contains at least the variable cost which quantifies the performance of that experimental run, and optionally, the variables uncer (for uncertainty) and bad (if the run failed).
   This process is repeated many times until a halting condition is met.

3. Once the optimization process is complete, M-LOOP prints to the console the parameters and cost of the best run performed during the experiment, and a prediction of what the optimal parameters are (with the corresponding predicted cost and uncertainty).
   M-LOOP also produces a set of plots that allow the user to visualize the optimization process and cost landscape.
   During operation and at the end M-LOOP writes these files to disk: 

   - *M-LOOP_[datetime].log* a log of the console output and other debugging information during the run.
   - *controller_archive_[datetime].txt* an archive of all the experimental data recorded and the results.
   - *learner_archive_[datetime].txt* an archive of the model created by the machine learner of the experiment.
   - If using the neural net learner, then several *neural_net_archive* files will be saved which store the fitted neural nets.

In what follows we will unpack this process and give details on how to configure and run M-LOOP.

Launching M-LOOP
----------------

Launching M-LOOP is performed by executing the command M-LOOP on the console. You can also provide the name of your configuration file if you do not want to use the default with the command::

   M-LOOP -c [config_filename]

.. _sec-configuration-file:
   
Configuration File
------------------

The configuration file contains a list of options and settings for the optimization run. Each option must be started on a new line and formatted as::

   [keyword] = [value]
   
You can add comments to your file using #.
Everything past the # will be ignored.
Examples of relevant keywords and syntax for the values are provided in :ref:`sec-examples` and a comprehensive list of options are described in :ref:`sec-examples`.
The values should be formatted with python syntax.
Strings should be surrounded with single or double quotes and arrays of values can be surrounded with square brackets/parentheses with numbers separated by commas.
In this tutorial we will examine the example file *tutorial_config.txt*

.. include:: ../examples/tutorial_config.txt
   :literal:

We will now explain the options in each of their groups.
In almost all cases you will only need to adjust the parameters settings and halting conditions, but we have also described a few of the most commonly used extra options. 

Parameter settings
~~~~~~~~~~~~~~~~~~

The number of parameters and constraints on what parameters can be tried are defined with a few keywords:

.. include:: ../examples/tutorial_config.txt
   :literal:
   :start-after: #Parameter settings
   :end-before: #Halting conditions

num_params defines the number of parameters, min_boundary defines the minimum value each of the parameters can take and max_boundary defines the maximum value each parameter can take. Here there are two value which each must be between -1 and 1.

first_params defines the first parameters the learner will try.
You only need to set this if you have a safe set of parameters you want the experiment to start with.
Just delete this keyword if any set of parameters in the boundaries will work.

trust_region defines the maximum change allowed in the parameters from the best parameters found so far.
In the current example the region size is 2 by 2, with a trust region of 40% .
Thus the maximum allowed change for the second run will be [0 +/- 0.8, 0 +/- 0.8].
Alternatively you can provide a list of values for trust_region, which should have one entry for each parameter.
In that case each entry specifies the maximum change for the corresponding parameter.
When specified as a list, the elements are interpreted as the absolute amplitude of the change, *not* the change as a fraction of the allowed range.
Setting trust_region to [0.4, 0.4] would make the maximum allowed change for the second run be [0 +/- 0.4, 0 +/- 0.4].
Generally, specifying the trust_region is only needed if your experiment produces bad results when the parameters are changed significantly between runs.
Simply delete this keyword if your experiment works with any set of parameters within the boundaries.

Halting conditions
~~~~~~~~~~~~~~~~~~

The halting conditions define when the optimization will stop. We present three options here:

.. include:: ../examples/tutorial_config.txt
   :literal:
   :start-after: #Halting conditions
   :end-before: #Learner options

max_num_runs is the maximum number of runs that the optimization algorithm is allowed to run. max_num_runs_without_better_params is the maximum number of runs allowed before a lower cost and better parameters is found. Finally, when target_cost is set, if a run produces a cost that is less than this value the optimization process will stop.

When multiple halting conditions are set, the optimization process will halt when any one of them is met. 

If you do not have any prior knowledge of the problem use only the keyword max_num_runs and set it to the highest value you can wait for. If you have some knowledge about what the minimum attainable cost is or there is some cost threshold you need to achieve, you might want to set the target_cost. max_num_runs_without_better_params is useful if you want to let the optimization algorithm run as long as it needs until there is a good chance the global optimum has been found. 

If you do not want one of the halting conditions, simply delete it from your file. For example if you just wanted the algorithm to search as long as it can until it found a global minimum you could set:

.. include:: ../examples/tutorial_config.txt
   :literal:
   :start-after: #maximum number of runs
   :end-before: target_cost

Learner Options
~~~~~~~~~~~~~~~

There are many learner specific options (and different learner algorithms) described in :ref:`sec-examples`. Here we just present a common one:

.. include:: ../examples/tutorial_config.txt
   :literal:
   :start-after: #Learner options
   :end-before: #Timing options

If the cost you provide has noise in it, meaning the cost you calculate would fluctuate if you did multiple experiments with the same parameters, then set this flag to True.
If the costs you provide have no noise then set this flag to False.
M-LOOP will automatically determine if the costs have noise in them or not, so if you are unsure, just delete this keyword and it will use the default value of True. 

Timing options
~~~~~~~~~~~~~~

M-LOOP's default optimization algorithm learns how the experiment works by fitting the parameters and costs using a gaussian process.
This learning process can take some time. If M-LOOP is asked for new parameters before it has time to generate a new prediction, it will use the training algorithm to provide a new set of parameters to test.
This allows for an experiment to be run while the learner is still thinking.
The training algorithm by default is differential evolution.
This algorithm is also used to do the first initial set of experiments which are then used to train M-LOOP.
If you would prefer M-LOOP waits for the learner to come up with its best prediction before running another experiment you can change this behavior with the option:

.. include:: ../examples/tutorial_config.txt
   :literal:
   :start-after: #Timing options
   :end-before: #File format options
   
Set no_delay to true to ensure that there are no pauses between experiments and set it to false if you want to give M-LOOP the time to come up with its most informed choice.
Sometimes doing fewer more intelligent experiments will lead to an optimum quicker than many quick unintelligent experiments.
You can delete the keyword if you are unsure and it will default to True.  

File format options
~~~~~~~~~~~~~~~~~~~

You can set the file formats for the archives produced at the end and the files exchanged with the experiment with the options:

.. include:: ../examples/tutorial_config.txt
   :literal:
   :start-after: #File format options
   :end-before: #Visualizations

interface_file_type controls the file format for the files exchanged with the experiment. controller_archive_file_type and learner_archive_file_type control the format of the respective archives.  

There are three file formats currently available: 'mat' is for MATLAB readable files, 'pkl' if for python binary archives created using the `pickle package <https://docs.python.org/3/library/pickle.html>`_, and 'txt' human readable text files. For more details on these formats see :ref:`sec-data`.

Visualization
~~~~~~~~~~~~~

By default M-LOOP will display a set of plots that allow the user to visualize the optimization process and the cost landscape. To change this behavior use the option:

.. include:: ../examples/tutorial_config.txt
   :literal:
   :start-after: #Visualizations
   
Set it to false to turn the visualizations off. For more details see :ref:`sec-visualizations`.

Interface
---------

There are many options for how to connect M-LOOP to your experiment.
Here we consider the most generic method, writing and reading files to disk.
For other options see :ref:`sec-interfaces`.
If you design a bespoke interface for your experiment please consider :ref:`sec-contributing` to the project by sharing your method with other users.

.. include:: ./interfaces.rst
   :start-after: .. tutorials-interface-include-start
   :end-before: .. tutorials-interface-include-end
   
Parameters and cost function
----------------------------

Choosing the right parameterization of your experiment and cost function will be an important part of getting great results. 

If you have time dependent functions in your experiment you will need to choose a parametrization of these function before interfacing them with M-LOOP. M-LOOP will take more time and experiments to find an optimum, given more parameters. But if you provide too few parameters, you may not be able to achieve your cost target.

Fortunately, the visualizations provided after the optimization will help you determine which parameters contributed the most to the optimization process. Try with whatever parameterization is convenient to start and use the data produced afterwards to guide you on how to better improve the parametrization of your experiment. 

Picking the right cost function from experimental observables will also be important.
M-LOOP will always find a global optimum as quickly as it can, but if you have a poorly chosen cost function, the global optimum may not be what you really wanted.
Make sure you pick a cost function that will uniquely produce the result you want.
Again, do not be afraid to experiment and use the data produced by the optimization runs to improve the cost function you are using.

Have a look at our `paper <http://www.nature.com/articles/srep25890>`_ on using M-LOOP to create a Bose-Einstein Condensate for an example of choosing a parametrization and cost function for an experiment.

.. _sec-results:

Results
-------

Once M-LOOP has completed the optimization, it will output results in several ways.

M-LOOP will print results to the console.
It will give the parameters of the experimental run that produced the lowest cost.
It will also provide a set of parameters which are predicted to produce the lowest average cost.
If there is no noise in the costs your experiment produced, then the best parameters and predicted best parameters will be the same.
If there was some noise in your costs then it is possible that there will be a difference between the two.
This is because the noise might have caused a set of experimental parameters to produce a lower cost than they typically would due to a random fluke.
The real optimal parameters that correspond to the minimum average cost are the predicted best parameters.
In general, use the predicted best parameters (when provided) as the final result of the experiment. 

M-LOOP will produce an archive for the controller and machine learner.
The controller archive contains all the data gathered during the experimental run and also other configuration details set by the user.
By default it will be a 'txt' file which is human readable.
If the meaning of a keyword and its associated data in the file is unclear, just :ref:`search` the documentation with the keyword to find a description.
The learner archive contains a model of the experiment produced by the machine learner algorithm, which is currently a gaussian process by default.
By default it will also be a 'txt' file.
For more detail on these files see :ref:`sec-data`.

M-LOOP, by default, will produce a set of visualizations. These plots show the optimizations process over time and also predictions made by the learner of the cost landscape. For more details on these visualizations and their interpretation see :ref:`sec-visualizations`.

.. _sec-python-experiment:

Python controlled experiment 
============================

If you have an experiment that is already under python control you can use M-LOOP as an API.
Below we go over the example python script *python_controlled_experiment.py*.
You should also read over the :ref:`first tutorial <sec-standard-experiment>` to get a general idea of how M-LOOP works.

When integrating M-LOOP into your laboratory remember that it will be controlling your experiment, not vice versa.
Hence, at the top level of your python script you will execute M-LOOP which will then call on your experiment when needed.
Your experiment will not be making calls of M-LOOP.

An example script for a python controlled experiment is given in the examples folder called *python_controlled_experiment.py*, which is included below

.. literalinclude:: ../examples/python_controlled_experiment.py
   :language: python
   :linenos:

Each part of the code is explained in the following sections.
		
Imports
-------

The start of the script imports the libraries that are necessary for M-LOOP to work:

.. literalinclude:: ../examples/python_controlled_experiment.py
   :language: python
   :end-before: #Declare your custom class that inherits from the Interface class

The first group of imports are just for python 2 compatibility. M-LOOP is targeted at python3, but has been designed to be bilingual. These imports ensure backward compatibility.

The second group of imports are the most important modules M-LOOP needs to run. The interfaces and controllers modules are essential, while the visualizations module is only needed if you want to view your data afterwards.

Lastly, you can add any other imports you may need.

Custom Interface
----------------

M-LOOP takes an object oriented approach to controlling the experiment. This is different than the functional approach taken by other optimization packages, like scipy. When using M-LOOP you must make your own class that inherits from the Interface class in M-LOOP. This class must implement a method called *get_next_cost_dict* that takes a set of parameters, runs your experiment and then returns the appropriate cost and uncertainty. 

An example of the simplest implementation of a custom interface is provided below ::

	#Declare your custom class that inherits from the Interface class
	class SimpleInterface(mli.Interface):
		
		#the method that runs the experiment given a set of parameters and returns a cost
		def get_next_cost_dict(self,params_dict):
			
			#The parameters come in a dictionary and are provided in a numpy array
			params = params_dict['params']
			
			#Here you can include the code to run your experiment given a particular set of parameters
			#For this example we just evaluate a simple function
			cost = np.sum(params**2)
			uncer = 0
			bad = False
			
			#The cost, uncertainty and bad boolean must all be returned as a dictionary
			cost_dict = {'cost':cost, 'uncer':uncer, 'bad':bad}
			return cost_dict

The code above defines a new class that inherits from the Interface class in M-LOOP.
Note that this code is different from the example above; we will consider this later.
It is slightly more complicated than just defining a method, however there is a lot more flexibility when taking this approach.
You should put the code you use to run your experiment in the *get_next_cost_dict* method.
This method is executed by the interface whenever M-LOOP wants a cost corresponding to a set of parameters.

When you actually run M-LOOP you will need to make an instance of your interface. To make an instance of the class above you would use::
	
	interface = SimpleInterface()
	
This interface is then provided to the controller, which is discussed in the next section.

Dictionaries are used for both input and output of the method, to give the user flexibility. For example, if you had a bad run, you do not have to return a cost and uncertainty, you can just return a dictionary with bad set to True::

	cost_dict = {'bad':True}
	return cost_dict

By taking an object oriented approach, M-LOOP can provide a lot more flexibility when controlling your experiment. For example if you wish to start up your experiment or perform some initial numerical analysis you can add a customized constructor or __init__ method for the class. We consider this in the main example:

.. literalinclude:: ../examples/python_controlled_experiment.py
   :language: python
   :start-after: import time
   :end-before: def main():
    
In this code snippet we also implement a constructor with the *__init__()* method.
Here we just define a numpy array which defines the minimum_parameter values.
We can call this variable whenever we need in the *get_next_cost_dict method*.
You can also define your own custom methods in your interface or even inherit from other classes.  

Once you have implemented your own Interface running M-LOOP can be done in three lines.

Running M-LOOP
--------------

Once you have made your interface class, running M-LOOP can be as simple as three lines. In the example script M-LOOP is run in the main method:

.. literalinclude:: ../examples/python_controlled_experiment.py
   :language: python
   :start-after: return cost_dict
   :end-before: #The results of the optimization will be saved
		
In the code snippet we first make an instance of our custom interface class called interface. We then create an instance of a controller. The controller will run the experiment and perform the optimization. You must provide the controller with the interface and any of the M-LOOP options you would normally provide in the configuration file. In this case we give five options, which do the following:

1. *max_num_runs = 1000* sets the maximum number of runs to be 1000.
2. *target_cost = -2.99* sets a cost that M-LOOP will halt at once it has been reached.
3. *num_params = 3* sets the number of parameters to be 3.
4. *min_boundary = [-2,-2,-2]* defines the minimum values of each of the parameters.
5. *max_boundary = [2,2,2]* defines the maximum values of each of the parameters. 

There are many other options you can use. Have a look at :ref:`sec-configuration-file` for a detailed introduction into all the important configuration options. Remember you can include any option you would include in a configuration file as keywords for the controller. For more options you should look at all the config files in :ref:`sec-examples`, or for a comprehensive list look at the :ref:`sec-api`.

Once you have created your interface and controller you can run M-LOOP by calling the optimize method of the controller. So in summary M-LOOP is executed in three lines::

	interface = CustomInterface()
	controller = mlc.create_controller(interface, [options])
	controller.optimize()

Results
-------

The results will be displayed on the console and also saved in a set of files. Have a read over :ref:`sec-results` for more details on the results displayed and saved. Also read :ref:`sec-data` for more details on data formats and how it is stored.

Within the python environment you can also access the results as attributes of the controller after it has finished optimization. The example includes a simple demonstration of this:

.. literalinclude:: ../examples/python_controlled_experiment.py
   :language: python
   :start-after: controller.optimize()
   :end-before: #You can also run the default sets of visualizations

All of the results saved in the controller archive can be directly accessed as attributes of the controller object. For a comprehensive list of the attributes of the controller generated after an optimization run see the :ref:`sec-api`.

Visualizations
--------------

For each controller there is normally a default set of visualizations available.
The visualizations for the Gaussian Process, the default optimization algorithm, are described in :ref:`sec-visualizations`.
Visualizations can be called through the visualization module.
The example includes a simple demonstration of this:

.. literalinclude:: ../examples/python_controlled_experiment.py
   :language: python
   :start-after: print(controller.best_params)
   :end-before: #Ensures main is run when this code is run as a script

This code snippet will display all the visualizations available for that controller. There are many other visualization methods and options available that let you control which plots are displayed and when.
See the :ref:`sec-api` for details.
