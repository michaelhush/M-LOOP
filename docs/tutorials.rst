.. _sec-tutorial:

=========
Tutorials
=========

Here we provide some tutorials on how to use M-LOOP. M-LOOP is flexible and can be customized with a variety of :ref:`options <sec-examples>` and :ref:`interfaces <sec-interfaces>`. Here we provide some basic tutorials to get you up and started as quick as possible.

There are two different approaches to using M-LOOP:

1. You can execute M-LOOP from a command line (or shell) and configure it using a text file. 
2. You can use M-LOOP as a :ref:`python API <sec-api>`.

If you have a standard experiment, that is operated by LabVIEW, Simulink or some other method, then you should use option 1 and follow the :ref:`first tutorial <sec-standard-experiment>`. If your experiment is operated using python, you should consider using option 2 as it will give you more flexibility and control, in which case, look at the :ref:`second tutorial <sec-python-experiment>`.

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

.. _sec-configuration-file:
   
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

Note you do not have to include all of the keywords, you must provide at least a cost or the bad keyword set to true. For example a successful run can simply be::

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

.. _sec-results:

Results
-------

Once M-LOOP has completed the optimization, it will output results in several ways.

M-LOOP will print results to the console. It will give the parameters of the experimental run that produced the lowest cost. It will also provide a set of parameters which are predicted to be produce the lowest average cost. If there is no noise in the costs your experiment produced, then the best parameters and predicted best parameters will be the same. If there was some noise your costs then it is possible that there will be a difference between the two. This is because the noise might have resulted with a set of experimental parameters that produced a lower cost due to a random fluke. The real optimal parameters that correspond to the minimum average cost are the predicted best parameters. In general, use the predicted best parameters (when provided) as the final result of the experiment. 

M-LOOP will produce an archive for the controller and machine learner. The controller archive contains all the data gathered during the experimental run and also other configuration details set by the user. By default it will be a 'txt' file which is human readable. If the meaning of a keyword and its associated data in the file is unclear, just :ref:`search` the documentation with the keyword to find a description. The learner archive contains a model of the experiment produced by the machine learner algorithm, which is currently a gaussian process. By default it will also be a 'txt' file. For more detail on these files see :ref:`sec-data`.

M-LOOP, by default, will produce a set of visualizations. These plots show the optimizations process over time and also predictions made by the learner of the cost landscape. For more details on these visualizations and their interpretation see :ref:`sec-visualizations`.

.. _sec-python-experiment:

Python controlled experiment 
============================

If you have an experiment that is already under python control you can use M-LOOP as an API. Below we go over the example python script *python_controlled_experiment.py* you should also read over the :ref:`first tutorial <sec-standard-experiment>` to get a general idea of how M-LOOP works.

When integrating M-LOOP into your laboratory remember that it will be controlling you experiment, not vice versa. Hence, at the top level of your python script you will execute M-LOOP which will then call on your experiment when needed. Your experiment will not be making calls of M-LOOP.

An example script for a python controlled experiment is given in the examples folder called *python_controlled_experiment.py*, which is copied below::

	#Imports for python 2 compatibility
	from __future__ import absolute_import, division, print_function
	__metaclass__ = type

	#Imports for M-LOOP
	import mloop.interfaces as mli
	import mloop.controllers as mlc
	import mloop.visualizations as mlv

	#Other imports
	import numpy as np
	import time
	
	#Declare your custom class that inherits from the Interface class
	class CustomInterface(mli.Interface):
		
		#Initialization of the interface, including this method is optional
		def __init__(self):
			#You must include the super command to call the parent class, Interface, constructor 
			super(CustomInterface,self).__init__()
			
			#Attributes of the interface can be added here
			#If you want to pre-calculate any variables etc. this is the place to do it
			#In this example we will just define the location of the minimum
			self.minimum_params = np.array([0,0.1,-0.1])
			
		#You must include the get_next_cost_dict method in your class
		#this method is called whenever M-LOOP wants to run an experiment
		def get_next_cost_dict(self,params_dict):
			
			#Get parameters from the provided dictionary
			params = params_dict['params']
			
			#Here you can include the code to run your experiment given a particular set of parameters
			#In this example we will just evaluate a sum of sinc functions
			cost = -np.sum(np.sinc(params - self.minimum_params))
			#There is no uncertainty in our result
			uncer = 0
			#The evaluation will always be a success
			bad = False
			#Add a small time delay to mimic a real experiment
			time.sleep(1)
			
			#The cost, uncertainty and bad boolean must all be returned as a dictionary
			#You can include other variables you want to record as well if you want
			cost_dict = {'cost':cost, 'uncer':uncer, 'bad':bad}
			return cost_dict
		
	def main():
		#M-LOOP can be run with three commands
		
		#First create your interface
		interface = CustomInterface()
		#Next create the controller, provide it with your controller and any options you want to set
		controller = mlc.create_controller(interface, max_num_runs = 1000, target_cost = -2.99, num_params = 3, min_boundary = [-2,-2,-2], max_boundary = [2,2,2])
		#To run M-LOOP and find the optimal parameters just use the controller method optimize
		controller.optimize()
		
		#The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
		print('Best parameters found:')
		print(controller.best_params)
		
		#You can also run the default sets of visualizations for the controller with one command
		mlv.show_all_default_visualizations(controller)
		

	#Ensures main is run when this code is run as a script
	if __name__ == '__main__':
		main()

Each part of the code is explained in the following sections.
		
Imports
-------

The start of the script imports the libraries that are necessary for M-LOOP to work::

	#Imports for python 2 compatibility
	from __future__ import absolute_import, division, print_function
	__metaclass__ = type

	#Imports for M-LOOP
	import mloop.interfaces as mli
	import mloop.controllers as mlc
	import mloop.visualizations as mlv

	#Other imports
	import numpy as np
	import time
	
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
			params = params_dict['params']pre-calculate
			
			#Here you can include the code to run your experiment given a particular set of parameters
			#For this example we just evaluate a simple function
			cost = np.sum(params**2)
			uncer = 0
			bad = False
			
			#The cost, uncertainty and bad boolean must all be returned as a dictionary
			cost_dict = {'cost':cost, 'uncer':uncer, 'bad':bad}
			return cost_dict

The code above defines a new class that inherits from the Interface class in M-LOOP. Note this code is different to the example above, we will consider this later. It is slightly more complicated than just defining a method, however there is a lot more flexibility when taking this approach. You should put the code you use to run your experiment in the *get_next_cost_dict* method. This method is executed by the interface whenever M-LOOP wants a cost corresponding to a set of parameters.

When you actually run M-LOOP you will need to make an instance of your interface. To make an instance of the class above you would use::
	
	interface = SimpleInterface()
	
This interface is then provided to the controller, which is discussed in the next section.

Dictionaries are used for both input and output of the method, to give the user flexibility. For example, if you had a bad run, you do not have to return a cost and uncertainty, you can just return a dictionary with bad set to True::

	cost_dict = {'bad':True}
	return cost_dict

By taking an object oriented approach, M-LOOP can provide a lot more flexibility when controlling your experiment. For example if you wish to start up your experiment or perform some initial numerical analysis you can add a customized constructor or __init__ method for the class. We consider this in the main example::

	class CustomInterface(mli.Interface):
    
		#Initialization of the interface, including this method is optional
		def __init__(self):
			#You must include the super command to call the parent class, Interface, constructor 
			super(CustomInterface,self).__init__()
			
			#Attributes of the interface can be added here
			#If you want to pre-calculate any variables etc. this is the place to do it
			#In this example we will just define the location of the minimum
			self.minimum_params = np.array([0,0.1,-0.1])
			
		#You must include the get_next_cost_dict method in your class
		#this method is called whenever M-LOOP wants to run an experiment
		def get_next_cost_dict(self,params_dict):
			
			#Get parameters from the provided dictionary
			params = params_dict['params']
			
			#Here you can include the code to run your experiment given a particular set of parameters
			#In this example we will just evaluate a sum of sinc functions
			cost = -np.sum(np.sinc(params - self.minimum_params))
			#There is no uncertainty in our result
			uncer = 0
			#The evaluation will always be a success
			bad = False
			#Add a small time delay to mimic a real experiment
			time.sleep(1)
			
			#The cost, uncertainty and bad boolean must all be returned as a dictionary
			#You can include other variables you want to record as well if you want
			cost_dict = {'cost':cost, 'uncer':uncer, 'bad':bad}
			return cost_dict
    
In this code snippet we also implement a constructor. Here we just define a numpy array which defines the minimum_parameter values. We can call this variable whenever we need in the *get_next_cost_dict method*. You can also define your own custom methods in your interface or even inherit from other classes.  

Once you have implemented your own Interface running M-LOOP can be done in three lines.

Running M-LOOP
--------------

Once you have made your interface class running M-LOOP can be as simple as three lines. In the example script M-LOOP is run in the main method::

	def main():
		#M-LOOP can be run with three commands
		
		#First create your interface
		interface = CustomInterface()
		#Next create the controller, provide it with your controller and any options you want to set
		controller = mlc.create_controller(interface, max_num_runs = 1000, target_cost = -2.99, num_params = 3, min_boundary = [-2,-2,-2], max_boundary = [2,2,2])
		#To run M-LOOP and find the optimal parameters just use the controller method optimize
		controller.optimize()
		
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

Within the python environment you can also access the results as attributes of the controller after it has finished optimization. The example includes a simple demonstration of this::

		#The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
		print('Best parameters found:')
		print(controller.best_params)

All of the results saved in the controller archive can be directly accessed as attributes of the controller object. For a comprehensive list of the attributes of the controller generated after an optimization run see the :ref:`sec-api`.

Visualizations
--------------

For each controller there is normally a default set of visualizations available. The visualizations for the Gaussian Process, the default optimization algorithm, is described in :ref:`sec-visualizations`. Visualizations can be called through the visualization module. The example includes a simple demonstration of this::

		#You can also run the default sets of visualizations for the controller with one command
		mlv.show_all_default_visualizations(controller)

This code snippet will display all the visualizations available for that controller. There are many other visualization methods and options available that let you control which plots are displayed and when, see the :ref:`sec-api` for details. 









