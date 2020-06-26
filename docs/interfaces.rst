.. _sec-interfaces:

==========
Interfaces
==========

Currently M-LOOP supports three ways to interface your experiment

1. File interface where parameters and costs are exchanged between the experiment and M-LOOP through files written to disk. This approach is described in a :ref:`tutorial <sec-standard-experiment>`. 
2. Shell interface where parameters and costs are exchanged between the experiment and M-LOOP through information piped through a shell (or command line). This option should be considered if you can execute your experiment using a command from a shell. 
3. Implementing your own interface through the M-LOOP python API.

Each of these options is described below. If you have any suggestions for interfaces please consider :ref:`sec-contributing` to the project.

File interface
==============

The simplest method to connect your experiment to M-LOOP is with the file interface where data is exchanged by writing files to disk. To use this interface you can include the option::

   interface='file'
   
in your configuration file. The file interface happens to be the default, so this is not necessary.

.. tutorials-interface-include-start

The file interface works under the assumption that your experiment follows the following algorithm.

1. Wait for the file *exp_input.txt* to be made on the disk in the same folder in which M-LOOP is run.
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

.. tutorials-interface-include-end

Shell interface
===============

The shell interface is used when experiments can be run from a command in a shell. M-LOOP will still need to be configured and executed in the same manner described for a file interface as describe in :ref:`tutorial <sec-standard-experiment>`. The only difference is how M-LOOP starts the experiment and reads data. To use this interface you must include the following options::

	interface_type='shell'
	command='./run_exp'
	params_args_type='direct'
	
in the configuration file. The interface keyword simply indicates that you want M-LOOP to operate the experiment through the shell. The other two keywords need to be customized to your needs.

The command keyword should be provided with the command on the shell that runs the experiment.
In the example above the executable would be *run_exp*. Note M-LOOP will try and execute the command in the folder that you run M-LOOP from.
If this causes trouble you should just include the absolute address of your executable.
Your command can be more complicated than a single word, for example if you want to include some options like './run_exp --verbose -U' this would also be acceptable. 

The params_args_type keyword controls how M-LOOP delivers the parameters to the executable. If you use the 'direct' option the parameters will just be fed directly to the experiment as arguments. For example if the command was ./run_exp and the parameters to test next were 1.3, -23 and 12, M-LOOP would execute the following command::

	./run_exp 1.3 -23 12

The other ``params_args_type`` option is ``'named'``, in which case each parameter is fed to the experiment as a named option.
As of yet, M-LOOP does not use the values from ``param_names`` when calling the executable.
Instead the executable is passed arguments named ``param1``, ``param2`` and so on.
Given the same parameters as before, M-LOOP would execute the command::

	./run_exp --param1 1.3 --param2 -23 --param3 12
	
After the experiment has run and a cost (and uncertainty or bad value) has been found they must be provided back to M-LOOP through the shell. For example if you experiment completed with a cost 1.3, uncertainty 0.1 you need to program your executable to print the following to the shell::

	M-LOOP_start
	cost = 1.3
	uncer = 0.1
	M-LOOP_end

You can also output other information to the shell and split up the information you provide to M-LOOP if you wish. The following output would also valid.

	Running experiment... Experiment complete.
	Checking it was valid... It worked.
	M-LOOP_start
	bad = False
	M-LOOP_end
	Calculating cost... Was 3.2.
	M-LOOP_start
	cost = 3.2
	M-LOOP_end
	
Python interfaces 
=================

If your experiment is controlled in python you can use M-LOOP as an API in your own custom python script. In this case you must create your own implementation of the abstract interface class to control the experiment. This is explained in detail in the :ref:`tutorial for python controlled experiments <sec-python-experiment>`.
