.. _sec-interfaces:

Interfaces
==========

Currently M-LOOP only support the File interface, which is also described in :ref:`sec-tutorial`. There will be more added very soon. If you have any suggestions for interfaces please consider :ref:`sec-contributing` to the project.

File Interface
--------------

The simplest method to connect your experiment to M-LOOP is with the file interface where data is exchanged by writing files to disk. To use this interface you can include the option::

   interface='file'
   
in you configuration file. The file interface happens to be the default, so this is not necessary. 

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

Note you do not have to include all of the keywords, you must provide at least a cost or the bad keyword set to false. For example a succesful run can simply be::

   cost = 0.3
   
and failed experiment can be as simple as::

   bad = True
   
Once the *exp_output.txt* has been written to disk, M-LOOP will read it and delete it. 
