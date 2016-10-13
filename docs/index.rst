======
M-LOOP
======

The Machine-Learner Online Optimization Package is designed to automatically and rapidly optimize the parameters of a scientific experiment or computer controller system. 

.. figure:: _static/M-LOOPandBEC.png
   :alt: M-LOOP optimizing a BEC.
   
   M-LOOP in control of an ultra-cold atom experiment. M-LOOP was able to find an optimal set of ramps to evaporatively cool a thermal gas and form a Bose-Einstein Condensate. 

Using M-LOOP is simple, once the parameters of your experiment is computer controlled, all you need to do is determine a cost function that quantifies the performance of an experiment after a single run. You can then hand over control of the experiment to M-LOOP which will find a global optimal set of parameters that minimize the cost function, by performing a few experiments and testing different parameters. M-LOOP uses machine-learning to predict how the parameters of the experiment relate to the cost, it uses this model to pick the next best parameters to test to find an optimum as quickly as possible. 

M-LOOP not only finds an optimal set of parameters for the experiment it also provides a model of how the parameters are related to the costs which can be used to improve the experiment. 

If you use M-LOOP please cite our publication where we first used the package to optimize the production of a Bose-Einstein Condensate:

Fast Machine-Learning Online Optimization of Ultra-Cold-Atom Experiments. *Scientific Reports* **6**, 25890 (2016). DOI: `Link 10.1038/srep25890 <http://dx.doi.org/10.1038/srep25890>`_

http://www.nature.com/articles/srep25890

Quick Start
===========

To get M-LOOP running follow the :ref:`sec-installation` instructions and :ref:`sec-tutorial`. 

Contents
========

.. toctree::
   
   install
   tutorials
   interfaces
   data
   visualizations
   examples
   contributing
   api/index
   
Indices
=======

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

