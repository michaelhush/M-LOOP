.. _sec-installation:

============
Installation
============

M-LOOP is available on PyPI and can be installed with your favorite package manager; simply search for 'M-LOOP' and install. However, if you want the latest features and a local copy of the examples you should install M-LOOP using the source code from the `GitHub <https://github.com/michaelhush/M-LOOP>`_. Detailed installation instruction are provided below.

The installation process involves three steps.

1. Get a Python distribution with the standard scientific packages. We recommend installing :ref:`sec-anaconda`.
2. Install the latest release of :ref:`sec-M-LOOP`.
3. (Optional) :ref:`Test<sec-Testing>` your M-LOOP install.

If you are having any trouble with the installation you may need to check your the :ref:`package dependencies<sec-dependencies>` have been correctly installed.
If you are still having trouble, you can `submit an issue <https://github.com/michaelhush/M-LOOP/issues>`_ to the GitHub.

.. _sec-anaconda:

Anaconda
========

We recommend installing Anaconda to get a python environment with all the required scientific packages. The Anaconda distribution is available here:

https://www.anaconda.com/

Follow the installation instructions they provide.

M-LOOP is targeted at python 3 but also supports 2. Please use python 3 if you do not have a reason to use 2, see :ref:`sec-py3vpy2` for details.

.. _sec-m-loop:

M-LOOP
======

You have two options when installing M-LOOP, you can perform a basic installation of the last release with pip or you can install from source to get the latest features. We recommend installing from source so you can test your installation, see all the examples and get the most recent bug fixes.

Installing from source
----------------------

M-LOOP can be installed from the latest source code with three commands::

   git clone git://github.com/michaelhush/M-LOOP.git
   cd ./M-LOOP
   pip install -e .

The first command downloads the latest source code for M-LOOP from GitHub into the current directory, the second moves into the M-LOOP source directory, and the third command builds the package and creates a link from you python package to the source.
If you are using linux or MacOS you may need admin privileges to run the installation step.

At any time you can update M-LOOP to the latest version from GitHub by running the command::

   git pull origin master

in the M-LOOP directory. 

Installing with pip
-------------------

M-LOOP can be installed with pip with a single command::

   pip install M-LOOP
   
If you are using linux or MacOS you may need admin privileges to run the command. To update M-LOOP to the latest version use::

   pip install M-LOOP --upgrade


.. _sec-Testing:

Testing
=======

If you have installed from source, you can test your installation by running the command::

   pytest
   
In the M-LOOP source code directory. The tests should take around five minutes to complete. If you find a error please consider :ref:`sec-contributing` to the project and report a bug on the `GitHub <https://github.com/michaelhush/M-LOOP>`_.

If you installed M-LOOP using pip, you will not need to test your installation. 

.. _sec-dependencies:

Dependencies
============

M-LOOP requires the following packages to run correctly.

============   =======
Package        Version
============   =======
docutils       >=0.3
matplotlib     >=1.5
numpy          >=1.11
pip            >=7.0  
pytest         >=2.9
setuptools     >=26   
scikit-learn   >=0.18
scipy          >=0.17
tensorflow     >=1.1.0
============   =======  

These packages should be automatically installed by pip or the script setup.py when you install M-LOOP. The setup script itself requires pytest-runner.

However, if you are using Anaconda some packages that are managed by the conda command may not be correctly updated, even if your installation passes all the tests. In this case, you will have to update these packages manually. You can check what packages you have installed and their version with the command::

   conda list
   
To install a package that is missing, say for example pytest, use the command::

   conda install pytest
   
To update a package to the latest version, say for example scikit-learn, use the command::

   conda update scikit-learn

Once you install and update all the required packages with conda M-LOOP should run correctly. 

Documentation
=============

The latest documentation will always be available here online. If you would also like a local copy of the documentation, and you have downloaded the source code, enter the docs folder and use the command::

   make html
   
Which will generate the documentation in docs/_build/html.

.. _sec-py3vpy2:

Python 3 vs 2
=============

M-LOOP is developed in python 3 and it gets the best performance in this environment. This is primarily because other packages that M-LOOP uses, like numpy, run fastest in python 3. The tests typically take about 20% longer to complete in python 2 than 3.

If you have a specific reason to stay in a python 2 environment (you may use other packages which are not python 3 compatible) then you can still use M-LOOP without upgrading to 3. However, if you do not have a specific reason to stay with python 2, it is highly recommended you use the latest python 3 package.
