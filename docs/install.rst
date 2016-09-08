.. _sec-installation:

Installation
============
M-LOOP is available on PyPI and can be installed with your favorite package manager. However, we currently recommend you install from the source code to ensure you have the latest improvements and bug fixes. 

The installation process involves three steps.

1. Get a Python 3 distribution with the standard scientific packages. We recommend installing :ref:`sec-anaconda`.
2. Install the development version of :ref:`sec-M-LOOP`.
3. :ref:`Test<sec-Testing>` your M-LOOP install.

.. _sec-anaconda:

Anaconda
--------
We recommend installing Anaconda to get a python 3 environment with all the required scientific packages. The Anaconda distribution is available here:

https://www.continuum.io/downloads

Follow the installation instructions they provide.

M-LOOP requires a python 3.\* environment. If you want to retain compatibility with python 2.\* Anaconda supports installing multiple python environments on the same machine, see:

http://conda.pydata.org/docs/py2or3.html#create-python-2-or-3-environments. 

.. _sec-m-loop:

M-LOOP
------
M-LOOP can be installed from the source code with three commands::

   git clone git://github.com/michaelhush/M-LOOP.git
   cd ./M-LOOP
   python setup.py develop

The first command downloads the latest source code for M-LOOP from github into the current directory, the second moves into the M-LOOP source directory, and the third link builds the package and creates a link from you python package to the source. You may need admin privileges to run the setup script.

At any time you can update M-LOOP to the latest version from github by running the command::

   git pull origin master

in the M-LOOP directory. 

.. _sec-Testing:

Test Installation
-----------------

To test your M-LOOP installation use the command::

   python setup.py test
   
In the M-LOOP source code directory. The tests should take around five minutes to complete. If you find a error please consider :ref:`sec-contributing` to the project and report a bug on the `github <https://github.com/michaelhush/M-LOOP>`_.

Documentation
-------------

If you would also like a local copy of the documentation enter the docs folder and use the command::

   make html
   
Which will generate the documentation in docs/_build/html.

