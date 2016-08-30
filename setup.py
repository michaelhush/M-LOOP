'''
Setup script for M-LOOP using setuptools. See the documentation of setuptools for further details. 
'''

import mloop as ml
from setuptools import setup, find_packages
setup(
    name = 'M-LOOP',
    version = ml.__version__,
    packages = find_packages(),
    scripts = ['./bin/M-LOOP'],
    
    setup_requires=['pytest-runner'],
    install_requires = ['docutils>=0.3'],
    tests_require=['pytest'],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt','*.md'],
    },
    author = 'Michael R Hush',
    author_email = 'MichaelRHush@gmail.com',
    description = 'M-LOOP: Machine-learning online optimization package. A python package of automated optimization tools - enhanced with machine-learning - for quantum scientific experiments, computer controlled systems or other optimization tasks.',
    license = 'MIT',
    keywords = 'automated machine learning optimization optimisation science experiment quantum',
    url = 'https://github.com/michaelhush/M-LOOP/', 
    download_url = 'https://github.com/michaelhush/M-LOOP/tarball/v2.0.1',

    classifiers = ['Development Status :: 2 - Pre-Alpha',
                  'Intended Audience :: Science/Research',
                  'Intended Audience :: Manufacturing',
                  'License :: OSI Approved :: MIT License',
                  'Natural Language :: English',
                  'Operating System :: MacOS :: MacOS X',
                  'Operating System :: POSIX :: Linux',
                  'Operating System :: Microsoft :: Windows',
                  'Programming Language :: Python :: 3 :: Only',
                  'Programming Language :: Python :: Implementation :: CPython',
                  'Topic :: Scientific/Engineering',
                  'Topic :: Scientific/Engineering :: Artificial Intelligence',
                  'Topic :: Scientific/Engineering :: Physics']
)