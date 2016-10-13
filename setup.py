'''
Setup script for M-LOOP using setuptools. See the documentation of setuptools for further details. 
'''
from __future__ import absolute_import, division, print_function

import multiprocessing as mp
import mloop as ml

from setuptools import setup, find_packages

def main():
    setup(
        name = 'M-LOOP',
        version = ml.__version__,
        packages = find_packages(),
        entry_points={
            'console_scripts': [
                'M-LOOP = mloop.cmd:run_mloop'
            ],
        },
        
        setup_requires=['pytest-runner'],
        install_requires = ['pip>=7.0'
                            'docutils>=0.3',
                            'numpy>=1.11',
                            'scipy>=0.17',
                            'matplotlib>=1.5',
                            'pytest>=2.9',
                            'scikit-learn>=0.18'],
        tests_require=['pytest','setuptools>=26'],
        
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
        download_url = 'https://github.com/michaelhush/M-LOOP/tarball/v2.1.0',
    
        classifiers = ['Development Status :: 2 - Pre-Alpha',
                      'Intended Audience :: Science/Research',
                      'Intended Audience :: Manufacturing',
                      'License :: OSI Approved :: MIT License',
                      'Natural Language :: English',
                      'Operating System :: MacOS :: MacOS X',
                      'Operating System :: POSIX :: Linux',
                      'Operating System :: Microsoft :: Windows',
    				  'Programming Language :: Python :: 2.7',
    				  'Programming Language :: Python :: 3.4',
    				  'Programming Language :: Python :: 3.5',
                      'Programming Language :: Python :: Implementation :: CPython',
                      'Topic :: Scientific/Engineering',
                      'Topic :: Scientific/Engineering :: Artificial Intelligence',
                      'Topic :: Scientific/Engineering :: Physics']
    )

if __name__=='__main__':
    mp.freeze_support()
    main()  