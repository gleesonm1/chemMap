#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path


this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'src', 'chemMap', '_version.py'), encoding='utf-8') as f:
    exec(f.read())

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="chemMap",
    version=__version__,
    author="Matthew Gleeson",
    author_email="gleesonm@berkeley.edu",
    description="chemMap",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gleesonm1/chemMap",
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required

    package_data={
        # Include all pickle files
        "": ["*.jl"],
    },
    install_requires=[
            'pandas',
            'numpy',
            'matplotlib',
            'scikit-learn',
            'scipy'
            ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
