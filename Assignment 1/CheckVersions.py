"""
File name: CheckVersions.py
Author: Naran Bat
Description: Check library versions
Inputs: None
Outputs: 
    - Library versions
Date created: 2025-08-26
""" 
import sys
import scipy
import numpy
import pandas
import sklearn

# print versions
print("python:", sys.version)
print("scipy:", scipy.__version__)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)

# hello world
print("hello world!")
