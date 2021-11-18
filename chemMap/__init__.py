__version__ = "0.1"
__author__ = 'Matthew Gleeson'


import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import sys
import scipy
from scipy import stats
from scipy import ndimage as nd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster import hierarchy
from pylab import plot, ginput, show, axis
import random
from sklearn.ensemble import RandomForestClassifier
from scipy.ndimage import label
from scipy.ndimage import find_objects



# This has the functions to load count or wt% data
from chemMap.loaddata import *
# This contains necessary information about the elements of interest
from chemMap.elements import *
# This contains code required to perform cluster analysis and/or  machine learning classification
from chemMap.group import *
# This contains code required to generate a map of a sample
from chemMap.elementmap import *
# This contains code required to plot transects and separate crystal populations according to their size
from chemMap.crystalanalysis import *
# This contains code required calculate molar ratios from the available data
from chemMap.ratios import *

