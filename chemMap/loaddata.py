## Python Mapping tool
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
from .elements import element_properties

def loadwt(path,Oxide,filter=None):
    """
    Load in quantitative wt% data.

    Parameters:
    ----------
    path: string, required
        path to the location of the data.

    oxide: list, required
        List of the elements (or oxides) to be extracted from the data folder.

    filter: bool, required
        If True, a 3-by-3 median filter will be applied to the data.

    Returns:
    ----------
    Data: dict
        Python dictionary containing a numpy array for each element or oxide loaded.

    """
    Data={}
    for ox in Oxide:
        Data[ox] = np.genfromtxt(path + ox + ' Wt%.csv', delimiter = ',')
        Data[ox] = np.nan_to_num(Data[ox], copy = False, nan = 0.0) # with 0s rather than NaNs
        if filter is not None:
            Data[ox] = nd.median_filter(Data[ox], size=3)

    return Data

def loadcnt(path,Oxide,filter=None):
    """
    Load in raw count data.

    Parameters:
    ----------
    path: string, required
        path to the location of the data.

    oxide: list, required
        List of the elements to be extracted from the data folder.

    filter: bool, required
        If True, a 3-by-3 median filter will be applied to the data.

    Returns:
    ----------
    Data: dict
        Python dictionary containing a numpy array for each element loaded.

    """
    Data={}
    for ox in Oxide:
        Data[ox] = np.genfromtxt(path + ox + ' K series.csv', delimiter = ',')
        Data[ox] = np.nan_to_num(Data[ox], copy = False, nan = 0.0) # with 0s rather than NaNs
        if filter is not None:
            Data[ox] = nd.median_filter(Data[ox], size=3)

    return Data

def calcOxides(Data, Oxide):
    """
    Convert element wt% data to oxide wt% data

    Parameters:
    ----------
    Data: dict, required
        Python dictionary containing a numpy array for each element of interest.

    Oxide: list, required
        List containing the elements that the user wishes to convert into oxide concentrations.

    Returns:
    ----------
    Quant_ox: dict
        Python dictionary with a numpy array for each oxide specified by the user.

    """
    Quant_ox = {}
    for ox in Oxide:
        if (element_properties[ox][3] > 1) and (element_properties[ox][4] > 1):
            Quant_ox[ox + str(element_properties[ox][3]) + 'O' + str(element_properties[ox][4])]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])/(element_properties[ox][2]*element_properties[ox][3])
        if (element_properties[ox][3] > 1) and (element_properties[ox][4] == 1):
            Quant_ox[ox + str(element_properties[ox][3]) + 'O']=Data[ox]*(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])/(element_properties[ox][2]*element_properties[ox][3])
        if (element_properties[ox][3] == 1) and (element_properties[ox][4] > 1):
            Quant_ox[ox + 'O' + str(element_properties[ox][4])]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])/(element_properties[ox][2]*element_properties[ox][3])
        if (element_properties[ox][3] == 1) and (element_properties[ox][4] == 1):
            Quant_ox[ox + 'O']=Data[ox]*(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])/(element_properties[ox][2]*element_properties[ox][3])

    return Quant_ox

def loadtransect(path, Oxide):
    """
    Load quantitative oxide data from an EDS linescan.

    Parameters:
    ----------
    path: string, required
        path to the location of the data.

    oxide: list, required
        List of the elements to be extracted from the data folder.

    Returns:
    ----------
    Data: pandas dataframe
        Python dataframe containing the linescan values and distances.

    """

    Data=pd.DataFrame()
    for ox in Oxide:
        df = pd.read_csv(path + ox + ' Ox%.csv')
        if (element_properties[ox][3] > 1) and (element_properties[ox][4] > 1):
            Data[ox + str(element_properties[ox][3]) + 'O' + str(element_properties[ox][4])] = df[ox + ' Ox%']
        if (element_properties[ox][3] > 1) and (element_properties[ox][4] == 1):
            Data[ox + str(element_properties[ox][3]) + 'O'] = df[ox + ' Ox%']
        if (element_properties[ox][3] == 1) and (element_properties[ox][4] > 1):
            Data[ox + 'O' + str(element_properties[ox][4])] = df[ox + ' Ox%']
        if (element_properties[ox][3] == 1) and (element_properties[ox][4] == 1):
            Data[ox +'O'] = df[ox + ' Ox%']

    Data['Distance'] = df['Distance (µm)']

    return Data
