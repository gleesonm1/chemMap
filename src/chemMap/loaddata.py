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
import os

def loadwt(path,filter=None):
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
    el_dirs = [d for d in os.listdir(path) if d.endswith(".csv")]

    Data={}
    for e in el_dirs:
        i = np.genfromtxt(path + e, delimiter = ',')
        for j in list(element_properties.keys()):
            if e.split(" ",1)[0] == j:
                e = j
        Data[e] = np.nan_to_num(i[:,0:-1], copy = False)
        if filter is not None:
            Data[e] = nd.median_filter(Data[e], size = 3)

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

def calcOxides(Data, Oxide, Copy = None):
    """
    Convert element wt% data to oxide wt% data

    Parameters:
    ----------
    Data: dict, required
        Python dictionary containing a numpy array for each element of interest.

    Oxide: list, required
        List containing the elements that the user wishes to convert into oxide concentrations.

    Copy: str, optional
        True or False. Default False.

    Returns:
    ----------
    Quant_ox: dict
        Python dictionary with a numpy array for each oxide specified by the user.

    """
    if Copy is None:
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

    if Copy is not None:
        Quant_ox = Data.copy()
        for ox in Oxide:
            if (element_properties[ox][3] > 1) and (element_properties[ox][4] > 1):
                Quant_ox[ox]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])/(element_properties[ox][2]*element_properties[ox][3])
            if (element_properties[ox][3] > 1) and (element_properties[ox][4] == 1):
                Quant_ox[ox]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])/(element_properties[ox][2]*element_properties[ox][3])
            if (element_properties[ox][3] == 1) and (element_properties[ox][4] > 1):
                Quant_ox[ox]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])/(element_properties[ox][2]*element_properties[ox][3])
            if (element_properties[ox][3] == 1) and (element_properties[ox][4] == 1):
                Quant_ox[ox]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])/(element_properties[ox][2]*element_properties[ox][3])

    return Quant_ox

def calcElements(Data, Oxide):
    """
    Convert oxide wt% data to element wt% data

    Parameters:
    ----------
    Data: dict or pandas dataframe, required
        Python dictionary containing a numpy array for each element of interest.

    Oxide: list, required
        List containing the elements that the user wishes to convert into oxide concentrations.

    Returns:
    ----------
    Quant_ox: dict or pandas dataframe
        Python dictionary with a numpy array for each oxide specified by the user.

    """

    Quant_el = Data.copy()
    for ox in Oxide:
        if (element_properties[ox][3] > 1) and (element_properties[ox][4] > 1):
            Quant_el[ox]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3])/(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])
        if (element_properties[ox][3] > 1) and (element_properties[ox][4] == 1):
            Quant_el[ox]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3])/(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])
        if (element_properties[ox][3] == 1) and (element_properties[ox][4] > 1):
            Quant_el[ox]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3])/(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])
        if (element_properties[ox][3] == 1) and (element_properties[ox][4] == 1):
            Quant_el[ox]=Data[ox]*(element_properties[ox][2]*element_properties[ox][3])/(element_properties[ox][2]*element_properties[ox][3]+15.999*element_properties[ox][4])

    return Quant_el

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
