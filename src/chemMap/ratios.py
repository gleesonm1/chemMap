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
from .elements import correction_models

def calcRatios(Dat, oxide = None, ratios = None, numerator = None, denominator = None):
    """
    Function used to calculate the molar ratio of two elements (A/(A+B)) from wt% quantitative chemical data. This can either be done by specifying one of the pre-defined ratios (Mg#, An, AnK, Cr#) or the numerator and denominator for a new ratio.

    Parameters
    --------
    Data: dict, required
        Dictionary containing numpy arrays for each element.

    oxide: bool, optional
        If not None, ratios will be calculated assuming the input data is in the form of oxide wt% (e.g., MgO wt%). If None (default), calculations will be performed assuming that the data is in the form of element wt% (e.g., Mg wt%).

    ratios: list, optional
        Provide a list of the pre-defined ratios to be calculated (e.g., ratios = ['An','Mg#']. If no ratio provided, please instead provide a numerator and denominator.

    numerator: string, optional
    denominator: string, optional
        To calculate a ratio that is not pre-defined, please provide the numerator and denominator elements. Please note, even if dealing with oxide concentrations only the element should be provided here (e.g., 'Mg', rather than 'MgO').

    Returns
    --------
    Data: dict
        Copy of the input dictionary with additional entries for all ratios calculated.

    """
    Data = Dat.copy()

    if oxide is not None:
        if ratios is not None:
            for r in ratios:
                if r == 'Mg#':
                    Data[r] = (Data['MgO']/40.3044)/(Data['FeO']/71.844+Data['MgO']/40.3044)

                if r == 'An':
                    Data[r] = (Data['CaO']/56.0774)/(Data['CaO']/56.0774+2*Data['Na2O']/61.9789)

                if r == 'AnK':
                    Data[r] = (Data['CaO']/56.0774)/(Data['CaO']/56.0774+2*Data['Na2O']/61.9789+2*Data['K2O']/94.2)

                if r == 'Cr#':
                    Data[r] = (Data['Cr2O3']/(element_properties['Cr'][3]*element_properties['Cr'][2]+element_properties['Cr'][4]*15.999))/(Data['Cr2O3']/(element_properties['Cr'][3]*element_properties['Cr'][2]+element_properties['Cr'][4]*15.999) + Data['Al2O3']/(element_properties['Al'][3]*element_properties['Al'][2]+element_properties['Al'][4]*15.999))

        if numerator is not None and denominator is not None:
            if (element_properties[numerator][3] > 1) and (element_properties[numerator][4] > 1):
                num_ox = numerator + str(element_properties[ox][3]) + 'O' + str(element_properties[ox][4])
            if (element_properties[numerator][3] > 1) and (element_properties[numerator][4] == 1):
                num_ox = numerator + str(element_properties[ox][3]) + 'O'
            if (element_properties[numerator][3] == 1) and (element_properties[numerator][4] > 1):
                num_ox = numerator + 'O' + str(element_properties[ox][4])
            if (element_properties[numerator][3] == 1) and (element_properties[numerator][4] == 1):
                num_ox = numerator + 'O'

            if (element_properties[denominator][3] > 1) and (element_properties[denominator][4] > 1):
                den_ox = denominator + str(element_properties[ox][3]) + 'O' + str(element_properties[ox][4])
            if (element_properties[denominator][3] > 1) and (element_properties[denominator][4] == 1):
                den_ox = denominator + str(element_properties[ox][3]) + 'O'
            if (element_properties[denominator][3] == 1) and (element_properties[denominator][4] > 1):
                den_ox = denominator + 'O' + str(element_properties[ox][4])
            if (element_properties[denominator][3] == 1) and (element_properties[denominator][4] == 1):
                den_ox = denominator + 'O'

            Data[numerator + '/(' + numerator + '+' + denominator +')'] = (Data[num_ox]/(element_properties[numerator][3]*element_properties[numerator][2]+element_properties[numerator][4]*15.999))/(Data[num_ox]/(element_properties[numerator][3]*element_properties[numerator][2]+element_properties[numerator][4]*15.999) + Data[den_ox]/(element_properties[denominator][3]*element_properties[denominator][2]+element_properties[denominator][4]*15.999))

    if oxide is None:
        if ratios is not None:
            for r in ratios:
                if r == 'Mg#':
                    Data[r] = (Data['Mg']/element_properties['Mg'][2])/(Data['Mg']/element_properties['Mg'][2]+Data['Fe']/element_properties['Fe'][2])

                if r == 'An':
                    Data[r] = (Data['Ca']/element_properties['Ca'][2])/(Data['Ca']/element_properties['Ca'][2]+Data['Na']/element_properties['Na'][2])

                if r == 'AnK':
                    Data[r] = (Data['Ca']/element_properties['Ca'][2])/(Data['Ca']/element_properties['Ca'][2]+Data['Na']/element_properties['Na'][2]+Data['K']/element_properties['K'][2])

                if r == 'Cr#':
                    Data[r] = (Data['Cr']/element_properties['Cr'][2])/(Data['Cr']/element_properties['Cr'][2]+Data['Al']/element_properties['Al'][2])

        if numerator is not None and denominator is not None:
            Data[numerator + '/(' + numerator + '+' + denominator +')'] = (Data[numerator]/element_properties[numerator][2])/(Data[numerator]/element_properties[numerator][2]+Data[denominator]/element_properties[demoninator][2])

    return Data

def Norm(Data, Oxide):
    """
    Data for each pixel is normalised to 1

    Parameters:
    ----------
    Data: dict
        Python dictionary of different elements

    Oxide: list
        list of oxide names to be involved in

    Returns:
    ----------
    norm: dict
        Python dictionary containing the normalised data.
    """
    sum_ox = np.zeros(np.shape(Data[Oxide[0]]))

    for ox in Oxide:
        sum_ox = sum_ox + Data[ox]

    norm=Data.copy()
    for ox in Oxide:
        norm[ox] = Data[ox]/sum_ox

    return norm

def AZ_calc(norm_entry, Elements):
    """
    Calculate the mean atomic number and atomic weight of each pixel.

    Parameters:
    ----------
    norm: dict
        Python dictionary of normalised data

    Elements: list
        list of elements to be used in the AZ calculations

    Returns:
    ----------
    norm: dict
        Also now includes numpy arrays for mean A and Z of each pixel
    """

    norm = norm_entry.copy()

    O = np.zeros(np.shape(norm[Elements[0]]))
    for el in Elements:
        dummy = norm[el]*(element_properties[el][2]*element_properties[el][3] + element_properties['O'][2]*element_properties[el][4])/(element_properties[el][2]*element_properties[el][3])
        O = O + (dummy - norm[el])
    norm['O'] = O

    Elements.append('O')

    norm = Norm(norm, Elements)

    A = np.zeros(np.shape(norm[Elements[0]]))
    Z = np.zeros(np.shape(norm[Elements[0]]))

    for el in Elements:
        A = A + norm[el]*element_properties[el][2]
        Z = Z + norm[el]*element_properties[el][1]

    norm['A'] = A
    norm['Z'] = Z

    return norm

def h_factor(norm):
    """
    Calculates a h-factor for every pixel

    Parameters:
    ----------
    norm: dict
        Python dictionary of normalised data

    Returns:
    ----------
    h-factor: numpy array
        array of the h-factor values
    """

    h_factor = 1.2 * norm['A']/(norm['Z']**2)

    return h_factor

def h_factor_new(norm_entry, Elements):
    """
    Calculate the mean atomic number and atomic weight of each pixel.

    Parameters:
    ----------
    norm: dict
        Python dictionary of normalised data

    Elements: list
        list of elements to be used in the AZ calculations

    Returns:
    ----------
    norm: dict
        Also now includes numpy arrays for mean A and Z of each pixel
    """

    norm = norm_entry.copy()

    O = np.zeros(np.shape(norm[Elements[0]]))
    for el in Elements:
        dummy = norm[el]*(element_properties[el][2]*element_properties[el][3] + element_properties['O'][2]*element_properties[el][4])/(element_properties[el][2]*element_properties[el][3])
        O = O + (dummy - norm[el])
    norm['O'] = O

    Elements.append('O')

    norm = Norm(norm, Elements)

    h = np.zeros(np.shape(norm[Elements[0]]))

    for el in Elements:
        h = h + norm[el]*(1.2*element_properties[el][2]/(element_properties[el][1]**2))

    return h

def ApparentRatio(counts, ratio = None):
    """
    Parameters:
    ----------
    counts: dict
        Python dictionary containing raw count data

    ratio: string
        Ratio to be calculated from the raw data.

    Returns:
    counts: dict
        New numpy array for Apparent Ratio included
    """
    if ratio == 'Mg#':
        counts[ratio]=counts['Mg']/(counts['Mg']+counts['Fe'])

    if ratio == 'An':
        counts[ratio]=counts['Ca']/(counts['Ca']+counts['Na'])

    if ratio == 'AnK':
        counts[ratio]=counts['Ca']/(counts['Ca']+counts['Na']+counts['K'])

    if ratio == 'Cr#':
        counts[ratio]=counts['Cr']/(counts['Cr']+counts['Al'])

    return counts

def detRatio(cnts, Elements, corr = None, ratio = None):
    """
    Calculates the concentration (or ratio) from normalise, h-factor multiplied data. Load raw count data and the normalisation will be done here.

    Parameters:
    ----------
    counts: dict, required
        Python dictionary containing the raw count data.

    Elements: list
        full list of elements included in the calculations.

    corr: string, optional
        Correction factor to be used in the calculations.

    ratio: string, required
        Ratio or element to be quantified here.

    Returns:
    ----------
    norm: dict
        copy of python dictionary with chosen ratio included
    """
    counts = cnts.copy()

    if ratio == 'Mg#' or 'Cr#' or 'An' or 'AnK':
        counts = ApparentRatio(counts, ratio = ratio)

    norm = Norm(counts, Elements)

    #norm = AZ_calc(norm, Elements)

    #h_Factor = h_factor(norm, Elements)

    #for el in Elements:
    #    norm[el]=norm[el]#*h_Factor

    if ratio is None:
        ratio = 'Mg#'

    norm = ApparentRatio(norm, ratio = ratio)

    # apply correction factor
    if corr is None:
        corr = 'general'

    poly = correction_models[corr][ratio][1]

    quant_r = np.zeros(np.shape(norm[Elements[0]]))

    for i in range(0,len(poly)):
        quant_r = quant_r + poly[i] * norm[ratio]**i

#    norm[ratio + '_quant_' + corr] = quant_r

    return quant_r
