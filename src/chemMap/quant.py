## Python Mapping tool
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import sys
import scipy
from chemMap.loaddata import *
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
from .elements import mineral_properties
import os
from scipy.optimize import curve_fit

def fit_func(x, a):
    return x*a

def fit_func_bkg(x, a, c):
    return x*a + c


## load standards and create calibration
def create_calibration(path, matrix_correction = None, background_correction = None, std_datatype = None):
    """
    create calibration curves from each element in the calibration dataset
    """
    Minerals = load_standards(path)
    Stds = true_comp(path+'Standards.xlsx', std_datatype = std_datatype)

    Stds_raw, base_corr, base_inte = cal_params(Minerals, Stds, matrix_correction = matrix_correction, background_correction = background_correction)

    return Minerals, Stds, Stds_raw, base_corr, base_inte

def load_standards(path):
    """
    Load counts or estimated wt% data for standards
    """
    all_subdirs = [d for d in os.listdir(path) if os.path.isdir(path + d)]
    Minerals = dict.fromkeys(all_subdirs)

    for m in Minerals:
        Minerals[m] = loadwt(path + m + '/', filter = True)

    return Minerals

def true_comp(file, std_datatype = None):
    """
    Load in a Pandas DataFrame of true standard compositions
    """
    Stds = pd.read_excel(file)
    if std_datatype == 'Ox%':
        Stds = calcElements(Stds, list(Stds.columns[2,-1]))

    return Stds

def cal_params(Minerals, Stds, matrix_correction = None, background_correction = None):
    Stds_raw = standard_average(Stds, Minerals)

    base_corr = dict.fromkeys(Stds.columns[2:-1])
    base_inte = base_corr.copy()

    for e in base_corr:
        if background_correction is None:
            params = curve_fit(fit_func, Stds[e][Stds['Calibrate'] == 'Y'], Stds_raw[e][Stds_raw['Calibrate'] == 'Y'])
            base_inte[e] = 0

        if background_correction is not None:
            params = curve_fit(fit_func_bkg, Stds[e][Stds['Calibrate'] == 'Y'], Stds_raw[e][Stds_raw['Calibrate'] == 'Y'])
            base_inte[e] = params[0][1]

        base_corr[e] = params[0][0]

    return Stds_raw, base_corr, base_inte

def standard_average(Stds, Minerals):
    """
    Determine the average count rate for each element in each standard. Place results in a pandas DataFrame to compare to known values.
    """
    Stds_raw = pd.DataFrame(columns = list(Stds.columns))
    Stds_raw['Mineral'] = Stds['Mineral']
    Stds_raw['Standard'] = Stds['Standard']
    Stds_raw['Calibrate'] = Stds['Calibrate']
    for m in Minerals:
        for e in Minerals[m]:
            Stds_raw[e][Stds_raw['Standard'] == m] = np.nanmean(Minerals[m][e])

    return Stds_raw

## load standards or raw data and calculate quantitative compositions
def calc_comps(Data, base_corr, base_inte, matrix_correction = None):
    Conc = dict.fromkeys(list(Data.keys()))
    for e in base_corr:
        Conc[e] = (Data[e]-base_inte[e])/base_corr[e]

    return Conc

def check_stds(Minerals, base_corr, base_inte, matrix_correction = None):
    Conc = dict.fromkeys(list(Minerals.keys()))
    for c in Conc:
        Conc[c] = calc_comps(Minerals[c], base_corr, base_inte)

    return Conc

## plot standards results

def plot_std_corr(Stds, Stds_raw, el, base_corr, base_inte, matrix_correction = None, MFC = None):
    f, a = plt.subplots(np.shape(el)[0], np.shape(el)[1])
    for i in range(np.shape(el)[0]):
        for j in range(np.shape(el)[1]):
            if MFC is None:
                a[i][j].plot(Stds[el[i][j]], Stds_raw[el[i][j]], 'ok')
            else:
                for s in MFC:
                    a[i][j].plot(Stds[el[i][j]][Stds['Mineral'] == s], Stds_raw[el[i][j]][Stds_raw['Mineral'] == s], 'ok', markerfacecolor = MFC[s])

            x_fit = np.array([np.nanmin(Stds[el[i][j]]), np.nanmax(Stds[el[i][j]])])
            y_fit = x_fit*base_corr[el[i][j]]+base_inte[el[i][j]]

            a[i][j].plot(x_fit,y_fit, '--')
            a[i][j].set_ylabel('Raw data')
            a[i][j].set_xlabel(el[i][j] + ' wt%')

def plot_std_comparison(Conc, Stds, el, Group = None):
    for s in Conc:
        f, d = plt.subplots(3,3, figsize = (10,8))
        f.suptitle(s)
        for i in range(np.shape(el)[0]):
            for j in range(np.shape(el)[1]):
                axx = d[i][j].hist(Conc[s][el[i][j]].flatten(), density = True, bins = 20, alpha = 0.6)

                if Group is not None:
                    A = np.zeros(Group*50)
                    for k in range(Group*50):
                        A[k] = np.nanmean(np.random.choice(Conc[s][el[i][j]].flatten(), size = Group, replace = False))

                    d[i][j].hist(A, density = True, bins = 10, alpha = 0.6, color = 'red')

                d[i][j].plot([Stds[el[i][j]][Stds['Standard'] == s],
                              Stds[el[i][j]][Stds['Standard'] == s]],
                            [0, np.max(axx[0])], '--k')

                # if s == 'Diopside':
                #     if el[i][j] == 'Ca':
                #         print(np.nanstd(A)/np.nanmean(A))
                #         print(np.nanstd(Conc[s][el[i][j]].flatten())/np.nanmean(Conc[s][el[i][j]].flatten()))


## calc whole-rock
def wholerock(Data, Oxide, X = None, iterations = None):
    ForWR = pd.DataFrame()
    for E in list(Data.keys()):
        ForWR[E] = Data[E][Data['Cluster'] != 'nan'].flatten()

    A = np.array([None]*len(ForWR[E]))

    for C in np.unique(ForWR['Cluster']):
        A[np.where(ForWR['Cluster'] == C)] = mineral_properties[C][1]

    ForWR['rho'] = A

    if X is None:
        WR = pd.DataFrame(columns = Oxide, data = np.zeros((1,len(Oxide))))
        for E in Oxide:
            WR[E].loc[0] = np.nansum(ForWR[E]*ForWR['rho'])/np.nansum(ForWR['rho'])

    if X is not None:
        Num = round(len(ForWR[E])*X)
        if iterations is None:
            WR = pd.DataFrame(columns = Oxide, data = np.zeros((1, len(Oxide))))
            idx = np.random.choice(np.linspace(0,len(ForWR[E])-1,len(ForWR[E])), round(len(ForWR[E]*X)), replace = False).astype(int)
            for O in Oxide:
                WR[O].loc[0] = np.nansum(ForWR[O].values[np.ix_(idx)]*ForWR['rho'].values[np.ix_(idx)])/np.nansum(ForWR['rho'].values[np.ix_(idx)])
        if iterations is not None:
            WR = pd.DataFrame(columns = Oxide, data = np.zeros((iterations, len(Oxide))))
            for i in range(iterations):
                idx = np.random.choice(np.linspace(0,len(ForWR[E])-1,len(ForWR[E])), round(len(ForWR[E])*X), replace = False).astype(int)
                for O in Oxide:
                    WR[O].loc[i] = np.nansum(ForWR[O].values[np.ix_(idx)]*ForWR['rho'].values[np.ix_(idx)])/np.nansum(ForWR['rho'].values[np.ix_(idx)])

    return WR



