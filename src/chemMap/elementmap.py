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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def ElementMap(Data, Element, ColMap = None, Resolution = None, Bounds = None, Cluster = None, background = None, alpha = None):
    """
    Create a colourmap plot of one or more element.

    Parameters:
    ----------
    Data: dict, required
        Python dictionary containing numpy arrays for each element.

    Element: list, required
        List of string variables defining the elements or ratios to be plotted in the colormap.

    ColMap: list, optional
        List of colormap options to be used for the figure. If None, default values will be used ['viridis', 'magma', 'Reds', 'Blues']

    Resolution: float, optional
        The size of each pixel in the map. If not None, a 1 mm scale bar will be plotted by default.

    Bounds: array, optional
        min and max value of the element or ratio of interest. Only values within this range will be plotted.

    Cluster: list or array, optional
        Specify one or more clusters to target in the colourmap.

    Returns:
    ----------
    f, a: figure and subplot axes

    """
    plt.rcParams.update({'font.size': 12})

    X_1=np.linspace(0,np.shape(Data[Element[0]])[1]-1, np.shape(Data[Element[0]])[1])
    Y_1=np.linspace(0, np.shape(Data[Element[0]])[0]-1, np.shape(Data[Element[0]])[0])
    X, Y = np.meshgrid(X_1, Y_1)

    f = plt.figure()
    f.set_size_inches(8, 8*len(Y_1)/len(X_1))
    if background is not None:
        a = f.add_subplot(111, aspect = "equal")
        a.patch.set_facecolor('k')
    else:
        a = f.add_subplot(111, aspect = "equal")

    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)
    a.invert_yaxis()

    # if alpha is None:
    #     alpha = 1

    i = 0

    if ColMap is None:
        ColMap = ['viridis', 'magma', 'Reds', 'Blues']

    for E in Element:

        cmap=plt.get_cmap(ColMap[i])
        Dat = Data[E].copy()

        if Bounds is not None:
            Dat[np.where(Dat<Bounds[i][0])]=Bounds[i][0]
            Dat[np.where(Dat>Bounds[i][1])]=Bounds[i][1]

        if Cluster is not None:
            Dat[np.where(Data['Cluster'] != Cluster[i])] = np.nan

        # determine new colormap
        my_cmap = cmap(np.arange(cmap.N))
        if alpha is not None:
            my_cmap[:,3] = np.linspace(0,alpha,len(my_cmap[:,3]))

        my_cmap = ListedColormap(my_cmap)

        z1 = a.pcolormesh(X, Y, Dat, cmap = my_cmap, zorder = 2, shading = 'auto')

        cbaxes = f.add_axes([0.02, 0.1+0.8/(len(Element))*i, 0.02, -0.05+0.8/len(Element)])

        cbar=plt.colorbar(z1, cax=cbaxes)
        if E == 'Mg#' or 'An':
            cbar.set_label(E, rotation=90, fontsize = 16)
        else:
            cbar.set_label(E + ' (wt%)', rotation=90, fontsize = 14)

        if Resolution is not None:
            a.plot([len(X[0])-3*len(X[0])/4,len(X[0])-3*len(X[0])/4+1000/Resolution],[10,10],'k-',linewidth=2)

        i = i + 1

    plt.draw()
    plt.show()

    return f, a