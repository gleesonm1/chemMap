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

def ElementMap(Data, Element, ColMap = None, Resolution = None, Bounds = None, Cluster = None):
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
    X=np.linspace(0,len(Data[Element[0]][0,:])-1,len(Data[Element[0]][0,:]))
    Y=np.linspace(0,len(Data[Element[0]][:,0])-1,len(Data[Element[0]][:,0]))
    X, Y = np.meshgrid(X, Y)

    f, a = plt.subplots(1, 1, figsize=(15,15*Y[-1,-1]/X[-1,-1]))
    a.axis('off')
    a.set_aspect(len(X)/len(Y))

    i = 0

    if ColMap is None:
        ColMap = ['viridis', 'magma', 'Reds', 'Blues']

    for E in Element:
        if Bounds is not None:
            Dat = Data[E].copy()
            Dat[np.where(Dat<Bounds[i][0])]=np.nan
            Dat[np.where(Dat>Bounds[i][1])]=np.nan

        if Cluster is None:
            z1=a.contourf(X, Y, Dat, 20, cmap=ColMap[i],zorder=0)
            z1=a.contourf(X, Y, Dat, 20, cmap=ColMap[i],zorder=0)
            z1=a.contourf(X, Y, Dat, 20, cmap=ColMap[i],zorder=0)
        else:
            A = Dat
            A[np.where(Data['Cluster'] != Cluster[i])] = np.nan
            z1=a.contourf(X, Y, A, 20, cmap=ColMap[i],zorder=0)
            z1=a.contourf(X, Y, A, 20, cmap=ColMap[i],zorder=0)
            z1=a.contourf(X, Y, A, 20, cmap=ColMap[i],zorder=0)

        cbaxes = f.add_axes([0.02, 0.7-0.3*i, 0.02, 0.25])
        cbar=plt.colorbar(z1,cax=cbaxes)
        if E == 'Mg#' or 'An':
            cbar.set_label(E, rotation=90)
        else:
            cbar.set_label(E + ' (wt%)', rotation=90)

        if Resolution is not None:
            a.plot([len(X[0])-3*len(X[0])/4,len(X[0])-3*len(X[0])/4+1000/Resolution],[10,10],'k-',linewidth=2)

        i = i + 1

    plt.draw()
    plt.show()

    return f, a