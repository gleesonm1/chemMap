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
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

def plot_CompositionMap(Data = None, Element = None, Bounds = None, Phase = None, 
            cmap = "viridis", background = 'k', Resolution = None, scalebar_loc = "lower right", save_fig = None):
    Dat = Data.copy()
    X_1=np.linspace(0,np.shape(Dat[list(Dat.keys())[0]])[1]-1, np.shape(Dat[list(Dat.keys())[0]])[1])
    Y_1=np.linspace(0, np.shape(Dat[list(Dat.keys())[0]])[0]-1, np.shape(Dat[list(Dat.keys())[0]])[0])
    X, Y = np.meshgrid(X_1, Y_1)

    f = plt.figure()
    f.set_size_inches(8, 8*len(Y_1)/len(X_1))
    a = f.add_subplot(111, aspect = "equal")
    a.patch.set_facecolor(background)

    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

    if type(Element) == str:
        if Phase is None:
            if Bounds is not None:
                Dat[Element][np.where(Dat[Element] < Bounds[0])] = Bounds[0]
                Dat[Element][np.where(Dat[Element] > Bounds[1])] = Bounds[1]
            z1 = a.imshow(Dat[Element], cmap = cmap, interpolation='none', origin='lower')
        else:
            if "Mineral" in Data.keys():
                s = "Mineral"
            else:
                s = "Cluster"

            Dat[Element][np.where(Dat[s] != Phase)] = np.nan
            if Bounds is not None:
                Dat[Element][np.where(Dat[Element] < Bounds[0])] = Bounds[0]
                Dat[Element][np.where(Dat[Element] > Bounds[1])] = Bounds[1]
            z1 = a.imshow(Dat[Element], cmap = cmap, interpolation='none', origin='lower')


        cbaxes = f.add_axes([0.90, 0.2, 0.02, 0.6])

        cbar=plt.colorbar(z1, cax=cbaxes)
        if Element == 'Mg#' or 'An' or 'AnK':
            cbar.set_label(Element, rotation=90, fontsize = 16)
        else:
            cbar.set_label(Element, rotation=90, fontsize = 14)
    else:
        A = {}
        for e in Element:
            try:
                A[e] = plt.get_cmap(cmap[e])
            except ValueError:
                # Define the colors
                color_start = (1, 1, 1, 0)  # Transparent white
                color_end = cmap[e]  # You can replace 'blue' with any valid matplotlib color

                # Create the colormap
                A[e] = LinearSegmentedColormap.from_list('custom_colormap', [color_start, color_end], N=256)
        
        B = {}
        if Bounds is None:
            for e in Element:
                B[e] = plt.Normalize(vmin=np.nanmin(Dat[e]), vmax=np.nanmax(Dat[e]))
        else:
            for e in Element:
                B[e] = plt.Normalize(vmin=Bounds[e][0], vmax=Bounds[e][1])
                Dat[e][np.where(Dat[e] < Bounds[e][0])] = Bounds[e][0]
                Dat[e][np.where(Dat[e] > Bounds[e][1])] = Bounds[e][1]

        color = {}
        color_max = {}
        for e in Element:
            color[e] = A[e](B[e](Dat[e]), alpha=B[e](Dat[e]))
            color_max[e] = A[e](B[e](np.nanmax(Dat[e])))

        sum = np.zeros(np.shape(color[e][:,:,3]))
        for e in Element:
            sum = sum + color[e][:,:,3]

        prop = {}
        for e in Element:
            prop[e] = color[e][:,:,3]/sum

        combined = np.zeros(np.shape(color[e]))
        combined[:,:,3] = combined[:,:,3]+1
        for i in range(3):
            for e in Element:
                combined[:,:,i] = combined[:,:,i] + prop[e]*color[e][:,:,i]
                # combined[:,:,i] = combined[:,:,i] + prop[e]*color_max[e][i]
        
        z1 = a.imshow(combined, interpolation='none', origin='lower')

        # cbaxes = {}
        # if i in range(len(Element)):
        i = 0
        for e in Element:
            cbaxes = f.add_axes([0.90, 0.1 + 0.8*i/len(Element), 0.02, 0.75/len(Element)])            
            cbar=plt.colorbar(plt.cm.ScalarMappable(norm=B[e], cmap=A[e]), cax=cbaxes)
            i = i + 1
            if e == 'Mg#' or 'An' or 'AnK':
                cbar.set_label(e, rotation=90, fontsize = 16)
            else:
                cbar.set_label(e, rotation=90, fontsize = 14)

    if Resolution is not None:
        scalebar = ScaleBar(Resolution, "um", length_fraction=0.2, location = scalebar_loc)
        a.add_artist(scalebar)

        # a.plot([len(X[0])-3*len(X[0])/4,len(X[0])-3*len(X[0])/4+100/Resolution],[10,10],'k-',linewidth=2)


    plt.show()

    if save_fig is not None:
        plt.savefig(save_fig + "_phaseMap.svg", dpi=600)

    return f, a


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