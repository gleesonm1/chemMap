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
from copy import deepcopy
from matplotlib.colors import ListedColormap, is_color_like
from matplotlib.colors import Normalize, to_rgba
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from copy import deepcopy
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.colors as mcolors
from chemMap.crystalanalysis import *

# def plot_CompositionMap(Data = None, Element = None, Bounds = None, Phase = None, 
#             cmap = "viridis", background = 'k', Resolution = None, scalebar_loc = "lower right", save_fig = None):
#     Dat = deepcopy(Data)
#     X_1=np.linspace(0,np.shape(Dat[list(Dat.keys())[0]])[1]-1, np.shape(Dat[list(Dat.keys())[0]])[1])
#     Y_1=np.linspace(0, np.shape(Dat[list(Dat.keys())[0]])[0]-1, np.shape(Dat[list(Dat.keys())[0]])[0])
#     X, Y = np.meshgrid(X_1, Y_1)

#     f = plt.figure()
#     f.set_size_inches(8, 8*len(Y_1)/len(X_1))
#     a = f.add_subplot(111, aspect = "equal")
#     a.patch.set_facecolor(background)

#     a.get_xaxis().set_visible(False)
#     a.get_yaxis().set_visible(False)

#     if type(Element) == str:
#         if Phase is None:
#             if Bounds is not None:
#                 Dat[Element][np.where(Dat[Element] < Bounds[0])] = Bounds[0]
#                 Dat[Element][np.where(Dat[Element] > Bounds[1])] = Bounds[1]
#             z1 = a.imshow(Dat[Element], cmap = cmap, interpolation='none', origin='lower')
#         else:
#             if "Mineral" in Dat.keys():
#                 s = "Mineral"
#             else:
#                 s = "Cluster"

#             Dat[Element][np.where(Dat[s] != Phase)] = np.nan
#             if Bounds is not None:
#                 Dat[Element][np.where(Dat[Element] < Bounds[0])] = Bounds[0]
#                 Dat[Element][np.where(Dat[Element] > Bounds[1])] = Bounds[1]
#             z1 = a.imshow(Dat[Element], cmap = cmap, interpolation='none', origin='lower')


#         cbaxes = f.add_axes([0.90, 0.2, 0.02, 0.6])

#         cbar=plt.colorbar(z1, cax=cbaxes)
#         if Element == 'Mg#' or 'An' or 'AnK':
#             cbar.set_label(Element, rotation=90, fontsize = 16)
#         else:
#             cbar.set_label(Element, rotation=90, fontsize = 14)
#     else:
#         A = {}
#         for e in Element:
#             try:
#                 A[e] = plt.get_cmap(cmap[e])
#             except ValueError:
#                 # Define the colors
#                 color_start = (1, 1, 1, 0)  # Transparent white
#                 color_end = cmap[e]  # You can replace 'blue' with any valid matplotlib color

#                 # Create the colormap
#                 A[e] = LinearSegmentedColormap.from_list('custom_colormap', [color_start, color_end], N=256)
        
#         B = {}
#         if Bounds is None:
#             for e in Element:
#                 B[e] = plt.Normalize(vmin=np.nanmin(Dat[e]), vmax=np.nanmax(Dat[e]))
#         else:
#             for e in Element:
#                 B[e] = plt.Normalize(vmin=Bounds[e][0], vmax=Bounds[e][1])
#                 Dat[e][np.where(Dat[e] < Bounds[e][0])] = Bounds[e][0]
#                 Dat[e][np.where(Dat[e] > Bounds[e][1])] = Bounds[e][1]

#         color = {}
#         color_max = {}
#         for e in Element:
#             color[e] = A[e](B[e](Dat[e]), alpha=B[e](Dat[e]))
#             color_max[e] = A[e](B[e](np.nanmax(Dat[e])))

#         sum = np.zeros(np.shape(color[e][:,:,3]))
#         for e in Element:
#             sum = sum + color[e][:,:,3]

#         prop = {}
#         for e in Element:
#             prop[e] = color[e][:,:,3]/sum

#         combined = np.zeros(np.shape(color[e]))
#         combined[:,:,3] = combined[:,:,3]+1
#         for i in range(3):
#             for e in Element:
#                 combined[:,:,i] = combined[:,:,i] + prop[e]*color[e][:,:,i]
#                 # combined[:,:,i] = combined[:,:,i] + prop[e]*color_max[e][i]
        
#         z1 = a.imshow(combined, interpolation='none', origin='lower')

#         # cbaxes = {}
#         # if i in range(len(Element)):
#         i = 0
#         for e in Element:
#             cbaxes = f.add_axes([0.90, 0.1 + 0.8*i/len(Element), 0.02, 0.75/len(Element)])            
#             cbar=plt.colorbar(plt.cm.ScalarMappable(norm=B[e], cmap=A[e]), cax=cbaxes)
#             i = i + 1
#             if e == 'Mg#' or 'An' or 'AnK':
#                 cbar.set_label(e, rotation=90, fontsize = 16)
#             else:
#                 cbar.set_label(e, rotation=90, fontsize = 14)

#     if Resolution is not None:
#         scalebar = ScaleBar(Resolution, "um", length_fraction=0.2, location = scalebar_loc)
#         a.add_artist(scalebar)

#         # a.plot([len(X[0])-3*len(X[0])/4,len(X[0])-3*len(X[0])/4+100/Resolution],[10,10],'k-',linewidth=2)


#     plt.show()

#     if save_fig is not None:
#         plt.savefig(save_fig + "_phaseMap.svg", dpi=600)

#     return f, a

# def plot_CompositionMap(Data=None, Elements=None, Bounds=None, Phases=None,
#                          cmap="viridis", background='k', Resolution=None, scalebar_loc="lower right", save_fig=None):
#     Dat = deepcopy(Data)
#     X_1 = np.linspace(0, np.shape(Dat[list(Dat.keys())[0]])[1] - 1, np.shape(Dat[list(Dat.keys())[0]])[1])
#     Y_1 = np.linspace(0, np.shape(Dat[list(Dat.keys())[0]])[0] - 1, np.shape(Dat[list(Dat.keys())[0]])[0])
#     X, Y = np.meshgrid(X_1, Y_1)

#     f = plt.figure()
#     f.set_size_inches(8, 8 * len(Y_1) / len(X_1))
#     a = f.add_subplot(111, aspect="equal")
#     a.patch.set_facecolor(background)

#     a.get_xaxis().set_visible(False)
#     a.get_yaxis().set_visible(False)

#     if isinstance(Elements, str):
#         Elements = [Elements]
#     if isinstance(Bounds, list) or Bounds is None:
#         Bounds = {e: Bounds for e in Elements}
#     if isinstance(Phases, list) or Phases is None:
#         Phases = {e: Phases for e in Elements}

#     for element in Elements:
#         phase = Phases.get(element)
#         bounds = Bounds.get(element)

#         Dat_copy = deepcopy(Dat[element])

#         if phase is not None:
#             phase_key = "Mineral" if "Mineral" in Dat.keys() else "Cluster"
#             Dat_copy[np.where(Dat[phase_key] != phase)] = np.nan

#         if bounds is not None:
#             Dat_copy[np.where(Dat_copy < bounds[0])] = bounds[0]
#             Dat_copy[np.where(Dat_copy > bounds[1])] = bounds[1]

#         z1 = a.imshow(Dat_copy, cmap=cmap, interpolation='none', origin='lower')

#         cbaxes = f.add_axes([0.90, 0.2, 0.02, 0.6])
#         cbar = plt.colorbar(z1, cax=cbaxes)
#         if element in ['Mg#', 'An', 'AnK']:
#             cbar.set_label(element, rotation=90, fontsize=16)
#         else:
#             cbar.set_label(element, rotation=90, fontsize=14)

#     if Resolution is not None:
#         scalebar = ScaleBar(Resolution, "um", length_fraction=0.2, location=scalebar_loc)
#         a.add_artist(scalebar)

#     plt.show()

#     if save_fig is not None:
#         plt.savefig(save_fig + "_phaseMap.svg", dpi=600)

#     return f, a

def plot_CompositionMap(Data=None, Elements=None, Bounds=None, Phases=None,
                         colormap=None, background='k', Resolution=None, 
                         legend_loc="upper left", save_fig=None,
                         scalebar_loc="lower right", transect = False,
                         transect_phase = None, transect_chem = "MgO"):
    Dat = deepcopy(Data)
    X_1 = np.linspace(0, np.shape(Dat[list(Dat.keys())[0]])[1] - 1, np.shape(Dat[list(Dat.keys())[0]])[1])
    Y_1 = np.linspace(0, np.shape(Dat[list(Dat.keys())[0]])[0] - 1, np.shape(Dat[list(Dat.keys())[0]])[0])
    X, Y = np.meshgrid(X_1, Y_1)

    f = plt.figure()
    f.set_size_inches(8, 8 * len(Y_1) / len(X_1))
    a = f.add_subplot(111, aspect="equal")
    a.patch.set_facecolor(background)

    a.get_xaxis().set_visible(False)
    a.get_yaxis().set_visible(False)

    if isinstance(Elements, str):
        Elements = [Elements]
    if Bounds is None:
        Bounds = {e: None for e in Elements}
    elif isinstance(Bounds, list):
        Bounds = {e: b for e, b in zip(Elements, Bounds)}
    if Phases is None:
        Phases = {e: None for e in Elements}
    if isinstance(Phases, list):
        Phases = {e: p for e, p in zip(Elements, Phases)}
    if colormap is None:
        colormap = {e: "viridis" for e in Elements}
    elif isinstance(colormap, list):
        colormap = {e: cmap for e, cmap in zip(Elements, colormap)}

    legend_items = []

    for element in Elements:
        phase = Phases.get(element)
        bounds = Bounds.get(element)
        cmap_element = colormap.get(element)

        Dat_copy = deepcopy(Dat[element])

        if phase is not None:
            phase_key = "Mineral" if "Mineral" in Dat.keys() else "Cluster"
            Dat_copy[np.where(Dat[phase_key] != phase)] = np.nan

        if bounds is None:
            bounds = [np.nanpercentile(Dat_copy, 10), np.nanpercentile(Dat_copy, 90)]
        if bounds is not None:
            Dat_copy[np.where(Dat_copy < bounds[0])] = bounds[0]
            Dat_copy[np.where(Dat_copy > bounds[1])] = bounds[1]

        # Handle colormaps or fixed colors
        mask = ~np.isnan(Dat_copy)

        # Handle both colormaps and single fixed colors
        if isinstance(cmap_element, str) and cmap_element in plt.colormaps():
            # Plot with a colormap
            a.imshow(Dat_copy, cmap=cmap_element, interpolation='none', origin='lower')

            legend_items.append({'label': phase, 'oxide': element, 'cmap': cmap_element, 'color': False})
        elif is_color_like(cmap_element):
            # Plot with a single fixed color only in valid regions
            color_data = np.zeros_like(Dat_copy, dtype=float)
            color_data[mask] = 1.0 
            color_data[~mask] = np.nan  # Set valid data points to 1
            a.imshow(color_data, cmap=ListedColormap([cmap_element]), interpolation='none', origin='lower', alpha=1)
            legend_items.append({'label': phase, 'oxide': '', 'cmap': False, 'color': cmap_element})


    # # Build the custom legend
    # ax_legend = f.add_axes([0.01, 0.9, 0.1, 0.09])
    # ax_legend.axis('off')

    # # ax_legend.text(0.15, 1.0, "Min", fontsize=12, ha="center", transform=ax_legend.transAxes)
    # # ax_legend.text(0.85, 1.0, "Max", fontsize=12, ha="center", transform=ax_legend.transAxes)

    # # y_offset = 0.9
    # # for phase_name, element_name, min_color, max_color in legend_items:
    # #     ax_legend.text(0.05, y_offset, phase_name, ha='left', va='center')
    # #     if element_name:
    # #         ax_legend.text(0.3, y_offset, element_name, ha='left', va='center')
    # #         ax_legend.add_patch(Rectangle((0.6, y_offset - 0.025), 0.1, 0.05, color=min_color))
    # #         ax_legend.add_patch(Rectangle((0.8, y_offset - 0.025), 0.1, 0.05, color=max_color))
    # #     else:
    # #         ax_legend.add_patch(Rectangle((0.7, y_offset - 0.025), 0.2, 0.05, color=min_color))
    # #     y_offset -= 0.1

    # # Plot each element as a row
    # for i, elem in enumerate(legend_items):
    #     label = elem['label']
    #     property_name = elem['oxide']
    #     cmap_name = elem.get('cmap')
    #     color = elem.get('color')

    #     y_pos = len(legend_items) - i
    #     ax_legend.text(0.1, y_pos, label, va='center', ha='left')
    #     ax_legend.text(0.4, y_pos, property_name, va='center', ha='left')

    #     if cmap_name:
    #         cmap = plt.get_cmap(cmap_name)
    #         gradient = np.linspace(0, 1, 256).reshape(1, -1)
    #         ax_legend.imshow(
    #             gradient, aspect='auto', cmap=cmap, 
    #             extent=[0.7, 1.7, y_pos - 0.25, y_pos + 0.25]
    #         )
    #     elif color:
    #         ax_legend.add_patch(
    #             plt.Rectangle(
    #                 (0.7, y_pos - 0.25), 1.0, 0.5, color=color, transform=ax_legend.transData
    #             )
    #         )

    if Resolution is not None:
        scalebar = ScaleBar(Resolution, "um", length_fraction=0.2, location=scalebar_loc)
        a.add_artist(scalebar)

    plt.show()

    if save_fig is not None:
        plt.savefig(save_fig + "_phaseMap.svg", dpi=600)

    if transect is True:
        Tr = Section(Data, a, transect_chem, Resolution, Mineral=transect_phase)

        return Tr, f, a 
    else:
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