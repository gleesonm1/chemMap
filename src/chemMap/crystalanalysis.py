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

def Section(DataEntry,a,Element,Resolution,Cluster=None, Mineral = None):
    """
    Function used to extract a transect from an element or ratio map. Code will prompt the user to place three 'clicks' on the map. The first two define the start and end of the transect and the third defines the width.

    Parameters:
    ---------
    Data: dict, required
        Dictionary containing numpy arrays for each element.

    a: matplotlib.axes._subplots.AxesSubplot, required
        The subplot axes that contains the relevant element map.

    Element: string, required
        The element or ratio that will be plotted.

    Resolution: float, required
        The size of each pixel - required to calculate the distance parameter

    Cluster: string or float, optional
        Define which cluster should be targeted.

    Returns:
    ----------
    Tr: pandas DataFrame:
        Dataframe containing the composition of each pixel used in the transect, their X-Y coordinates and the assumed distance along the transect for each point.

    Other outputs:
    ----------
    A white box is placed on the map in 'a' defining the region used to calculate the transect.

    A second plot is produced where the top graph displays the element vs distance relationship for every point included in the transect. The bottom graph then provides a moving average calculation along that transect (average all points within a Distance of +/- 2*Resolution), with the shaded region displaying the 2*sigma variation around that mean.

    """

    Data = DataEntry.copy()

    print('Click twice to define the start and the end of the chosen transect, and once to define the width of the area to consider')
    pts = np.asarray(plt.ginput(3, timeout=-1)) # it will wait for three clicks
    x1,y1=pts[0,:]
    x2,y2=pts[1,:]
    x3,y3=pts[2,:]

    # determine transect line and distance perpendicular to line
    m1=(y2-y1)/(x2-x1)
    c1=y1-m1*x1
    m2=-1/m1
    c2=y3-m2*x3

    x=(c2-c1)/(m1-m2)
    y=(m1*x)+c1
    d=np.sqrt((x3-x)**2+(y3-y)**2)

    a.plot([x1,x2],[y1,y2],'-k',linewidth=2,zorder=10)
    if m1>0:
        a.plot([x1+np.sqrt((x3-x)**2),x1-np.sqrt((x3-x)**2),x2-np.sqrt((x3-x)**2),x2+np.sqrt((x3-x)**2),x1+np.sqrt((x3-x)**2)],[y1-np.sqrt((y3-y)**2),y1+np.sqrt((y3-y)**2),y2+np.sqrt((y3-y)**2),y2-np.sqrt((y3-y)**2),y1-np.sqrt((y3-y)**2)],'-w')
    if m1<0:
        a.plot([x1+np.sqrt((x3-x)**2),x1-np.sqrt((x3-x)**2),x2-np.sqrt((x3-x)**2),x2+np.sqrt((x3-x)**2),x1+np.sqrt((x3-x)**2)],[y1+np.sqrt((y3-y)**2),y1-np.sqrt((y3-y)**2),y2-np.sqrt((y3-y)**2),y2+np.sqrt((y3-y)**2),y1+np.sqrt((y3-y)**2)],'-w')
    plt.draw()
    plt.show()

    # determine the location of every point
    dd=1

    X1=np.round(np.linspace(x1,x2,int(np.sqrt((x2-x1)**2+(y2-y1)**2)+1)))
    Y1=np.round(np.linspace(y1,y2,int(np.sqrt((x2-x1)**2+(y2-y1)**2)+1)))
    _X=np.round(np.linspace(X1+np.sqrt((x3-x)**2),X1-np.sqrt((x3-x)**2),int(2*np.sqrt((x3-x)**2+(y3-y)**2)+1)))
    _Y=np.round(np.linspace(Y1-np.sqrt((y3-y)**2),Y1+np.sqrt((y3-y)**2),int(2*np.sqrt((x3-x)**2+(y3-y)**2)+1)))

    Tr=pd.DataFrame()
    for ox in Data.keys():
        Tr[ox]=np.zeros(len(_Y.flatten()))
        A = Data[ox].copy()
        if Cluster is not None:
            A[np.where(Data['Cluster'] != Cluster)] = np.nan
        if Mineral is not None:
            A[np.where(Data['Mineral'] != Mineral)] = np.nan
        Tr[ox]=A[(_Y.flatten().astype(int)),(_X.flatten().astype(int))]

    C=_Y.flatten()-m2*_X.flatten()
    XX=(C-c1)/(m1-m2)
    YY=(m1*XX+c1)
    Tr['Distance']=np.sqrt((XX-x1)**2+(YY-y1)**2)*Resolution
    Tr['X']=_X.flatten()
    Tr['Y']=_Y.flatten()

    L=np.linspace(2*Resolution,np.max(Tr['Distance'])-2*Resolution,int(np.round(np.max(Tr['Distance'])/(0.5*Resolution)+1))) # distance for moving average and std
    Av=np.zeros(len(L))
    St=np.zeros(len(L))

    for i in range(0,len(L)):
        Av[i]=np.nanmean(Tr[Element][(Tr['Distance']>L[i]-2*Resolution) & (Tr['Distance']<L[i]+2*Resolution)])
        St[i]=2*np.nanstd(Tr[Element][(Tr['Distance']>L[i]-2*Resolution) & (Tr['Distance']<L[i]+2*Resolution)])

    f, b = plt.subplots(2,1, figsize=(8,6))
    b[0].plot(Tr['Distance'],Tr[Element],'ow',markerfacecolor=[0.5,0.5,0.5], zorder = 1)
    b[0].set_ylabel(Element)

    b[1].plot(L,Av,'-k',linewidth=2,zorder=1)
    b[1].fill(np.array([L,np.flip(L)]).flatten(),np.array([Av-St,np.flip(Av)+np.flip(St)]).flatten(),color="lightgrey", zorder = 0)
    b[1].set_ylabel(Element)

    plt.show()

    return Tr


# def Section(Data, a, Element, Resolution, Cluster=None, Mineral=None):
#     """
#     Extracts a transect from an element or ratio map based on user-defined clicks.

#     Parameters:
#     - Data: dict, contains numpy arrays for each element.
#     - a: matplotlib AxesSubplot containing the element map.
#     - Element: str, the element or ratio to be plotted.
#     - Resolution: float, pixel size used to calculate distance.
#     - Cluster: str/float, optional, specifies which cluster to extract.
#     - Mineral: str, optional, specifies which mineral to extract.

#     Returns:
#     - Tr: pandas DataFrame with composition, coordinates, and distance along the transect.
#     """

#     Data = Data.copy()

#     plt.show()
#     print('Click twice to define the transect (start and end), then click once to set the width.')
#     pts = np.asarray(plt.ginput(3, timeout=-1))  # Waits for 3 clicks
#     x1, y1 = pts[0]
#     x2, y2 = pts[1]
#     x3, y3 = pts[2]

#     # Compute slope and perpendicular distance
#     if x2 == x1:  # Handle vertical line case
#         m1 = np.inf
#     else:
#         m1 = (y2 - y1) / (x2 - x1)

#     c1 = y1 - m1 * x1 if m1 != np.inf else None
#     m2 = -1 / m1 if m1 != 0 and m1 != np.inf else 0  # Perpendicular slope
#     c2 = y3 - m2 * x3 if c1 is not None else None

#     # Find projection of (x3, y3) onto the transect
#     if m1 == np.inf:
#         x, y = x1, y3
#     else:
#         x = (c2 - c1) / (m1 - m2)
#         y = m1 * x + c1

#     d = np.sqrt((x3 - x) ** 2 + (y3 - y) ** 2)

#     # Plot the transect and rectangle
#     a.plot([x1, x2], [y1, y2], '-k', linewidth=2, zorder=10)
#     rect_x = [x1 + d, x1 - d, x2 - d, x2 + d, x1 + d]
#     rect_y = [y1 - d, y1 + d, y2 + d, y2 - d, y1 - d]
#     a.plot(rect_x, rect_y, '-w')

#     plt.draw()
#     plt.show()

#     # Define points along the transect
#     num_pts = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + 1)
#     X1 = np.linspace(x1, x2, num_pts)
#     Y1 = np.linspace(y1, y2, num_pts)

#     # Define the rectangle's width along the perpendicular
#     num_width_pts = int(2 * d / Resolution) + 1
#     _X, _Y = np.meshgrid(X1, np.linspace(y1 - d, y1 + d, num_width_pts))
#     _X, _Y = _X.flatten(), _Y.flatten()

#     # Clip indices to avoid out-of-bound errors
#     _X = np.clip(_X.astype(int), 0, Data[Element].shape[1] - 1)
#     _Y = np.clip(_Y.astype(int), 0, Data[Element].shape[0] - 1)

#     # Extract data
#     Tr = pd.DataFrame()
#     for ox in Data.keys():
#         A = Data[ox].copy()
#         if Cluster is not None:
#             A[np.where(Data['Cluster'] != Cluster)] = np.nan
#         if Mineral is not None:
#             A[np.where(Data['Mineral'] != Mineral)] = np.nan
#         Tr[ox] = A[_Y, _X]

#     # Compute distances
#     Tr['Distance'] = np.sqrt((X1 - x1) ** 2 + (Y1 - y1) ** 2) * Resolution
#     Tr['X'], Tr['Y'] = _X, _Y

#     # Compute moving average
#     L = np.linspace(2 * Resolution, np.max(Tr['Distance']) - 2 * Resolution, 
#                     int(np.round(np.max(Tr['Distance']) / (0.5 * Resolution) + 1)))
#     Av = np.array([np.nanmean(Tr[Element][(Tr['Distance'] > l - 2 * Resolution) & 
#                                           (Tr['Distance'] < l + 2 * Resolution)]) for l in L])
#     St = np.array([2 * np.nanstd(Tr[Element][(Tr['Distance'] > l - 2 * Resolution) & 
#                                              (Tr['Distance'] < l + 2 * Resolution)]) for l in L])

#     # Plot results
#     fig, ax = plt.subplots(2, 1, figsize=(8, 6))
#     ax[0].scatter(Tr['Distance'], Tr[Element], c='gray', marker='o', label='Raw Data')
#     ax[0].set_ylabel(Element)
#     ax[0].legend()

#     ax[1].plot(L, Av, '-k', linewidth=2, label='Moving Average')
#     ax[1].fill_between(L, Av - St, Av + St, color="lightgrey", alpha=0.5)
#     ax[1].set_ylabel(Element)
#     ax[1].legend()

#     plt.show()
    
#     return Tr


def Size(Data,Cluster=None):
    """
    Function to identify the size (in number of pixels) of a particular mineral phase (and/or cluster).

    Parameters:
    ----------
    Data: dict, required
        Python dictionary containing numpy arrays for each element.

    Cluster: string or float, required
        Define the cluster to be targeted

    Returns:
    ----------
    Data: dict,
        Python dictionary with a new entry titled 'Size'.

    """
    SizeAnalysis = Data['Cluster']
    SizeAnalysis = np.where(SizeAnalysis == Cluster, 1, 0)
    Data['Size'] = SizeAnalysis

    l,n = label(SizeAnalysis)
    f = find_objects(l)

    for i in range(0,len(f)):
        Data['Size'][f[i][0].start:f[i][0].stop,f[i][1].start:f[i][1].stop] = np.count_nonzero(SizeAnalysis[f[i][0].start:f[i][0].stop,f[i][1].start:f[i][1].stop])

    Data['Size'][np.where(SizeAnalysis==0)]=0

    return Data

def PointComp(DataEntry, subaxes = None, clicks = None, Element = None, Phase = None, size = None, text = None):
    Data = DataEntry.copy()

    if clicks is None:
        clicks = 1

    if size is None:
        size = 5

    print('Click ' + str(clicks) + ' times to select the point compositions you want to extract')
    pts = np.asarray(plt.ginput(clicks, timeout=-1))

    if Element is None:
        Element = list(Data.keys())
        if 'Cluster' in Element:
            Element.remove('Cluster')
        if 'Mineral' in Element:
            Element.remove('Mineral')

    Results = pd.DataFrame(data = np.zeros((clicks, len(Element))), columns = Element)

    for i in range(clicks):
        x, y = pts[i,:]
        for E in Element:
            if Phase is None:
                Data[E] = DataEntry[E].copy()
                Results[E].loc[i] = np.nanmean(Data[E][np.ix_(np.linspace(round(y) - (size-1)/2,round(y)+(size-1)/2,size).astype(int),np.linspace(round(x)-(size-1)/2,round(x)+(size-1)/2,size).astype(int))])
            if Phase is not None:
                Data[E] = DataEntry[E].copy()
                Data[E][np.where(Data['Mineral'] != Phase)] = np.nan
                Results[E].loc[i] = np.nanmean(Data[E][np.ix_(np.linspace(round(y) - (size-1)/2,round(y)+(size-1)/2,size).astype(int),np.linspace(round(x)-(size-1)/2,round(x)+(size-1)/2,size).astype(int))])

        if subaxes is not None:
            subaxes.fill([round(x)-(size-1)/2, round(x)-(size-1)/2, round(x)+(size-1)/2, round(x)+(size-1)/2, round(x)-(size-1)/2],[round(y)-(size-1)/2, round(y)+(size-1)/2, round(y)+(size-1)/2, round(y)-(size-1)/2, round(y)-(size-1)/2], color = [1,1,1], alpha = 0.8, zorder = 10)
            if text is not None:
                if text != 'count':
                    subaxes.text(round(x)+(size)/2, round(y)+(size)/2, text + '=' + str(round(Results[text].loc[i],2)), zorder = 11, c = 'r', fontsize = 14, fontweight = 'bold')
                else:
                    subaxes.text(round(x)+(size)/2, round(y)+(size)/2, str(i), zorder = 11, c = 'w', fontsize = 14, fontweight = 'bold', backgroundcolor = 'k')



    return Results

