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

def Section(DataEntry,a,Element,Resolution,Cluster=None):
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
    b[0].plot(Tr['Distance'],Tr[Element],'ow',markerfacecolor=[0.5,0.5,0.5])
    b[0].set_ylabel(Element)

    b[1].plot(L,Av,'-k',linewidth=2,zorder=1)
    b[1].fill(np.array([L,np.flip(L)]).flatten(),np.array([Av-St,np.flip(Av)+np.flip(St)]).flatten(),color="lightgrey")
    b[1].set_ylabel(Element)

    plt.show()

    return Tr

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

