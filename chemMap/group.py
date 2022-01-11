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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from .elements import element_properties
from sklearn.decomposition import PCA

def Cluster(Data,Oxide,number_of_clusters, Plot = None, Cluster = None, Name = None, ShowComp = None):
    """
    k-mean cluster analysis.

    Parameters:
    ----------
    Data: dict, required
        Python dictionary containing numpy arrays for each element.

    Oxide: list, required
        list of the oxides or elements (depending on the nature of the dict variable) that are to be involved in the cluster analysis.

    number_of_clusters: int, required
        number of clusters, simple as.

    Plot: bool, optional
        If True, a map showing the relative location of the different clusters will be shown.

    Cluster: string, optional
        If not None, cluster analysis will be performed on a single phase (i.e., to identify differen plagioclase subpopulations).

    Name: bool or list, optional
        If True, the user will be prompted to input a series of three letter identification codes to name each cluster. If type(Name) = list, the string variables in the list will be used to name the clusters.

    ShowComp: bool, optional
        If True, the composition of each phase will be printed.

    Returns:
    ----------
    Data: dict
        Python dictionary containing an additional entry detailing the cluster that each pixel belongs to.

    """

    for k in Oxide:
        Data[k] = np.nan_to_num(Data[k], nan = 0.0)

    Dat=[]
    i=0
    for ox in Oxide:
        if Cluster is None:
            A = Data[ox].flatten()
            A = (A - np.nanmin(A))/(np.nanmax(A) - np.nanmin(A))
            if i==0:
                Dat = A
                i=1
            else:
                Dat = np.vstack((Dat, A))
        if Cluster is not None:
            A = Data[ox][np.where(Data['Cluster']==Cluster)].flatten()
            A = (A - np.nanmin(A))/(np.nanmax(A) - np.nanmin(A))
            if i==0:
                Dat = A
                i=1
            else:
                Dat = np.vstack((Dat, A))

    if Cluster is not None:
        kmeans=KMeans(n_clusters=number_of_clusters,random_state=0).fit(Dat.T)
        B = np.zeros(len(Data['Cluster'].flatten()))
        B[np.where(Data['Cluster'].flatten() == Cluster)] = kmeans.labels_+1
        Data['Cluster_' + str(Cluster)]=B.reshape(np.shape(Data[list(Oxide)[0]]))

    if Cluster is None:
        kmeans=KMeans(n_clusters=number_of_clusters,random_state=0).fit(Dat.T)
        Data['Cluster']=kmeans.labels_.reshape(np.shape(Data[list(Oxide)[0]]))

    if Name is not None:
        if Name==True:
            Data, Nm = NameCluster(Data, Return = True)
        else:
            Data, Nm = NameCluster(Data,Name = Name, Return = True)

    if Plot is not None:
        if Name is not None:
            if Cluster is None:
                a = PlotCluster(Data, number_of_clusters, Name = Nm)
            else:
                a = PlotCluster(Data, number_of_clusters, Name = Nm, Cluster=Cluster)
        else:
            if Cluster is None:
                a = PlotCluster(Data, number_of_clusters)
            else:
                a = PlotCluster(Data, number_of_clusters, Cluster = Cluster)

    if ShowComp is not None:
        for i in range(0,number_of_clusters):
            if Cluster is None:
                for ox in Oxide:
                    if Name is None:
                        print('Cluster ' + str(i) + ',' + ox + ' = '+ str(round(np.nanmean(np.nanmean(Data[ox][np.where(Data['Cluster']==i)])),2)))
                    else:
                        print('Cluster ' + Nm[i] + ',' + ox + ' = '+ str(round(np.nanmean(np.nanmean(Data[ox][np.where(Data['Cluster']==Nm[i])])),2)))

            if Cluster is not None:
                for ox in Oxide:
                    print('Cluster_' + str(Cluster) + ' ' + str(i) + ',' + ox + ' = '+ str(round(np.nanmean(np.nanmean(Data[ox][np.where(Data['Cluster_' + str(Cluster)]==i)])),2)))

    return Data

def PlotCluster(Data, number_of_clusters, Name = None, Cluster = None):
    """
    Plot the different clusters.

    Parameters:
    ----------
    Data: dict, required
        Python dictionary containing numpy arrays for each element and a numpy array containing the cluster information.

    number_of_clusters: float, required
        Number of clusters specified previously.

    Name: list, optional
        If no value entered, the code will prompt the user to enter a three-letter identification code for each cluster. If a list is entered, the values in this list will be used as the cluster names.

    Cluster: string or float, optional

    Returns:
    ----------
    Data: a,
        matplotlib subplot

    """
    Oxide = list(Data.keys())
    X_1=np.linspace(0,np.shape(Data[Oxide[0]])[1]-1, np.shape(Data[Oxide[0]])[1])
    Y_1=np.linspace(0,np.shape(Data[Oxide[0]])[0]-1, np.shape(Data[Oxide[0]])[0])
    X, Y = np.meshgrid(X_1, Y_1)

    f, a = plt.subplots(1, 1, figsize=(15*X[-1,-1]/X[-1,-1],15*Y[-1,-1]/X[-1,-1]))#(15,15*Y[-1,-1]/X[-1,-1]))
    a.axis('off')
    a.set_aspect('equal')

    a.set_xlim([0,len(X_1)])
    a.set_ylim([0,len(Y_1)])

    H = f.get_figheight()
    W = f.get_figwidth()
    dpi = f.get_dpi()

    marker = (dpi*W)/(len(X_1))

    cmap = plt.get_cmap('viridis')
    cmap = cmap(np.linspace(0,1,number_of_clusters))

    if Cluster is None:
        for i in range(number_of_clusters):
            if Name is None:
                a.plot(X[np.where(Data['Cluster'] == i)],Y[np.where(Data['Cluster'] == i)],'s', color = cmap[i], markeredgecolor = cmap[i], markersize = marker, label = "Cluster" + str(i), markeredgewidth = 0)
            else:
                if Name[i] == 'nan':
                    a.plot(X[np.where(Data['Cluster'] == Name[i])],Y[np.where(Data['Cluster'] == Name[i])],'s', color = 'k', markeredgecolor = 'k', markersize = marker, label = Name[i], markeredgewidth = 0)
                else:
                    a.plot(X[np.where(Data['Cluster'] == Name[i])],Y[np.where(Data['Cluster'] == Name[i])],'s', color = cmap[i], markeredgecolor = cmap[i], markersize = marker, label = Name[i], markeredgewidth = 0)

        a.legend(markerscale = 10)

    if Cluster is not None:
        for i in range(number_of_clusters):
            if Name is None:
                a.plot(X[np.where(Data['Cluster_'+str(Cluster)] == i)],Y[np.where(Data['Cluster_'+str(Cluster)] == i)],'s', color = cmap[i], markeredgecolor = 'k', markersize = marker, markeredgewidth = 0)



    plt.show()

    return a

def NameCluster(Data,Name=None, Return = None):
    """
    Name each of the clusters.

    Parameters:
    ----------
    Data: dict, required
        Python dictionary containing numpy arrays for each element and a numpy array containing the cluster information.

    Name: list, optional
        If no value entered, the code will prompt the user to enter a three-letter identification code for each cluster. If a list is entered, the values in this list will be used as the cluster names.

    Returns:
    ----------
    Data: dict,
        Python dictionary where the Data['Cluster'] numpy array has been converted from integers to strings.

    """
    Oxide = list(Data.keys())
    A=Data['Cluster'].flatten()
    L=np.array(['nan']*len(A))
    Nm=np.array(['nan']*(np.max(Data['Cluster'])+1))
    for i in range(0,np.max(Data['Cluster'])+1):
        if Name is None:
            for ox in Oxide:
                print('Cluster ' + str(i) + ',' + ox + ' = '+ str(round(np.nanmean(np.nanmean(Data[ox][np.where(Data['Cluster']==i)])),2)))
            N=input("Type a three letter identification code for this cluster: \n")
        else:
            N=Name[i]

        L[np.where(A==i)[0]]=N
        Nm[i]=N

    Data['Cluster']=L.reshape(np.shape(Data[Oxide[0]]))

    if Return is not None:
        return Data, Nm

    else:
        return Data


def PrincipalComponentAnalysis(DataEntry, Oxide, Cluster = None, number_of_components = None):
    """
    Perform principal component analysis.

    Parameters:
    ----------
    DataEntry: dict, required
        Python dictionary containing numpy arrays for each element and a numpy array containing the cluster information.

    Oxide: list, required
        Elements or Oxides to be included in the PCA analysis.

    Cluster: string, optional
        Cluster to be used if the idea is to only focus on one sample.

    number_of_components: float, optional
        number of components for PCA. default is 2.

    Returns:
    ----------
    Data: dict,
        Python dictionary

    """

    Data = DataEntry.copy()

    if number_of_components is None:
        number_of_components = 2

    for k in Oxide:
        Data[k] = np.nan_to_num(Data[k], nan = 0.0)

    minmaxscaler = MinMaxScaler()

    Dat=[]
    i=0
    for ox in Oxide:
        if Cluster is None:
            A = MinMaxScaler().fit_transform(Data[ox]).flatten()
            if i==0:
                Dat = A
                i=1
            else:
                Dat = np.vstack((Dat, A))
        if Cluster is not None:
            A = MinMaxScaler().fit_transform(Data[ox][np.where(Data['Cluster'] == Cluster)].reshape(len(Data['Cluster'][np.where(Data['Cluster'] == Cluster)]),1)).flatten()
            if i==0:
                Dat = A
                i=1
            else:
                Dat = np.vstack((Dat, A))

    pca = PCA(n_components = number_of_components)
    principalComponents = pca.fit_transform(Dat.T)

    for i in range(number_of_components):
        if Cluster is None:
            B = principalComponents[:,i]

        if Cluster is not None:
            B = np.zeros(len(Data['Cluster'].flatten()))
            B[np.where(Data['Cluster'].flatten() == Cluster)] = principalComponents[:,0]
            if type(Cluster) == type('str'):
                Data[Cluster + '_PCA ' + str(i+1)] = B.reshape(np.shape(Data['Cluster']))
            else:
                Data[str(Cluster) + '_PCA ' + str(i+1)] = B.reshape(np.shape(Data['Cluster']))

    return Data