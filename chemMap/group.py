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

    if Plot is not None:
        X=np.linspace(0,len(Data[list(Oxide)[0]][0,:])-1,len(Data[list(Oxide)[0]][0,:]))
        Y=np.linspace(0,len(Data[list(Oxide)[0]][:,0])-1,len(Data[list(Oxide)[0]][:,0]))
        X, Y = np.meshgrid(X, Y)

        f, a = plt.subplots(1, 1, figsize=(15,15*Y[-1,-1]/X[-1,-1]))
        a.axis('off')
        a.set_aspect(len(X)/len(Y))
        if Cluster is None:
            z2=a.contourf(X, Y, Data['Cluster'], 20, cmap='viridis',zorder=0)
        if Cluster is not None:
            z2=a.contourf(X, Y, Data['Cluster_'+str(Cluster)], 20, cmap='viridis',zorder=0)

    if Name is not None:
        if Name==True:
            Data, Nm = NameCluster(Data, Return = True)
        else:
            Data, Nm = NameCluster(Data,Name = Name, Return = True)

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