# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 21:00:21 2016

@author: ceciliaLee
"""

import numpy as np
import math
import copy

## loads a .txt file containing lines of tab-delimited floats into a list
# order when load data: dataMat=np.mat(ks.load_dataSet(fileName))
def load_dataSet(fileName):
    dataMat=[]
    dataFile=open(fileName)
    for line in dataFile.readlines():
        curLine=line.strip().split('\t') # eliminate BLANK and split lines by 't'
        fitLine=map(float, curLine) # #map all elements to float()
        dataMat.append(fitLine)
    return dataMat # return a matrix of list with many lists

## Calculate Euclidean distance between 2 vectors
def Euclidean_dist(vectorA,vectorB):
    return math.sqrt(sum(np.power(vectorA - vectorB, 2)))
    
## Create a set of k random centroids for a give dataset
def select_randomCentroid(dataSet, k):
    n=np.shape(dataSet)[1]
    centroids= np.mat(np.zeros((k,n)))#create centroid matrix/integrate input elements into matrix
    ## Create cluster centroids,  within bounds of each dimension
    for j in range(n):
        minJ=min(dataSet[:,j])
        rangeJ=float(max(dataSet[:,j])-minJ)
        centroids[:,j]=np.mat(minJ + rangeJ * np.random.rand(k,1))
    return centroids
        
## implement k-means clustering algorithm
# clusterAssment:cluster assignment matrix, 
# contains 2 columns:the index of the cluster and the error(distance from cluster centroid to current point)
def k_means(dataSet,k,dist_measure=Euclidean_dist,create_centroid=select_randomCentroid):
    m=np.shape(dataSet)[0]  # number of rows(samples/points)
    clusterAssment=np.mat(np.zeros((m,2))) ##the assignment of points, [index of cluster,error]
    centroids=create_centroid(dataSet,k) ## create centroids
    clusterChanged=True  # flag parameter: if ==true: continue itteration
    while clusterChanged:
        clusterChanged=False
        for i in range(m): #for each data point assign it to the closest centroid
            min_dist=float("inf") 
            min_index=-1
            ## find the closest centroid of each point
            for j in range(k):
                dist_ij=dist_measure(centroids[j,:],dataSet[i,:]) 
                if dist_ij < min_dist:
                    min_dist=dist_ij
                    min_index=j
            if clusterAssment[i,0]!=min_index: # if the assignment of a certain point is changed: update the flag of clusterChanged
                clusterChanged=True
            clusterAssment[i,:]=min_index, min_dist**2
        print centroids
        ## recalculate and update centroids
        for centroid in range(k):
           # get all the point in this cluster
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==centroid)[0]] # np.nonzero():Return the indices of the elements that are non-zero.
            centroids[centroid,:]=np.mean(ptsInClust,axis=0)# calculate mean of all points (axis=0:mean calculation down the columns)
    return centroids, clusterAssment

def bi_kmeans(dataSet,k,dist_measure=Euclidean_dist):
    m = np.shape(dataSet)[0] # number of points
    # create a matrix(clusterAssment) to store the cluster assignment and squared error for each point in the dataset
    clusterAssment = np.mat(np.zeros((m,2))) # assignment of each point 
    
    ### Initially create one cluster
    centroid0 = np.mean(dataSet, axis=0).tolist()[0] #numpy.ndarray.tolist(): Return a copy of the array data as a (nested) Python list.
    centList =[centroid0] #create a list with one centroid 
    #calculate initial Error between that point and the centroid
    for j in range(m): 
        clusterAssment[j,1] = dist_measure(np.mat(centroid0), dataSet[j,:])**2  
        
    ## while loop, continue spliting clusters until you have the desired number of clusters.   
        # iterate over all the clusters and find the best cluster to split: according to the SSE after each split
    while (len(centList)<k):
        lowestSSE=float('inf') # initialize the lowest SSE to infinity
        
        ### Try splitting every cluster
        for i in range(len(centList)):  #start looping over each cluster in the centList cluster list.
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss =  k_means(ptsInCurrCluster, 2, dist_measure) # split the cluster i
            #k-means algorithm gives you two new centroids as well as the squared error for each of those centroids.
            sseSplit = sum(splitClustAss[:,1]) #compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit            
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit            
        ###### Update the cluster assignments
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        
        #these new cluster assignments are updated and the new centroid is appended to centList.
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return np.mat(centList), clusterAssment            

import urllib
import json

## gets a dictionary of values from Yahoo
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J' #JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    # urllib.urlencode():pack up your dictionary in a format you can pass on in a URL
    url_params = urllib.urlencode(params) #output format: flag=J&appid=aaa0VN6k&location=stAddress city
    yahooApi = apiStem + url_params      #print url_params
    print yahooApi
    c=urllib.urlopen(yahooApi)
    #return value is in JSON format
    return json.loads(c.read()) # decode it into a dictionary.    

# geoGrab() gets a dictionary of values from Yahoo, 
# while massPlaceFind() automates this and saves the relevant information to a file.
from time import sleep
def find_massPlace(fileName):
    fw=open('/Users/CeciliaLee/Desktop/places.txt','w') #opens a tab-delimited text file 
    for line in open(fileName).readlines():
        line = line.strip()# eliminate blank space
        lineArr = line.split('\t') # split by 't'
        retDict = geoGrab(lineArr[1], lineArr[2]) # gets the second and third fields fed to geoGrab()
        if retDict['ResultSet']['Error'] == 0: #checked to see if there are any errors.
            lat = float(retDict['ResultSet']['Results'][0]['latitude']) # lattitude
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])# longitude
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print "error fetching"
        # call sleelp() ensure that you don’t make too many API calls too quickly.
        sleep(1) #the sleep function is called to delay massPlaceFind() for one second.
    fw.close()
    
## Clustering geographic coordinates
###### Spherical distance measure and cluster-plotting functions

#distance metric for two points on the earth’s surface.(Spherical distance)
def distSLC(vecA, vecB): #Spherical Law of Cosines
    #sin() and cos() take radians as inputs. So convert from degrees to radians by dividing by 180 and multiplying by pi.
    a = np.sin(vecA[0,1]* math.pi/180) * np.sin(vecB[0,1]* math.pi/180)
    b = np.cos(vecA[0,1]* math.pi/180) * np.cos(vecB[0,1]*math.pi/180) * \
                      np.cos(math.pi * (vecB[0,0]-vecA[0,0]) /180)
    #returns the distance in miles for two points on the earth’s surface.    
    return np.arccos(a + b)*6371.0 #pi is imported with numpy    
    # numpy.arccos(): Trigonometric inverse cosine, element-wise.

import matplotlib
import matplotlib.pyplot as plt
# clusterClubs():wraps up parsing a text file, clustering, and plotting. 
def clusterClubs(numClust=5):# takes one input that’s the number of clusters you’d like to create.
    datList = []
    for line in open('/Users/Cecilialee/Desktop/places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])]) # extract the latitude and longtitude
    datMat = np.mat(datList) # convert into matrix
    myCentroids, clustAssing = bi_kmeans(datMat, numClust,dist_measure=distSLC)## calculate centroids and clusters assignment of each point
    ## Visualization   /plot clusters and points     
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('/Users/ceciliaLee/Desktop/Portland.png')# create a matrix from an image
    ax0.imshow(imgP)# plot the matrix
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()
    

if __name__ == "__main__":
    fileName='/Users/ceciliaLee/Desktop/testSet2.txt'
    dataMat=np.mat(load_dataSet(fileName)) 
  #  select_randomCentroid(dataMat,4)
  #  centList,clusterAssigned=bi_kmeans(dataMat,3)
  #  geoResults=geoGrab('1 VA Center', 'Augusta, ME')
    clusterClubs(5)
#    print 'centList:'
 #   print centList
#    print Euclidean_dist(dataMat[0],dataMat[1])
 #   print dataMat
    
    
    
        
    