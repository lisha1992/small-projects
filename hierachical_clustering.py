# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:13:50 2016

@author: ceciliaLee
"""

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


## settings
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

##### Generating sample data
#convert your data into a matrix X with n samples and m features
# generate two clusters: a with 100 points, b with 50:
np.random.seed(4711)  # for repeatability of this tutorial
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
X = np.concatenate((a, b),)
print X.shape  # 150 samples with 2 dimensions
plt.scatter(X[:,0], X[:,1])
plt.show()


###### Perform the Hoerarchical Clustering
# generate the linkage matrix
#linkage():Performs hierarchical/agglomerative clustering on the condensed distance matrix y.
#in each iteration will merge the two clusters which have the smallest distance according the selected method and metric.
Z = linkage(X, 'ward')#'ward' is one of the methods that can be used to calculate the distance between newly formed clusters.
print Z
#We can see that ach row of the resulting array has the format 
# [idx1, idx2, dist, sample_count].
print Z[0]
#In its first iteration the linkage algorithm decided to merge the two clusters 
#(original samples here) with indices 52 and 53, as they only had a distance of 0.04151. 
#This created a cluster with a total of 2 samples.



from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

c, coph_dists = cophenet(Z, pdist(X))
print c


print X[[33, 68, 62]]

idxs = [33, 68, 62]
plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1])  # plot all points
plt.scatter(X[idxs,0], X[idxs,1], c='r')  # plot interesting points in red again
plt.show()
idxs = [15, 69, 41]
plt.scatter(X[idxs,0], X[idxs,1], c='y')
plt.show()

######## Plotting a Dendrogram
# calculate full dendrogram
#horizontal lines are cluster merges
#vertical lines tell you which clusters/labels were 
#part of merge forming that new cluster
#heights of the horizontal lines tell you about the distance 
#that needed to be "bridged" to form the new cluster

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
# The height of that horizontal line tells you about the distance 
#at which this label was merged into another label or cluster
plt.show()

######### Dendrogram Truncation
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
   # show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)#show_contracted allows us to draw black dots at the heights of those previous cluster merges
plt.show()


########## annotating the distances inside the dendrogram
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata
    
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,  # useful in small plots so annotations don't overlap
)
plt.show()


######## Selecting a Distance Cut-Off aka Determining the Number of Clusters
# set cut-off to 50
max_d = 16  # max_d as in max_distance
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()


