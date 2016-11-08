#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 16:17:26 2016

@author: ceciliaLee
"""
import numpy as np
import csv
import pandas as pd
from datetime import datetime
import time
import datetime
import os

from matplotlib import pyplot as plt
np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
from scipy.cluster.hierarchy import dendrogram, linkage


## load data into pandas as DataFrame
def read_dataframe(csvFile):
    data=pd.read_csv(csvFile)
    data.columns='Callsign','Reg', 'Region', 'Enter_Reg', 'Exit_Reg', 'Occ_Time', 'Mode_3/A'
    return  data
   

def data_transformation(data):
    new_occ=[]
    land_date=[]
    for tm in data['Occ_Time']:
        temp=int(tm.split(':')[0])*60+int(tm.split(':')[1])
        new_occ.append(temp)
    for ld in data['Enter_Reg']:
        temp=ld.split()[0]
        land_date.append(temp)
    data['Occ_Time_Sec']=new_occ
    data['Land_date']=land_date
    fileName='/Users/CeciliaLee/Dropbox/Intren/HKIA/2/transformed_data.csv'
  #  data.to_csv(fileName,mode='a',index=False) 
    return data

def group_landing(data):
    fileName='/Users/CeciliaLee/Dropbox/Intren/HKIA/2/group_data.csv'  
    with open(fileName, 'wb') as f:
        writer=csv.writer(f)
        header=['Callsign','Reg', 'Region', 'Enter_Reg', 'Exit_Reg', 'Occ_Time', 'Mode_3/A', 'Occ_Time_Sec','Land_date']
        writer.writerow(header)
    f.close()         
       

    pieces=dict(list(data.groupby('Reg')))
    for pic in pieces.keys():
        pieces_by_mode=dict(list(pieces[pic].groupby('Mode_3/A')))
        for cs in pieces_by_mode.keys():
            pieces_by_mode[cs]=pieces_by_mode[cs].sort(columns='Enter_Reg')#sort by enter time
            pieces_by_mode[cs].to_csv(fileName,mode='a',header=None, index=False) 



def pre_clustering(csvFile):
    df=pd.read_csv(csvFile)
    arr=df.as_matrix()

    tma_idx=[]  # indeces of 'TMA_60'
    reg_set=set() # all regions
    for i in range(np.shape(arr)[0]):
        reg_set.add(arr[i][2])
        if arr[i][2]=='TMA_60':
            tma_idx.append(i)
    reg_list=list(reg_set)

## Merging all records of each landing into a list
    rec=[]      
    for i in range(1,len(tma_idx)):
        rec.append( arr[tma_idx[i-1]:tma_idx[i]] )
    rec.append(arr[tma_idx[-1]:])
    ser=pd.Series(rec)

 ####### cleansing        
    d=[]
    for i in range(len(ser)):
        if len(ser[i])<=2:
            d.append(i+2)
 
    re_list=[]
    for j in range(len(ser)):
        re_dict={}
        for i in range(len(ser.ix[j])):
            re_dict.setdefault(ser.ix[j][i][2], []).append(ser.ix[j][i][7])
        for key in re_dict.keys():
            re_dict[key]=sum(re_dict[key])
        re_list.append(re_dict)
    dfs = pd.DataFrame(re_list,columns=reg_list)
    dfs.fillna(0, inplace=True)
  #  print dfs.columns

    return d,dfs
    


    
def clean(d):
    csv_route='/Users/CeciliaLee/Dropbox/Intren/HKIA/2/'
    df=pd.read_csv(csv_route+'regions_occupancy_time.csv')
    dfs=df.iloc[:,1:]
    deleted_row =[]+d
    for i in range(dfs.shape[0]):
        if dfs.ix[i]['07L_25R_PHYSICAL']==0:
            deleted_row.append(i+1)

    first_deleted_row = 0
    with open(csv_route+'regions_occupancy_time.csv', 'rt') as infile, open(csv_route+'regions_occupancy_time_updated.csv', 'wt') as outfile:
        outfile.writelines(row for row_num, row in enumerate(infile, first_deleted_row)
                        if row_num not in deleted_row)        


from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


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
    
    
def Hierarchical_cluster(csvFile):
    df=pd.read_csv(csvFile)
    data=df.as_matrix()
    data=data[:,1:]
    # generate the linkage matrix
    Z = linkage(data, 'ward')
    c, coph_dists = cophenet(Z, pdist(data))
    print c
     ## Plotting a Dendrogram
    # calculate full dendrogram
    plt.figure(figsize=(130, 50))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    
 #   dendrogram(
 #       Z,
   #     leaf_rotation=90.,  # rotates the x axis labels
   #     leaf_font_size=8.,  # font size for the x axis labels
  #  )
    
    fancy_dendrogram(
        
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=18,  # show only the last p merged clusters
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        show_leaf_counts=True, # numbers in brackets are counts
        show_contracted=True,  # to get a distribution impression in truncated branches
        max_d = 10000  # max_d as in max_distance
    )
    plt.savefig('/Users/CeciliaLee/Dropbox/Intren/HKIA/2/Dendrogram_Truncated_Tree(all).png')
    plt.show()  
    
    return c, Z
    
def Hierarchical_cluster_part(csvFile):
    df=pd.read_csv(csvFile)
    data=df.as_matrix()
    data=data[:,1:]
    # generate the linkage matrix
    Z = linkage(data, 'ward')
    c, coph_dists = cophenet(Z, pdist(data))
    print c
     ## Plotting a Dendrogram
    # calculate full dendrogram
    plt.figure(figsize=(140, 60))
    plt.title('Hierarchical Clustering Dendrogram(part)')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        Z,
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=2.,  # font size for the x axis labels
    )
   # fancy_dendrogram(
       # Z,
       # truncate_mode='lastp',  # show only the last p merged clusters
        #p=18,  # show only the last p merged clusters
       # leaf_rotation=90.,  # rotates the x axis labels
      #  leaf_font_size=8.,  # font size for the x axis labels
     #   show_leaf_counts=True, # numbers in brackets are counts
    #    show_contracted=True,  # to get a distribution impression in truncated branches
   #     max_d = 6000  # max_d as in max_distance
  #  )
    plt.savefig('/Users/CeciliaLee/Dropbox/Intren/HKIA/2/Dendrogram_Tree(part).png')
    plt.show()  
    
    return c, Z   

if __name__ == "__main__":
    import scipy.cluster.hierarchy as sch
    csvFile='/Users/CeciliaLee/Dropbox/Intren/HKIA/2/'
    data=read_dataframe(csvFile + 'merged.csv')
    trans_data=data_transformation(data)
    group_landing(trans_data)
    deleted_rows,occ_time_dfs=pre_clustering(csvFile+'group_data.csv')
   # occ_time_dfs.to_csv(csvFile+'regions_occupancy_time.csv')
  #  clean(deleted_rows)

    c_part, Z_part=Hierarchical_cluster_part(csvFile+'regions_occupancy_time_part.csv')
   # max_d=40000
  #  clusters_all=sch.fcluster(Z_all, max_d, criterion='distance')
  #  myfile_part = open('/Users/CeciliaLee/Dropbox/Intren/HKIA/2/clustering_result_part.txt','w')
  # for z in Z_part:
   #     myfile_part.write("%s\n" % z)
   # myfile_part.close()

 #   c_all, Z_all=Hierarchical_cluster(csvFile+'regions_occupancy_time_updated.csv')
 #   max_d=10000
  #  clusters_all=sch.fcluster(Z_all, max_d, criterion='distance')
  #  myfile_all = open('/Users/CeciliaLee/Dropbox/Intren/HKIA/2/cluster_indices_all.txt','w')
   # for cluster in clusters_all:
   #     myfile_all.write("%s\n" % cluster)
   # myfile_all.close()
     
  #  c_part, Z_part=Hierarchical_cluster_part(csvFile+'regions_occupancy_time_part.csv')
  #  max_d=6000
  #  clusters_part=sch.fcluster(Z_part, max_d, criterion='distance')
  #  myfile_part = open('/Users/CeciliaLee/Dropbox/Intren/HKIA/2/clustering_result_part.txt','w')
   # for cluster in clusters_part:
   #     myfile_part.write("%s\n" % cluster)
  #  myfile_part.close()
    
    

  
