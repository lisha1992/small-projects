# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 14:33:28 2016

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

def load_data(csvFile):
    data=pd.read_csv(csvFile)
    return data
    
def call_flights_record(csvFile):
    df=pd.read_csv(csvFile)
    arr=df.as_matrix()

    tma_idx=[]  # indeces of 'TMA_60'
    reg_set=set() # all regions
    for i in range(np.shape(arr)[0]):
        reg_set.add(arr[i][2])
        if arr[i][2]=='TMA_60':
            tma_idx.append(i)
  #  reg_list=list(reg_set)

## Merging all records of each landing into a list
    rec=[]      
    for i in range(1,len(tma_idx)):
        rec.append( arr[tma_idx[i-1]:tma_idx[i]] )
    rec.append(arr[tma_idx[-1]:])
    ser=pd.Series(rec)
    return ser
 #   print type(ser)
def read_rota_dataframe(csvFile='/Users/CeciliaLee/Dropbox/Intren/HKIA/2/ROTAprog_201512.csv'):
    rota=pd.read_csv(csvFile)
    rota.columns='Callsign','Reg', 'Runway', 'Enter_RW', 'Exit_RW', 'Occ_Time', 'Mode_3/A','RET','Airlines_c','RealOccTime','AcftType1','AcftType2','Code','HourlT','Stand','Apron','Airlines_n'
    return  rota
    
    
def correlate(data,ser,rota):
    flight_ix=[]
    flight_reg=[]
  #  flight_reg_set=set()
    fs=[]
    for i in range(len(data)):
        if data.ix[i]['cluster_index_part']==8:
            flight_ix.append(data.ix[i]['sample_data_index'])
  #  print flight
    for i in flight_ix:
        flight_reg.append(ser[i][0][1])  
    print len(flight_reg)
    flight_reg=list(set(flight_reg))
    print flight_reg
    print len(flight_reg)
    
    for line in flight_reg:
   #     print line
        for i in range(len(rota)):
            if rota.ix[i]['Reg']==line:
          #      print rota.ix[i]
                fs.append(rota.ix[i])
                
 #   clusterFile = open('/Users/CeciliaLee/Dropbox/Intren/HKIA/2/result/cluster_reg_10.txt','w')
  #  for reg in flight_reg:
  #      clusterFile.write("%s\n" % reg)
   # clusterFile.close()
    
    rota_File = open('/Users/CeciliaLee/Dropbox/Intren/HKIA/2/result/rota_8.txt','w')
    for f in fs:
        rota_File.write("%s\n" % f)
    rota_File.close()  
    



if __name__ == "__main__":
    csvFile='/Users/CeciliaLee/Dropbox/Intren/HKIA/2/'
    data=load_data(csvFile+'Clustering_results.csv')
    rota=read_rota_dataframe()
  #  print data['sample_data_index']
    ser=call_flights_record(csvFile+'group_data.csv')
#    print rota.ix[0]['Reg']
    correlate(data,ser,rota)
    
    