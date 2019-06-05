# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:54:58 2019

@author: senthilku
"""

import pandas as pd
import numpy as np

data_train=pd.read_csv('mitbih_train.csv',header=None)
data_test=pd.read_csv('mitbih_test.csv',header=None)
targ_train=data_train.iloc[:,187]
targ_test=data_test.iloc[:,187]

ecg_train=data_train.iloc[:,0:187]
ecg_test=data_test.iloc[:,0:187]

samp=ecg_train.iloc[0,:]
import matplotlib.pyplot as plt
import seaborn as sb
plt.figure()
plt.plot(samp)

cata=pd.value_counts(targ_train).reset_index()
lab=np.array(cata)
plt.figure()
plt.bar(lab[:,0],lab[:,1])

lab0=np.array(np.where(targ_train==0)).flatten()
lab1=np.array(np.where(targ_train==1)).flatten()
lab2=np.array(np.where(targ_train==2)).flatten()
lab3=np.array(np.where(targ_train==3)).flatten()
lab4=np.array(np.where(targ_train==4)).flatten()

t_ms = np.arange(0, 187)*8/1000
plt.figure()
plt.plot(t_ms,ecg_train.iloc[lab0[5],:],label='cat0')
plt.plot(t_ms,ecg_train.iloc[lab1[5],:],label='cat1')
plt.plot(t_ms,ecg_train.iloc[lab2[5],:],label='cat2')
plt.plot(t_ms,ecg_train.iloc[lab3[5],:],label='cat3')
plt.plot(t_ms,ecg_train.iloc[lab4[5],:],label='cat4')
plt.legend()
plt.title("1-beat ECG for every category", fontsize=20)
plt.ylabel("Amplitude", fontsize=15)
plt.xlabel("Time (ms)", fontsize=15)

#Feature Extraction
from scipy.signal import find_peaks,peak_widths,peak_prominences,argrelmin,argrelmax,argrelextrema,spectrogram

f1_peak=[]
f2_peak_height=[]
f3_prom=[]
f4_wid=[]
f5_rmin=[]
f6_rmax=[]
f7_spec=[]

for i in range(len(ecg_train)):
    sig1=np.array(ecg_train.iloc[i,:])
    peaks, hei = find_peaks(sig1,height=0)
    height=hei['peak_heights']
    f1_peak.append(peaks)
    f2_peak_height.append(height)
    prominences = peak_prominences(sig1, peaks)[0]
    f3_prom.append(prominences)
    wid=peak_widths(sig1,peaks)[0]
    f4_wid.append(wid)
    sig_min=argrelmin(sig1)[0]
    f5_rmin.append(sig_min)
    sig_max=argrelmax(sig1)[0]
    f6_rmax.append(sig_max)
    #plt.plot(sig1)
    #plt.plot(peaks, sig1[peaks], "x")
    f,t,sx=spectrogram(sig1,60)
    f7_spec.append(sx)
    
f1_len = max([len(x) for x in f1_peak])
f2_len = max([len(x) for x in f2_peak_height])
f3_len = max([len(x) for x in f3_prom])
f4_len = max([len(x) for x in f4_wid])
f7_len = max([len(x) for x in f7_spec])

F1_peak=np.zeros((len(f1_peak),f1_len),dtype=float)
for i in range(len(f1_peak)):
#i=0
    z_c=f1_len-len(f1_peak[i])
    F1_peak[i,:]=np.pad(f1_peak[i],(0,z_c),'constant')
    #if (len(x)!=f1_len):
        
F2_peak_height=np.zeros((len(f2_peak_height),f2_len),dtype=float)
for i in range(len(f2_peak_height)):
#i=0
    z_c=f2_len-len(f2_peak_height[i])
    F2_peak_height[i,:]=np.pad(f2_peak_height[i],(0,z_c),'constant')
    #if (len(x)!=f1_len):     

F3_prom=np.zeros((len(f3_prom),f3_len),dtype=float)
for i in range(len(f3_prom)):
#i=0
    z_c=f3_len-len(f3_prom[i])
    F3_prom[i,:]=np.pad(f3_prom[i],(0,z_c),'constant')
    #if (len(x)!=f1_len):     

F4_wid=np.zeros((len(f4_wid),f4_len),dtype=float)
for i in range(len(f4_wid)):
#i=0
    z_c=f4_len-len(f4_wid[i])
    F4_wid[i,:]=np.pad(f4_wid[i],(0,z_c),'constant')
    #if (len(x)!=f1_len): 
    
F7_spec=np.squeeze(np.array(f7_spec))

Feat=np.concatenate((F1_peak,F2_peak_height,F3_prom,F4_wid,F7_spec),axis=1)

from sklearn.model_selection import train_test_split
[xtrain,xtest,ytrain,ytest]=train_test_split(Feat,targ_train,test_size=0.3,random_state=42)

from sklearn.ensemble import RandomForestClassifier
ecg_model=RandomForestClassifier()
ecg_model.fit(xtrain,ytrain)
ypred=ecg_model.predict(xtest)

from sklearn.metrics import accuracy_score, confusion_matrix
acc=accuracy_score(ytest,ypred)
cm=confusion_matrix(ytest,ypred)





