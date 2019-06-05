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


#subC0 = np.random.choice(lab0, 800)
#subC1 = np.random.choice(lab1, 800)
#subC2 = np.random.choice(lab2, 800)
#subC3 = np.random.choice(lab3, 800)
#subC4 = np.random.choice(lab4, 800)


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical

y_train = to_categorical(targ_train)
y_test = to_categorical(targ_test)

model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(187,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ecg_train, y_train, epochs=20)

print("Evaluation: ")
mse, acc = model.evaluate(ecg_test, y_test)
print('mean_squared_error :', mse)
print('accuracy:', acc)


