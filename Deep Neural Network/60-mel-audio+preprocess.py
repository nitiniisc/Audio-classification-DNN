
# coding: utf-8

# In[17]:


from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv 
import pandas as pd
import glob
from scipy.io.wavfile import read
import wave, os, glob
import numpy as np
import sys
import librosa
import pickle


# In[18]:



data = pd.read_csv("../DCASE dataset/meta.txt", sep="\t" , header = None)
print(len(data))
#data.columns = ["a", "b", "c"]
label_map = {'beach':0, 'bus':1, 'cafe/restaurant':2, 'car':3, 'city_center':4, 'forest_path':5, 'grocery_store':6, 
            'home':7, 'library':8, 'metro_station':9, 'office':10,'park':11, 'residential_area':12, 'train':13, 'tram':14}
print("labeling done")


# In[19]:


import torch.optim as optim
from sklearn.utils import shuffle
new_data = shuffle(data)
#print(data)
batch_size=512


# In[20]:


#path2="/home/nitin/audio_project/myworks/DCASE dataset"
path = "./"


# In[21]:



wavlabel=[]
wavpath=[]
for row in new_data[0]:
    #print(row)
    wavpath.append(row)
for row1 in new_data[1]:
    #print(row1)
    wavlabel.append(row1)
print(len(wavpath))        


# In[27]:


print(len(wavpath))
train=[]
for i in range(5):
    #data1[1][i]
    wavfil=wavpath[i]
    #print(wavfil)
    label=wavlabel[i]
    #print(label)
    #print([label_map[label]])
    filename=glob.glob(os.path.join(path, wavfil))
    y, sr = librosa.load((os.path.join(path, wavfil)))
    x=librosa.feature.melspectrogram(y=y, sr=sr,n_mels=60)
    #log_S = librosa.logamplitude(S, ref_power=np.max)
    #print(x.shape)
    if(i%100==0):
        print(i)
    train.append(x)
    #print(i)
print("labeled")


# In[23]:


print(len(wavpath))
print(len(train))
labell=[]
for i in range(len(wavlabel)):
    #print(wavlabel[i])
    #wavfil=data1[0][i]
    #print(wavfil)
    label2=wavlabel[i]
    seq=np.array(label_map[label2])
    x=np.zeros(15)
    x[seq]=1
    y=np.reshape(x,(1,15))
    labell.append(y)
print(len(labell))
print(len(train))


# In[24]:


with open('./60-mel_train.pkl', 'wb') as f:
    pickle.dump(train, f)


# In[25]:


with open('./60-mel_train_label.pkl', 'wb') as f:
    pickle.dump(labell, f)


# In[26]:


with open('./60-mel_train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('./60-mel_train_label.pkl', 'rb') as f:
    labell = pickle.load(f)    


# In[28]:


test_data = pd.read_csv("../DCASE dataset/test/meta.txt", sep="\t" , header = None)
print(len(test_data))
test = []
#path2="/home/nitin/audio_project/myworks/DCASE dataset"
path = "./test/"


# In[29]:


for i in range(len(test_data)):
    wavfil=test_data[0][i]
    #print(wavfil)
    label=test_data[1][i]
    #print(label)
    #print([label_map[label]])
    filename=glob.glob(os.path.join(path, wavfil))
    y, sr = librosa.load((os.path.join(path, wavfil)))
    x=librosa.feature.melspectrogram(y=y, sr=sr,n_mels=60)
    if(i%100==0):
        print(i)
    test.append(x)
    #print(x.shape)
print("labeled")


# In[30]:


test_label=[]
for i in range(len(test_data)):
    wavfil=test_data[0][i]
    #print(wavfil)
    label=test_data[1][i]
    seq=np.array(label_map[label])
    x=np.zeros(15)
    x[seq]=1
    y=np.reshape(x,(1,15))
    test_label.append(y)
#print(labell)    


# In[31]:


with open('./60-mel_test.pkl', 'wb') as f:
    pickle.dump(test, f)


# In[32]:


with open('./60-mel_test_label.pkl', 'wb') as f:
    pickle.dump(test_label, f)


# In[33]:


with open('./60-mel_test.pkl', 'rb') as f:
    test = pickle.load(f)
with open('./60-mel_test_label.pkl', 'rb') as f:
    test_label = pickle.load(f)    

