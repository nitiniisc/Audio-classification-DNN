
# coding: utf-8

# In[1]:


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


# In[2]:


"""
data = pd.read_csv("../DCASE dataset/meta.txt", sep="\t" , header = None)
print(len(data))
#data.columns = ["a", "b", "c"]
label_map = {'beach':0, 'bus':1, 'cafe/restaurant':2, 'car':3, 'city_center':4, 'forest_path':5, 'grocery_store':6, 
            'home':7, 'library':8, 'metro_station':9, 'office':10,'park':11, 'residential_area':12, 'train':13, 'tram':14}
print("labeling done")
"""


# In[3]:


#print(data)
import torch.optim as optim
from sklearn.utils import shuffle
#new_data = shuffle(data)
#print(data)
batch_size=512


# In[4]:


#path2="/home/nitin/audio_project/myworks/DCASE dataset"
path = "./"


# In[5]:


"""
wavlabel=[]
wavpath=[]
for row in new_data[0]:
    #print(row)
    wavpath.append(row)
for row1 in new_data[1]:
    #print(row1)
    wavlabel.append(row1)
print(len(wavpath))        
"""


# In[6]:


"""
print(len(wavpath))
train=[]
for i in range(len(wavpath)):
    #data1[1][i]
    wavfil=wavpath[i]
    #print(wavfil)
    label=wavlabel[i]
    #print(label)
    #print([label_map[label]])
    filename=glob.glob(os.path.join(path, wavfil))
    y, sr = librosa.load((os.path.join(path, wavfil)))
    x=librosa.feature.melspectrogram(y=y, sr=sr,n_mels=60)
    #print(x.shape)
    if(i%100==0):
        print(i)
    train.append(x)
    #print(i)
print("labeled")
"""


# In[7]:


'''
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
'''


# In[8]:


"""
with open('./60-mel_train.pkl', 'wb') as f:
    pickle.dump(train, f)
"""    


# In[9]:


"""
with open('./60-mel_train_label.pkl', 'wb') as f:
    pickle.dump(labell, f)
"""    


# In[10]:


with open('./60-mel_train.pkl', 'rb') as f:
    train = pickle.load(f)
with open('./60-mel_train_label.pkl', 'rb') as f:
    labell = pickle.load(f)    


# In[11]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(25860, 10000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, 15)
        self.drop= nn.Dropout(p=0.2)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x) 
        x = self.sig(self.fc3(x))
        #x = self.sig(x)
        return x


# In[12]:


net = Net().cuda()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


# In[13]:


nput=train[0]
print(nput.shape)
inputss=np.reshape(nput,[1,60*431])
labels=labell[0]
#print("label---",labels)
inputs = Variable(torch.FloatTensor(torch.from_numpy(inputss).float())).cuda()
labels = Variable(torch.FloatTensor(torch.from_numpy(labels).float())).cuda()
optimizer.zero_grad()
outputs = net(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
print(loss)


# In[14]:


net.train()
for epoch in range(500):  # loop over the dataset multiple times
    for i in range(0,len(train), batch_size):
        optimizer.zero_grad()
        #print(i)
        x_batch = train[i:i+batch_size]
        x = np.reshape(x_batch,[-1,60*431])
        y_batch = labell[i:i+batch_size]
        y=np.array(y_batch)
        #print(len(x_batch))
        #print(len(y_batch))
        inputs = Variable(torch.FloatTensor(torch.from_numpy(x).float())).cuda()
        labels = Variable(torch.FloatTensor(torch.from_numpy(y).float())).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss,epoch)
        
print('Finished Training')


# In[15]:


"""
test_data = pd.read_csv("../DCASE dataset/test/meta.txt", sep="\t" , header = None)
print(len(test_data))
test = []
#path2="/home/nitin/audio_project/myworks/DCASE dataset"
path = "./test/"
"""


# In[16]:


"""
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
"""


# In[17]:


#print(test[0].shape)


# In[18]:


""""
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
"""


# In[19]:


""""
with open('./60-mel_test.pkl', 'wb') as f:
    pickle.dump(test, f)
"""    


# In[20]:


"""
with open('./60-mel_train_label.pkl', 'wb') as f:
    pickle.dump(test_l, f)
"""    


# In[21]:


with open('./60-mel_test.pkl', 'rb') as f:
    test = pickle.load(f)
with open('./60-mel_test_label.pkl', 'rb') as f:
    test_label = pickle.load(f)    


# In[22]:


correct=0
for i in range(1):
    x_batch = train[i]
    x = np.reshape(x_batch,[-1,60*431])
    y_batch = labell[i]
    #print(y_batch)
    y=np.array(y_batch)
    inputs = Variable(torch.FloatTensor(torch.from_numpy(x).float())).cuda()
    #labels = Variable(torch.FloatTensor(torch.from_numpy(y).float())).cuda()
    outputs = net(inputs)
    #print(outputs)
    values, indices = torch.max(outputs, 0)
    #print(values)
    #print(indices)
    #print(i)


# In[23]:


net.eval()
correct=0
for i in range(len(train)):
    x_batch = train[i]
    x = np.reshape(x_batch,[-1,60*431])
    y_batch = labell[i]
    #print(y_batch)
    y=np.array(y_batch)
    inputs = Variable(torch.FloatTensor(torch.from_numpy(x).float())).cuda()
    labels = Variable(torch.FloatTensor(torch.from_numpy(y).float())).cuda()
    outputs = net(inputs)
    m = nn.Softmax(dim=1)
    #print(outputs)
    output1 = m(outputs)
    #print(output1)
    values, indices = torch.max(labels.data[0], 0)
    values1, indices1 = torch.max(outputs.data[0], 0)
    x1=indices.cpu().numpy()
    x2=indices1.cpu().numpy()
    if(x1==x2):
        correct+=1
        #print("hello")


# In[24]:


print(correct)


# In[25]:


print(len(train))


# In[26]:


print(correct/len(train))


# In[27]:


net.eval()
test_correct=0
for i in range(len(test)):
    x_batch = test[i]
    #print(x_batch.shape)
    x = np.reshape(x_batch,[-1,60*431])
    y_batch = test_label[i]
    #print(y_batch)
    y=np.array(y_batch)
    inputs = Variable(torch.FloatTensor(torch.from_numpy(x).float())).cuda()
    labels = Variable(torch.FloatTensor(torch.from_numpy(y).float())).cuda()
    outputs = net(inputs)
    m = nn.Softmax(dim=1)
    #print(outputs)
    output1 = m(outputs)
    #print(output1)
    values, indices = torch.max(labels.data[0], 0)
    values1, indices1 = torch.max(outputs.data[0], 0)
    x1=indices.cpu().numpy()
    x2=indices1.cpu().numpy()
    if(x1==x2):
        test_correct+=1
        #print("hello")


# In[28]:


print(test_correct)
print(len(test))
acc=test_correct/(len(test))


# In[29]:


print(acc)

