import platform
import io
import os


import matplotlib.pyplot as plt
#from google.colab import files
from matplotlib.pyplot import cm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
torch.manual_seed(1234)

import librosa
import librosa.display as display
#import librosa.stft as stft
import pdb

import sklearn
import sklearn.model_selection as ms

dataset=[]

use_gpu=torch.cuda.is_available()
print("GPU Available:{}".format(use_gpu))

def readFile(filepath):
    y,sr=librosa.load(filepath)

    D=librosa.stft(y)

    D_real, D_imag = np.real(D), np.imag(D)
    #print(D_imag)
    #D_energy = np.real(D)
    a=D_real**2+D_imag**2
    #print(a)
    D_energy = np.sqrt(D_real**2+D_imag**2)
    # a=D_real**2+D_imag**2
    # if a>=0:
    #     D_energy = np.sqrt(D_real**2+D_imag**2)
    # else:
    #     print(a)
    #     D_energy=0
    
    #result=np.log(D_energy)
    norm = librosa.util.normalize(D_energy)
    display.specshow(norm, y_axis='log', x_axis='time')
    #plt.imshow(result)
    #plt.savefig(filepath+".png")

    #plt.plot(result)
    #plt.show()
    result=np.pad(norm,([(0,0),(0,315-len(norm[0]))]),'constant')
    return result

# f=readFile("./laugh_1.m4a_0.wav")
# print(f)
#numItr = 0
#print("??????HELLO")

def import_data(folder,i):
    for file in os.listdir(folder):
        # temp=[]
        # f=folder+"/"+file
        # temp.append(torch.tensor(readFile(f)))
        # temp.append(torch.tensor(i))
        # dataset.append(temp)
        try:
            temp=[]
            f=folder+"/"+file
            temp.append(torch.tensor(readFile(f)))
            temp.append(torch.tensor(i))
            dataset.append(temp)
        except Exception:
            continue


"""Import Data"""
########################
import_data("901 - Silence",0)
#print(numItr)
import_data("902 - Noise",1)
#print(numItr)
import_data("903 - Baby laugh",2)
#print(numItr)
import_data("301 - Crying baby",3)
#print(numItr)
# print(len(dataset))

train_set,test_set=ms.train_test_split(dataset,train_size=344)

# print(len(train_set))
# print(len(test_set))
# 
# x=list(train_set)[0][0]
# print(x.shape)
# print("Target: {}".format(x[0]))

# plt.plot(train_set)
# plt.show()

# import pandas as pd

BATCH_SIZE=1
# #transform=transforms.ToTensor()

# #data
#train_set, test_set = train_test_split(features, labels, test_size=0.33, random_state=42)

train_loader=torch.utils.data.DataLoader(train_set,batch_size=BATCH_SIZE,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_set,batch_size=BATCH_SIZE,shuffle=True)

# transform=transforms.ToTensor()
# train_examples=datasets.MNIST(root="/tmp",train=True,download=True,transform=transform)
# test_examples=datasets.MNIST(root="/tmp",train=False,download=True,transform=transform)

# BATCH_SIZE=100

# train_loader=torch.utils.data.DataLoader(train_examples,batch_size=BATCH_SIZE,shuffle=True)
# test_loader=torch.utils.data.DataLoader(test_examples,batch_size=BATCH_SIZE,shuffle=False)

# print(len(train_loader))

class CNN_Baby(nn.Module):
    def __init__(self):
        super(CNN_Baby,self).__init__()
        # self.cv1=nn.Conv2d(1,16,5,stride=[1],padding=[0],dilation=[1])
        # self.cv2=nn.Conv2d(16,32,5,stride=[1],padding=[0],dilation=[1])
        # #self.fc=nn.Linear(322875,4)
        # self.fc=nn.Linear(12800,10)
        self.fc1=nn.Linear(1025*315,4)
        # self.fc2=nn.Linear(500,100)
        self.fc3=nn.Linear(500,4)
        #
    def forward(self,x):
        # out=F.relu(self.cv1(x))
        # out=F.relu(self.cv2(out))
        # out=out.view(BATCH_SIZE,-1)
        # out=self.fc(out)
        out=self.fc1(x)
        # out=F.relu(self.fc2(out))
        #out=F.relu(self.fc3(out))
        return out

cnn = CNN_Baby()
#print(cnn)
if use_gpu:
    cnn.cuda()

cnn.eval()
criterion = nn.CrossEntropyLoss() # because categorical
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

def train_baby(epoch,model,train_loader,optimizer):
    model.train()

    total_loss=0
    correct=0

    for i, (image,label) in enumerate(train_loader):

        optimizer.zero_grad()

        image=image.view(-1,1025*315)
        #image=image.float()
        #image.reshape(BATCH_SIZE,1025,315)
        #image=image.squeeze(0)
        #print(image.shape)
        
        if use_gpu:
            image=image.cuda()
            label=label.cuda()

        prediction=model(image)
        label=label.long()
        loss=criterion(prediction,label)

        loss.backward()

        optimizer.step()

        total_loss+=loss
        pred_classes = prediction.data.max(1,keepdim=True)[1]
        correct += pred_classes.eq(label.data.view_as(pred_classes)).sum().double()

    mean_loss=total_loss/len(train_loader.dataset)
    acc=correct/len(train_loader.dataset)

    print('Train Epoch: {}   Avg_Loss: {:.5f}   Acc: {}/{} ({:.3f}%)'.format(
        epoch, mean_loss, correct, len(train_loader.dataset),
        100. * acc))

    return mean_loss, acc

def eval_baby(model,eval_loader):
    
    model.eval()

    total_loss=0
    correct=0

    for i, (image,label) in enumerate(eval_loader):

        optimizer.zero_grad()

        image=image.view(-1,1025*315)
        #image=image.float()
        #image.squeeze(0)
        

        if use_gpu:
            image=image.cuda()
            label=label.cuda()

        prediction=model(image)
        label=label.long()

        loss=criterion(prediction,label)

        loss.backward()

        optimizer.step()

        total_loss+=loss

        pred_classes=prediction.data.max(1,keepdim=True)[1]

        correct+=pred_classes.eq(label.data.view_as(pred_classes)).sum().double()

    mean_loss=total_loss/len(eval_loader.dataset)
    acc=correct/len(eval_loader.dataset)

    print('Eval:  Avg_Loss: {:.5f}   Acc: {}/{} ({:.3f}%)'.format(
        mean_loss, correct, len(eval_loader.dataset),
        100. *acc)) 

    return mean_loss, acc

def save_model(epoch, model, path='./'):
    
    # file name and path 
    filename = path + 'neural_network_{}.pt'.format(epoch)
    
    # load the model parameters 
    torch.save(model.state_dict(), filename)
    
    
    return model

def load_model(epoch, model, path='./'):
    
    # file name and path 
    filename = path + 'neural_network_{}.pt'.format(epoch)
    
    # load the model parameters 
    model.load_state_dict(torch.load(filename))
    
    
    return model

# Number of epochs 
numEpochs = 20 

# checkpoint frequency 
checkpoint_freq = 10

# path to save the data 
path = './'

# empty lists 
train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

# traininng 
for epoch in range(1, numEpochs + 1):
    
    # train() function (see above)
    train_loss, train_acc = train_baby(epoch, cnn, train_loader, optimizer)
    
    # eval() functionn (see above)
    test_loss, test_acc = eval_baby(cnn, test_loader)    
    
    # append lists for plotting and printing 
    train_losses.append(train_loss)    
    test_losses.append(test_loss)
    
    train_accuracies.append(train_acc)    
    test_accuracies.append(test_acc)
    
    # Checkpoint
    if epoch % checkpoint_freq ==0:
        save_model(epoch, cnn, path)

# Last checkpoint
save_model(numEpochs, cnn, path)
    
print("\n\n\nOptimization ended.\n") 


def get_label_index(tensor_pred):
    npray = tensor_pred.detach().numpy()
    index = [npray[0]]
    for x in range(0,len(npray) -1):
        npray [x+1] > npray[x]
        index = (x+1)
    return index

results = []

f=readFile("cry.3gp")
p=cnn(torch.tensor(f).view(-1,1025*315))
print("Crying file test:{}".format(p))

results.append(get_label_index(p))

f=readFile("laugh_1.m4a_0.wav")
p=cnn(torch.tensor(f).view(-1,1025*315))
print("Laughing file test:{}".format(p))

results.append(get_label_index(p))

f=readFile("noise1.ogg")
p=cnn(torch.tensor(f).view(-1,1025*315))
print("Noise file test:{}".format(p))

results.append(get_label_index(p))

f=readFile("silence.wav_0.wav")
p=cnn(torch.tensor(f).view(-1,1025*315))
print("Silence file test:{}".format(p))

results.append(get_label_index(p))

def get_res():
    return np.asarray(results)
