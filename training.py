import torch
import math
import pickle
import argparse
import random

from scipy.special import softmax
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data_utils
import os
from model.topdown_gru import ConvGRUExplicitTopDown

def fetch_clues(labels, reference_imgs=None, dataset_ref=None):
    clues = torch.zeros((labels.shape[0], 1, 28, 28)) #dummy function to produce the right size
        
    return clues.to(device)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(42)
print(device)

# %% [markdown]
# # 1: Prepare dataset
print('Loading datasets')

MNIST_path='D:\LiNC research\data'
save_path = 'D:/LiNC research/saved_models/image_topdown.pt'
train_data = datasets.MNIST(root=MNIST_path, download=True, train=True, transform=T.ToTensor())
test_data = datasets.MNIST(root=MNIST_path, download=True, train=False, transform=T.ToTensor())

train_data = data_utils.Subset(train_data, torch.arange(1000))
test_data = data_utils.Subset(test_data, torch.arange(100))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


connection_strengths = [1, 1, 1] 
criterion = nn.CrossEntropyLoss()
model = ConvGRUExplicitTopDown((28, 28), 10, input_dim=1, 
                               hidden_dim=10, 
                               kernel_size=(3,3),
                               connection_strengths=connection_strengths,
                               num_layers=2,
                               reps= 2, 
                               topdown=True, 
                               topdown_type='image',
                               dtype = torch.FloatTensor,
                               return_bottom_layer=True)
#model = ConvGRU((28, 28), 10, input_dim=1, hidden_dim=10, kernel_size=(3,3), num_layers=args['layers'], 
#                       dtype=torch.cuda.FloatTensor, batch_first=True).cuda().float()
optimizer = optim.Adam(model.parameters())


for epoch in range(1):
    # trainning
    ave_loss = 0
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)
        out = model(x, fetch_clues(torch.randint(0, 10, (target.shape[0], ))))

        loss = criterion(out, target)
        ave_loss = ave_loss * 0.9 + loss.data * 0.1 #question: deleted [0] after data
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0 or (batch_idx+1) == len(train_loader):
            print ("==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx+1, ave_loss)")
                
    # testing
    correct_cnt, ave_loss = 0, 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        
        out = model(x, fetch_clues(torch.randint(0, 10, (target.shape[0], ))))
        loss = criterion(out, target)
        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        # smooth average
        ave_loss = ave_loss * 0.9 + loss.data * 0.1 #question: deleted [0] after data
        
        if(batch_idx+1) % 100 == 0 or (batch_idx+1) == len(test_loader):
            print ("==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx+1, ave_loss)")

torch.save(model.state_dict(), save_path)
