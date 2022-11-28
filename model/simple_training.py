'''
Pared-down training script for testing basic functionality only
'''

import torch
import math
import pickle
import argparse

from scipy.special import softmax
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T
from utils.datagen import *
from graph import Graph
from model.topdown_gru import ConvGRUExplicitTopDown

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type = bool, default = True, help = 'use gpu or not')
parser.add_argument('--epochs', type = int, default = 50)
parser.add_argument('--layers', type = int, default = 1)
parser.add_argument('--topdown_c', type = int, default = 10)
parser.add_argument('--topdown_h', type = int, default = 10)
parser.add_argument('--topdown_w', type = int, default = 10)
parser.add_argument('--hidden_dim', type = int, default = 10)
parser.add_argument('--reps', type = int, default = 1)
parser.add_argument('--topdown', type = str2bool, default = True)
parser.add_argument('--connection_decay', type = str, default = 'ones')
parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

parser.add_argument('--model_save', type = str, default = 'saved_models/audio_newdata.pt')
parser.add_argument('--results_save', type = str, default = 'results/no_topdown_newdata.npy')

args = vars(parser.parse_args())
if args['topdown_type'] != 'text' and args['topdown_type'] != 'image' and args['topdown_type'] != 'audio':
    raise ValueError('Topdown style not implemented')

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(42)
print(device)

# # 1: Prepare dataset
print('Loading datasets')

MNIST_path='D:\LiNC research\data'
save_path = 'D:/LiNC research/saved_models/image_topdown.pt'
train_data = datasets.MNIST(root=MNIST_path, download=True, train=True, transform=T.ToTensor())
test_data = datasets.MNIST(root=MNIST_path, download=True, train=False, transform=T.ToTensor())

# Label references, which help generate sequences (these guys tell you where to look in the dataset if you need a 6, 4, etc)
mnist_ref_train = generate_label_reference(train_data)
mnist_ref_test = generate_label_reference(test_data)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


connection_strengths = [1, 1, 1, 1] 
criterion = nn.CrossEntropyLoss()
connections = [[0,1,1,0],[0,0,1,1],[0,0,0,1], [0,0,0,0]] #V1 V2 V4 IT
input_node = 0 # V1
output_node = 3 #IT
graph = Graph(connections = connections, connection_strengths = connection_strengths, input_node = input_node, output_node = output_node)
model = graph.build_architecture()
# model = ConvGRUExplicitTopDown((28, 28), 10, input_dim=1, 
#                                hidden_dim=10, 
#                                kernel_size=(3,3),
#                                connection_strengths=connection_strengths,
#                                num_layers=2,
#                                reps= 2, 
#                                topdown=True, 
#                                topdown_type='image',
#                                dtype = torch.FloatTensor,
#                                return_bottom_layer=True,
#                                batch_first = False)
#model = ConvGRU((28, 28), 10, input_dim=1, hidden_dim=10, kernel_size=(3,3), num_layers=args['layers'], 
#                       dtype=torch.cuda.FloatTensor, batch_first=True).cuda().float()
optimizer = optim.Adam(model.parameters())

def test_sequence(dataloader, clean_data, dataset_ref):
    
    '''
    Inference
        :param dataloader
            dataloader to draw the target image from
        :param clean data (torchvision.Dataset)
            clean dataset to draw bottom-up sequence images from
        :param dataset_ref (list)
            if providing image clue, provide label reference as well
    '''
    correct = 0
    total = 0

    with torch.no_grad():

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            imgs, label = data
            imgs, label = imgs.to(device), label.to(device)
                
            # Generate a sequence that adds up to loaded image    
            input_seqs = sequence_gen(imgs, label, clean_data, dataset_ref, seq_style='addition')
                
            # Generate random topdown
            topdown = torch.rand(imgs.shape[0], args['topdown_c'], args['topdown_h'], args['topdown_w'])

            output = model(input_seqs.float(), topdown)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    #print('Accuracy of the network on the 10000 test images: %d %%' % (
    #    100 * correct / total))
    
    return correct/total

def train_sequence():
    running_loss = 0.0
        
    for i, data in enumerate(train_loader, 0):
            
        optimizer.zero_grad()
            
        imgs, label = data
        imgs, label = imgs.to(device), label.to(device)

        input_seqs = sequence_gen(imgs, label, train_data, mnist_ref_train, seq_style='addition')

        # Generate random topdown for testing purposes only
        topdown = torch.rand(imgs.shape[0], args['topdown_c'], args['topdown_h'], args['topdown_w'])
            
        output = model(input_seqs.float(), topdown)
            
        loss = criterion(output, label)
        running_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    return running_loss

losses = {'loss': [], 'train_acc': [], 'val_acc': []}    

if os.path.exists(args['model_save']):
    model.load_state_dict(torch.load(args['model_save']))
    print("Loading existing ConvGRU model")
else:
    print("No pretrained model found. Training new one.")

for epoch in range(args['epochs']):
    train_acc = test_sequence(train_loader, train_data, mnist_ref_train)
    val_acc = test_sequence(test_loader, test_data, mnist_ref_test)
    loss = train_sequence()
    
    printlog = '| epoch {:3d} | running loss {:5.4f} | train accuracy {:1.5f} |val accuracy {:1.5f}'.format(epoch, loss, train_acc, val_acc)

    print(printlog)
    losses['loss'].append(loss)
    losses['train_acc'].append(train_acc)
    losses['val_acc'].append(val_acc)

    with open(args['results_save'], "wb") as f:
        pickle.dump(losses, f)
        
    torch.save(model.state_dict(), args['model_save'])