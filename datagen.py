# Import dependencies
import os
import random
import string
import importlib
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import yaml
import h5py
from copy import deepcopy
from ambiguous.data_utils import *
from ambiguous.models.ambiguous_generator import MNISTGenerator
from torchvision.utils import save_image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

pure_pairs = np.array([(3,8),(8,3),(3,5),(5,3),(5,8),(8,5),(0,6),(6,0), (4,9), (9,4), (1, 7), (7,1)])

'''
Creates a label reference of size [num_classes x num_images_per_class] for sequence generation.
This label reference helps find an example of a specific class given a dataset.

    :param dataset (torchvision.datasets):
        dataset to generate label reference from
    :param num_classes (int)
    :param dataset_type (str):
        mnist or fsdd 
    :return label_ref(tensor):
        indices of images with given class for each class 
    
'''
def generate_label_reference(dataset, num_classes=10, dataset_type='mnist'):
    labels = dataset.labels.data() if dataset_type == 'fsdd' else dataset.targets
    label_ref = []

    for i in range(num_classes):
        label_ref.append((labels == i).nonzero()[0])
        
    return label_ref

'''
Given an MNIST image, generate a 3 image sequence based on its class (6-7-8 from 8, 8-9-0 from 0, etc)

    :param imgs (torch.tensor) 
        [batch_size x width x height]
        Target images at the end of the sequence. The sequence can clue to them or not.
    :param labels (torch.tensor) 
        [batch_size] OR [batch_size x num_class]
    :param clean_data (torchvision.datasets)
        dataset to draw randomly images of the sequence from
    :param mnist_ref (torch.tensor) 
        [num_classes x num_images_per_class] 
        label reference. Must be generated from the clean_data
    :param label_style (string):
        per-class for labels that are size [batch_size x num_class], anything else for labels with size [batch_size]
    :param seq_style (string):
        If 'order', sequences are like 8-9-0, 4-5-6. If addition, the first two numbers add up to the final number
    :param show (bool):
        show first sequence generated
    :return input_seqs (torch.tensor) 
        [batch_size x size of sequence (3) x width x height]
         
'''
def sequence_gen(imgs, label, clean_data, mnist_ref, seq_style = 'addition', label_style = 'amb', show=False):
    input_seqs = torch.zeros_like(imgs)
    input_seqs = torch.unsqueeze(input_seqs, 1).repeat(1, 3, 1, 1, 1)
    
    # for each image in batch
    for batch_idx in range(label.shape[0]):
        
        # get its label
        if label_style == 'per-class': # if labels are 0,1 for each class
            t2_label = torch.argmax(label[batch_idx]).item()
            
        else: # if labels are the class number
            t2_label = label[batch_idx].item()
        
        # the given image (ambiguous or not) is always the last number of the sequence
        t2 = imgs[batch_idx].unsqueeze(0)
        
        # rest of the images in sequence are all chosen from clean_data using the reference
        if seq_style == 'addition':
            t1_label = np.random.randint(t2_label) if t2_label != 0 else np.random.randint(10)
            t0_label = t2_label - t1_label
        elif seq_style == 'random':
            t1_label = random.choice(np.delete(np.arange(10), t2_label))
            t0_label = random.choice(np.delete(np.arange(10), t2_label))   
        else:
            t1_label = t2_label - 1
            t0_label = t1_label - 2
            
        t0 = torch.unsqueeze(clean_data[random.choice(mnist_ref[t0_label]).item()][0], 0).cuda()
        t1 = torch.unsqueeze(clean_data[random.choice(mnist_ref[t1_label]).item()][0], 0).cuda() 
        
        # construct sequence
        sequence = torch.vstack((t0, t1, t2))
        input_seqs[batch_idx] = sequence
        
        # show the first sequence generated if show=True
        if show==True and batch_idx==0:
            fig = plt.figure(figsize=(8, 8))
            columns = 3
            rows = 1
            for i in range(3):
                img = sequence[i].permute(1, 2, 0).cpu().detach()
                fig.add_subplot(rows, columns, i+1)
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.get_xaxis().set_visible(False)
                plt.imshow(img)
            plt.show()
            
    return input_seqs

'''
Given an MNIST image, generate a sequence of two images with same label, where the first image is the clue to the second

    :param imgs (torch.tensor) 
        [batch_size x width x height]
    :param labels (torch.tensor) 
        [batch_size] OR [batch_size x num_class]
    :param gen (MNISTGenerator)
        pre-trained ambiguous MNIST generator
    :param clean_data (torchvision.datasets)
        dataset to draw randomly images of the sequence from
    :param mnist_ref (torch.tensor) 
        [num_classes x num_images_per_class]
        label reference. Must be generated from the clean_data        
    :param label_style (string):
        per-class for labels that are size [batch_size x num_class], anything else for labels with size [batch_size]    
    :param seq_style (string):
        If 'order', sequences are like 8-9-0, 4-5-6. If addition, the first two numbers add up to the final number
    :param show (bool):
        show first sequence generated
    :return input_seqs (torch.tensor) 
        [batch_size x sequence size (2) x width x height]
         
'''
def choice_sequence_gen(imgs, label, clean_data, mnist_ref, full_ambiguity = True, show=False):
    input_seqs = torch.zeros_like(imgs)
    input_seqs = torch.unsqueeze(input_seqs, 1).repeat(1, 2, 1, 1, 1)
    
    # for each image in batch
    for batch_idx in range(label.shape[0]):
        
        t1_label = label[batch_idx].item()
        
        # the given image
        t1 = imgs[batch_idx].unsqueeze(0)
        
        # the other image is a clue 
        if full_ambiguity == True:
            # If using ambiguous bottom-up clue, use pre-trained decoder and encoder to generate a new ambiguous image
            t0_label = torch.randint(0, 10, (1, ))
            t0 = torch.unsqueeze(clean_data[random.choice(mnist_ref[t0_label]).item()][0].cuda(), 0)
        else: 
            # else, just return a clean image with the same label 
            t0 = torch.unsqueeze(clean_data[random.choice(mnist_ref[t1_label]).item()][0].cuda(), 0)
        
        # construct sequence
        sequence = torch.cat([t0, t1])
        input_seqs[batch_idx] = sequence
        
        # show the first sequence generated if show=True
        if show==True and batch_idx==0:
            fig = plt.figure(figsize=(8, 8))
            columns = 2
            rows = 1
            for i in range(2):
                img = sequence[i].permute(1, 2, 0).cpu().detach()
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(img)
                plt.savefig('sequence{:3d}.png'.format(batch_idx))
            plt.show()
            
    return input_seqs


'''
Given list of strings, return list where each string is a 1D tensor of size 27, 0 or 1 depending on presence of letter 
'''
def str_to_bits(strings):
    list_of_bits = []
    
    for word in strings:
        bits = torch.zeros(27).cuda()
        for c in word:
            bits[string.ascii_lowercase.index(c)] = 1
        list_of_bits.append(bits)
        
    return list_of_bits
        
    