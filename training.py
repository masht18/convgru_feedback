import torch
import math
import pickle
import argparse
import hub

from scipy.special import softmax
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T

from model.topdown_gru import ConvGRUExplicitTopDown
from utils.audio_dataset import MELDataset
from utils.datagen import *
from ambiguous.dataset.dataset import DatasetTriplet

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
parser.add_argument('--hidden_dim', type = int, default = 10)
parser.add_argument('--reps', type = int, default = 1)
parser.add_argument('--topdown', type = str2bool, default = True)
parser.add_argument('--task_prob', type = float, default = 0.5) # 1=fully ambiguous bottom-up, 0=clean bottom-up
parser.add_argument('--topdown_type', type = str, default = 'image')
parser.add_argument('--connection_decay', type = str, default = 'ones')
parser.add_argument('--return_bottom_layer', type = str2bool, default = False)

parser.add_argument('--model_save', type = str, default = 'saved_models/image_no_topdown_50.pt')
parser.add_argument('--results_save', type = str, default = 'results/no_topdown_50epoch.npy')

args = vars(parser.parse_args())
if args['topdown_type'] != 'text' and args['topdown_type'] != 'image' and args['topdown_type'] != 'audio':
    raise ValueError('Topdown style not implemented')

# %%
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Seed for reproducibility
torch.manual_seed(42)
print(device)

# %% [markdown]
# # 1: Prepare dataset
print('Loading datasets')

MNIST_path='datasets/'
transform = transforms.ToTensor()
clean_train_data = datasets.MNIST(root=MNIST_path, download=True, train=True, transform=transform)
clean_test_data = datasets.MNIST(root=MNIST_path, download=True, train=False, transform=transform)

# Label references, which help generate sequences
mnist_ref_train = generate_label_reference(clean_train_data)
mnist_ref_test = generate_label_reference(clean_test_data)

# Create dataloaders
amb_trainset = DatasetTriplet('/home/mila/m/mashbayar.tugsbayar/datasets/amnistV2', train=True)
amb_testset = DatasetTriplet('/home/mila/m/mashbayar.tugsbayar/datasets/amnistV2', train=False)
ambiguous_train = DataLoader(amb_trainset, batch_size=64, shuffle=True)
ambiguous_test = DataLoader(amb_testset, batch_size=64, shuffle=True)

clean_train_dataloader = DataLoader(clean_train_data, batch_size=64, shuffle=True)
clean_test_dataloader = DataLoader(clean_test_data, batch_size=64, shuffle=True)

# Subset clean data so it only has numbers used in ambiguous training
ambiguous_classes = [0, 1, 3, 4, 5, 6, 7, 8, 9]
test_indices = [indices for c in ambiguous_classes for indices in mnist_ref_test[c]]
clean_amb_class_test = Subset(clean_test_data, test_indices) 
clean_amb_class_test_dataloader = DataLoader(clean_amb_class_test, batch_size=64, shuffle=True)

train_indices = [indices for c in ambiguous_classes for indices in mnist_ref_train[c]]
clean_amb_class_train = Subset(clean_train_data, train_indices) 
clean_amb_class_train_dataloader = DataLoader(clean_amb_class_train, batch_size=64, shuffle=True)

# FSDD dataset in case of audio topdown
audio_ds = hub.load("hub://activeloop/spoken_mnist")
mel_ds = MELDataset(audio_ds)
audio_ref = generate_label_reference(audio_ds, dataset_type='fsdd')

print('Successfully loaded datasets')
# In case of text topdown 
text_labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
text_labels = str_to_bits(text_labels)

'''
Provide image or text clue, given labels
    
    :param reference_imgs (torchvision.Dataset)
        dataset of images to fetch clues from. If None, provide text clue
    :param dataset_ref (list)
        if providing image clue, provide label reference as well
        
'''
def fetch_clues(labels, clue_type='text', reference_imgs=None, dataset_ref=None):
    
    if clue_type == 'text':
        clues = torch.zeros(labels.shape[0], 27)
        for batch_idx in range(labels.shape[0]):
            clues[batch_idx] = text_labels[labels[batch_idx]]
    elif clue_type == 'image':   
        clues = torch.zeros((labels.shape[0], 1, 28, 28))
        for batch_idx in range(labels.shape[0]):
            clues[batch_idx] = reference_imgs[random.choice(dataset_ref[labels[batch_idx]]).item()][0]
    else:
        clues = torch.zeros((labels.shape[0], 1, 64, 64))
        for batch_idx in range(labels.shape[0]):
            clues[batch_idx] = mel_ds[random.choice(dataset_ref[labels[batch_idx]]).item()][0]
        
    return clues.to(device)

'''
Provide image or text clue, given labels
    
    :param dataloader
        dataloader to draw the target image from
    :param clean data (torchvision.Dataset)
        clean dataset to draw bottom-up sequence images from
    :param topdown_type ('image' or 'text' or 'audio')
    :param dataset_ref (list)
        if providing image clue, provide label reference as well
'''

def test_sequence(dataloader, clean_data, topdown_type='image', label_style='ambiguous', bottomup_ambiguity=False, relevant_clue=True):
    
    correct = 0
    total = 0

    with torch.no_grad():

        for i, data in enumerate(dataloader, 0):
            optimizer.zero_grad()

            imgs, label = data
            
            if label_style == 'ambiguous':
                pick = np.random.binomial(1, 0.5)
                imgs, label = imgs.to(device), label[:, pick].to(device)
            else:
                imgs, label = imgs.to(device), label.to(device)
                
            #input_seqs = sequence_gen(imgs, label, clean_data, mnist_ref_test, seq_style=seq_style)
            input_seqs = choice_sequence_gen(imgs, label, clean_test_data, mnist_ref_test, full_ambiguity=bottomup_ambiguity, show=True)
            
            if label_style == 'per-class':
                label = torch.argmax(label, 1)
                
            # Is the topdown going to relevant?
            if relevant_clue == True:
                clue_label = label
            else:
                clue_label = torch.randint(0, 10, (label.shape[0], ))
            
            # Generate the topdown signal based on given modality
            if topdown_type == 'audio':
                topdown = fetch_clues(clue_label, 'audio', audio_ds, audio_ref)
            elif topdown_type == 'image':
                topdown = fetch_clues(clue_label, 'image', clean_test_data, mnist_ref_test)
            else: 
                topdown = fetch_clues(clue_label, 'text')

            output = model(input_seqs.float(), topdown)

            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    #print('Accuracy of the network on the 10000 test images: %d %%' % (
    #    100 * correct / total))
    
    return correct/total

def train_sequence(topdown_type = 'image', p_amb_vs_clean_first = 0.5, p_correct_topdown = 0.5):
    amb_iter = iter(ambiguous_train)
    clean_iter = iter(clean_amb_class_train_dataloader)
    running_loss = 0.0
        
    for i, data in enumerate(ambiguous_train, 0):
            
        optimizer.zero_grad()
            
        imgs, label = data
        pick = np.random.binomial(1, 0.5)
        imgs, label = imgs.to(device), label[pick].to(device)
            
        # Additive or random sequence?
        bottomup_ambiguity = np.random.binomial(1, p_amb_vs_clean_first)
        #if bottomup_clue == False:
        #    seq_style = 'addition'
        #else:
        #    seq_style = 'random'
            
        # Generate sequence
        #input_seqs = sequence_gen(imgs, label, clean_train_data, mnist_ref_train, seq_style=seq_style)
        input_seqs = choice_sequence_gen(imgs, label, clean_train_data, mnist_ref_train, full_ambiguity=bottomup_ambiguity)
            
        if bottomup_ambiguity  == True:  # If bottom-up was ambiguous, give correct topdown info based on label
            clue_label = label
        else:   # Else if bottom-up signal was correct, give the correct answer or a random answer
            if np.random.binomial(1, p_correct_topdown) == 1:
                clue_label = label
            else:
                clue_label = torch.randint(0, 10, (label.shape[0], ))

        # Generate the topdown signal based on given modality
        if topdown_type == 'audio':
            topdown = fetch_clues(clue_label, 'audio', audio_ds, audio_ref)
        elif topdown_type == 'image':
            topdown = fetch_clues(clue_label, 'image', clean_test_data, mnist_ref_test)
        else: 
            topdown = fetch_clues(clue_label, 'text')
            
        output = model(input_seqs.float(), topdown)
            
        loss = criterion(output, label)
        running_loss += loss.item()
            
        loss.backward()
        optimizer.step()
    
    return running_loss


if args['connection_decay'] == 'biological':
    connection_strengths = [15711/16000, 14833/16000, 9439/16000]
else:
    connection_strengths = [1, 1, 1]

losses = {'loss': [], 'val_topdown_only': [], 'val_add_only': [], 'val_add_topdown': [], 'val_none': []}    
criterion = nn.CrossEntropyLoss()
model = ConvGRUExplicitTopDown((28, 28), 10, input_dim=1, 
                               hidden_dim=args['hidden_dim'], 
                               kernel_size=(3,3),
                               connection_strengths=connection_strengths,
                               num_layers=args['layers'],
                               reps=args['reps'], 
                               topdown=args['topdown'], 
                               topdown_type=args['topdown_type'],
                               return_bottom_layer=args['return_bottom_layer']).cuda().float()
#model = ConvGRU((28, 28), 10, input_dim=1, hidden_dim=10, kernel_size=(3,3), num_layers=args['layers'], 
#                       dtype=torch.cuda.FloatTensor, batch_first=True).cuda().float()
optimizer = optim.Adam(model.parameters())

if os.path.exists(args['model_save']):
    model = model.load_state_dict(torch.load(args['model_save']))
    print("Loading existing ConvGRU model")
else:
    print("No pretrained model found. Training new one.")

for epoch in range(args['epochs']):
    val_topdown_only = test_sequence(ambiguous_test, clean_test_data, topdown_type=args['topdown_type'], bottomup_ambiguity=True, relevant_clue=True)
    val_add_only = test_sequence(ambiguous_test, clean_test_data, topdown_type=args['topdown_type'], bottomup_ambiguity=False, relevant_clue=False)
    val_none = test_sequence(ambiguous_test, clean_test_data, topdown_type=args['topdown_type'], bottomup_ambiguity=True, relevant_clue=False)
    val_add_topdown = test_sequence(ambiguous_test, clean_test_data, topdown_type=args['topdown_type'], bottomup_ambiguity=False, relevant_clue=True)
    loss = train_sequence(topdown_type=args['topdown_type'], p_amb_vs_clean_first = args['task_prob'])
    
    printlog = '| epoch {:3d} | running loss {:5.4f} | val accuracy (add only) {:1.5f} | val accuracy (topdown_only) {:1.5f}  | val accuracy (add+topdown) {:1.5f}  | val accuracy (no clues) {:1.5f}'.format(epoch, loss, val_add_only, val_topdown_only, val_add_topdown, val_none)

    print(printlog)
    losses['loss'].append(loss)
    losses['val_topdown_only'].append(val_topdown_only)
    losses['val_add_only'].append(val_add_only)
    losses['val_add_topdown'].append(val_add_topdown)
    losses['val_none'].append(val_none)

    with open(args['results_save'], "wb") as f:
        pickle.dump(losses, f)
        
    torch.save(model.state_dict(), args['model_save'])

