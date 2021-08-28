"""Code to generate fewshot samples

This code is meant to be invoked from the command line. Code is adapted from the 
original run code. I tried to keep only the data loading code and edited the 
function so it only get the fewshot data for K-shot (command line argument). 
From there, each slice of the data is then saved to disk. (Yes this is terribly
inefficient on disk space, but I didn't want to make too many modifications to 
the primary run code if possible). 

This code will create an output directory in the same directory as this code. A
sub-directory for the run will be created and inside it, a drug folder and inside
that a tissue folder will be created.

    e.g. output / RUN-name / DRUG / TISSUE / data

In order to get all of the different tissues, the code will need to run multiple
times.

Note the testing dataset is fixed across trial, so a separate numpy file is 
created.
"""

import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
from torch.autograd import Variable
import sys
import torch.nn as nn
import pickle
import copy

filepath = os.path.realpath(__file__)
dir_name = os.path.dirname(filepath)
sys.path.append(os.path.dirname(dir_name))
home_dir = os.path.dirname(os.path.dirname(dir_name))

from model.data_loading import *
from model.utils import *
from model.score import *
from model.inner_loop import InnerLoop
from model.mlp import mlp

# Training settings
parser = argparse.ArgumentParser()
work_dic = home_dir + '/data/cell_line_lists/'
data_dic = home_dir + '/data/drug_feature/'
print(dir_name)
print(__file__)

parser.add_argument('--tissue', type=str, default='UPPER_AERODIGESTIVE_TRACT', help='Validation tissue, using the rest tissues for training')
parser.add_argument('--drug', type=str, default='AC220', help='Treated drug')
parser.add_argument('--K', type=int, default=5, help='Perform K shot learning')
#parser.add_argument('--tissue_list', type=str, default=work_dic + 'cell_line_data/tissue_cell_line_list.pkl', help='Cell line list for different tissues')
parser.add_argument('--num_trials', type=int, default=10, help='Number of trials for unseen tissue')
parser.add_argument('--seed', type=int, default=19, help='Random seed.')
parser.add_argument('--inner_batch_size', type=int, default=10, help='Batch size for each individual learning job')
parser.add_argument('--run_name', type=str, default='run', help='Run name')

args = parser.parse_args()

job_directory = home_dir + '/output/{}/'.format(args.run_name)

K = args.K
num_trials = args.num_trials
gene = args.drug

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


fewshot_directory = job_directory + 'fewshot_data/' + gene  + '/'

drug_tissue_list = work_dic + args.drug + '_tissue_cell_line_list.pkl'
with open(drug_tissue_list, 'rb') as f:
    drug_tissue_map = pickle.load(f)

# Load data
out= load_data_cell_line( drug_tissue_map, args.drug, args.tissue, K, path=data_dic )
train_feature, train_label, tissue_index_list, drug_test_feature, drug_test_label, _  = out
feature_dim = train_feature.shape[1]

tissue_list = work_dic + gene + '_tissue_cell_line_list.pkl'
with open(tissue_list, 'rb') as f:
    tissue_map = pickle.load(f)

tissue_directory = fewshot_directory + args.tissue + '/'
mkdir_cmd = 'mkdir -p ' + tissue_directory
os.system(mkdir_cmd)

for trial in range(num_trials):
    # Sample a few shot learning task. Here we use k training, and use the rest for testing. 
    unseen_train_loader, unseen_test_loader = get_unseen_data_loader(
        drug_test_feature, drug_test_label, args.K, args.inner_batch_size)
    
    # if len(unseen_train_loader) > 1 or len(unseen_test_loader) > 1: 
    #     raise RuntimeError("DID NOT EXPECT TRAIN_LOADER TO BE LONGER THAN 1!!!")

    train_X, train_y = [np.vstack([mat.cpu() for mat in mats]) for mats in  zip(*unseen_train_loader)]
    test_X, test_y = [np.vstack([mat.cpu() for mat in mats]) for mats in  zip(*unseen_test_loader)]
    
    for i in range(1, K+1): 
        filename = "{}_{}_{}-shot_{}-trial_train".format(args.drug, args.tissue, i, trial)
        filename = tissue_directory + filename
        np.savez(filename, train_X=train_X[:i], train_y=train_y[:i])
        
    filename = "{}_{}_{}-trial_test".format(args.drug, args.tissue, trial)
    filename = tissue_directory + filename
    np.savez(filename, test_X=test_X, test_y=test_y)
                    
