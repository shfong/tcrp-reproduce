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
#from data_loading import *
from utils import *
from score import *
from inner_loop import InnerLoop
from mlp import mlp

# Training settings
parser = argparse.ArgumentParser()
work_dic = '/share/data/jinbodata/siqi/Cancer_Drug_Xenograft/'
data_dic = '/share/data/jinbodata/siqi/Cancer_Drug_Xenograft/tissue_test_data/'

parser.add_argument('--tissue', type=str, default='UPPER_AERODIGESTIVE_TRACT', help='Validation tissue, using the rest tissues for training')
parser.add_argument('--drug', type=str, default='AC220', help='Treated drug')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=19, help='Random seed.')
parser.add_argument('--K', type=int, default=5, help='Perform K shot learning')
parser.add_argument('--meta_lr', type=float, default=0.001, help='Learning rate for meta-learning update')
parser.add_argument('--meta_batch_size', type=int, default=32, help='Meta-learning batch size, i.e. how many different tasks we need sample')
parser.add_argument('--inner_batch_size', type=int, default=10, help='Batch size for each individual learning job')
parser.add_argument('--inner_lr', type=float, default=0.001, help='Learning rate for ')
parser.add_argument('--test_inner_lr', type=float, default=0.001, help='Learning rate for ')
parser.add_argument('--num_updates', type=int, default=1000, help='Number of training epochs')
parser.add_argument('--num_inner_updates', type=int, default=1, help='Initial learning rate')
parser.add_argument('--num_out_updates', type=int, default=20, help='Final learning rate')
parser.add_argument('--num_trials', type=int, default=10, help='Number of trials for unseen tissue')
parser.add_argument('--layer', type=int, default=1, help='Number of layers of NN for single task')
parser.add_argument('--hidden', type=int, default=5, help='Number of hidden units of NN for single task')
parser.add_argument('--patience', type=int, default=3, help='Patience')
parser.add_argument('--tissue_list', type=str, default=work_dic + 'cell_line_data/tissue_cell_line_list.pkl', help='Cell line list for different tissues')
parser.add_argument('--log_folder', type=str, default=work_dic+'Log/', help='Log folder')
parser.add_argument('--tissue_num', type=int, default=13, help='Tissue number evolved in the inner update')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
K = args.K
num_trials = args.num_trials
meta_batch_size = args.meta_batch_size

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

drug_tissue_list = work_dic + 'cell_line_data/' + args.drug + '_tissue_cell_line_list.pkl'
with open(drug_tissue_list, 'rb') as f:
	drug_tissue_map = pickle.load(f)

# Load data
train_feature, train_label, tissue_index_list, drug_test_feature, drug_test_label, _ = load_data( drug_tissue_map, args.tissue, args.drug )
feature_dim = train_feature.shape[1]

# This is the meta-learning model
observed_tissue_model = mlp(feature_dim, args.layer, args.hidden)
observed_opt = torch.optim.Adam(observed_tissue_model.parameters(), lr=args.meta_lr, betas=(0.9, 0.99), eps=1e-05)
#observed_opt = torch.optim.SGD(observed_tissue_model.parameters(), lr=args.meta_lr)
#observed_tissue_model.eval()

# This is the inside learner
inner_net = InnerLoop(args.num_inner_updates, args.inner_lr, feature_dim, args.layer, args.hidden)

if args.cuda:
	torch.cuda.manual_seed(args.seed)
	observed_tissue_model.cuda()
	inner_net.cuda()

def zero_shot_test(unseen_test_loader):

	unseen_tissue_model = mlp( feature_dim, args.layer, args.hidden )

	# First need to copy the original meta learning model
	unseen_tissue_model.copy_weights( observed_tissue_model )
	unseen_tissue_model.cuda()
	
	unseen_tissue_model.eval()

	mtest_loss, mtest_pear_corr, test_prediction, test_true_label = evaluate_new( unseen_tissue_model, unseen_test_loader, 0 )

	return mtest_loss, mtest_pear_corr, test_prediction, test_true_label

def unseen_tissue_learn(unseen_train_loader, unseen_test_loader):

	unseen_tissue_model = mlp( feature_dim, args.layer, args.hidden )
	
	# First need to copy the original meta learning model	
	unseen_tissue_model.copy_weights( observed_tissue_model )	
	unseen_tissue_model.cuda()
	#unseen_tissue_model.train()
	unseen_tissue_model.eval()

	unseen_opt = torch.optim.SGD(unseen_tissue_model.parameters(), lr=args.inner_lr)
	#unseen_opt = torch.optim.Adam(unseen_tissue_model.parameters(), lr=args.test_inner_lr, betas=(0.9, 0.99), eps=1e-05)

	# Here test_feature and test_label contains only one tissue info
	#unseen_train_loader, unseen_test_loader = get_unseen_data_loader(test_feature, test_label, K, args.inner_batch_size)

	for i in range(args.num_inner_updates):
		
		in_, target = unseen_train_loader.__iter__().next()
		loss, _  = forward_pass( unseen_tissue_model, in_, target )
		unseen_opt.zero_grad()
		loss.backward()
		unseen_opt.step()

	# Test on the rest of cell lines in this tissue (unseen_test_loader)
	mtrain_loss, mtrain_pear_corr, _, _ = evaluate_new( unseen_tissue_model, unseen_train_loader,1 )
	mtest_loss, mtest_pear_corr, test_prediction, test_true_label = evaluate_new( unseen_tissue_model, unseen_test_loader,0 )

	return mtrain_loss, mtrain_pear_corr, mtest_loss, mtest_pear_corr, test_prediction, test_true_label

def meta_update(test_loader, ls):

	#print 'Meta update'
	in_, target = test_loader.__iter__().next()
	
	# We use a dummy forward / backward pass to get the correct grads into self.net
	loss, out = forward_pass(observed_tissue_model, in_, target)
	
	# Unpack the list of grad dicts
	gradients = { k: sum(d[k] for d in ls) for k in ls[0].keys() }
	
	#for k, val, in gradients.items():
	#	gradients[k] = val / args.meta_batch_size
	#	print k,':',gradients[k]

	# Register a hook on each parameter in the net that replaces the current dummy grad
	# with our grads accumulated across the meta-batch
	hooks = []
	for (k,v) in observed_tissue_model.named_parameters():
		
		def get_closure():
			key = k
			def replace_grad(grad):
				return gradients[key]
			return replace_grad
		
		hooks.append(v.register_hook(get_closure()))
	
	# Compute grads for current step, replace with summed gradients as defined by hook
	observed_opt.zero_grad()
	
	loss.backward()
	# Update the net parameters with the accumulated gradient according to optimizer
	observed_opt.step()
	# Remove the hooks before next training phase
	for h in hooks:
		h.remove()

# Here the training process starts

train_loss, train_corr, train_spear_corr = np.zeros((args.num_updates,K)), np.zeros((args.num_updates,K)), np.zeros((args.num_updates,K))
test_loss, test_corr, test_spear_corr = np.zeros((args.num_updates,K)), np.zeros((args.num_updates,K)), np.zeros((args.num_updates,K))

best_loss = 10000
best_epoch = 0

unseen_train_loader_list = []
unseen_test_loader_list = []

testing_path_suffix = data_dic + args.drug + '/' + args.tissue + '/'

unseen_train_loader_list, unseen_test_loader_list = [], []

for trial in range(num_trials):
	
	# Sample a few shot learning task. Here we use k training, and use the rest for testing. 
	#unseen_train_loader, unseen_test_loader = get_unseen_data_loader(test_feature, test_label, K, args.inner_batch_size)
	unseen_train, unseen_test = [], []

	for k in range(1,K+1):

		train_index_file = testing_path_suffix + args.tissue + '_' + args.drug + '_train_index_' + str(trial) + '_' + str(k) + '.npy'
		test_index_file = testing_path_suffix + args.tissue + '_' + args.drug + '_test_index_' + str(trial) + '_' + str(k) + '.npy'

		unseen_train_loader, unseen_test_loader = load_unseen_data_loader(train_index_file, test_index_file, drug_test_feature, drug_test_label, K, trial, batch_size=K)
		unseen_train.append( unseen_train_loader )
		unseen_test.append( unseen_test_loader )
	
	unseen_train_loader_list.append(unseen_train)
	unseen_test_loader_list.append(unseen_test)

zero_shot_test_dataset = du.TensorDataset( torch.FloatTensor(drug_test_feature), torch.FloatTensor(drug_test_label) )
zero_test_loader = du.DataLoader( zero_shot_test_dataset, batch_size=200 )
zero_test_data_list = []
for batch_feature, batch_label in zero_test_loader:
	zero_test_data_list.append((batch_feature.cuda(), batch_label.cuda()))

predict_folder = work_dic + 'MAML_prediction/' + args.drug + '/' + args.tissue + '/'
mkdir_cmd = 'mkdir -p ' + predict_folder
os.system(mkdir_cmd)

for epoch in range( args.num_updates ):

	zero_test_loss, zero_test_corr, test_prediction, test_true_label = zero_shot_test(zero_test_data_list)
	print '0 Few shot', epoch, 'meta training:', '-1', '-1', zero_test_loss, zero_test_corr

	epoch_folder = predict_folder + 'epochs_' + str(epoch) + '/'
	mkdir_cmd = 'mkdir -p ' + epoch_folder
	os.system(mkdir_cmd)

	zero_predict_file = epoch_folder + 'zero_shot_predict.npy'
	np.save(zero_predict_file, test_prediction)

	zero_true_file = epoch_folder + 'zero_shot_true.npy'
	np.save(zero_true_file, test_true_label)

	for k in range(1, K+1):

		tissue_train_loss, tissue_test_loss, tissue_train_corr, tissue_test_corr, = np.zeros((num_trials,)), np.zeros((num_trials,)), np.zeros((num_trials,)), np.zeros((num_trials,))

		for i in range( num_trials ):	
		# Evaluate on unseen test tasks
			tissue_train_loss[i], tissue_train_corr[i], tissue_test_loss[i], tissue_test_corr[i], test_prediction, test_true_label = unseen_tissue_learn( unseen_train_loader_list[i][k-1], unseen_test_loader_list[i][k-1] )

			k_shot_predict_file = epoch_folder + str(k) + '_' + str(i) + '_shot_predict.npy'
			np.save(k_shot_predict_file, test_prediction)

			k_shot_true_file = epoch_folder + str(k) + '_' + str(i) + '_shot_true.npy'
			np.save(k_shot_true_file, test_true_label)

		train_loss[epoch][k-1], train_corr[epoch][k-1] = tissue_train_loss.mean(), tissue_train_corr.mean()
		test_loss[epoch][k-1], test_corr[epoch][k-1] = tissue_test_loss.mean(), tissue_test_corr.mean()
	
		print k, 'Few shot', epoch, 'meta training:', train_loss[epoch][k-1], train_corr[epoch][k-1], test_loss[epoch][k-1], test_corr[epoch][k-1]

	# Collect a meta batch update
	grads = []
	
	meta_train_loss, meta_train_corr, meta_val_loss, meta_val_corr = np.zeros((meta_batch_size,)), np.zeros((meta_batch_size,)), np.zeros((meta_batch_size,)), np.zeros((meta_batch_size,))
	
	for i in range( meta_batch_size ):
	
		observed_train_loader, observed_test_loader = get_observed_data_loader( train_feature, train_label, tissue_index_list, K, args.inner_batch_size, args.tissue_num )
		#observed_train_loader, observed_test_loader = get_observed_data_loader( train_feature, train_label, tissue_index_list, K, K)
	
		inner_net.copy_weights( observed_tissue_model )

		metrics, g = inner_net.forward( observed_train_loader, observed_test_loader )

		grads.append( g )
			
		meta_train_loss[i], meta_train_corr[i], meta_val_loss[i], meta_val_corr[i] = metrics

	if meta_val_loss.mean() < best_loss:
	   	best_loss = meta_val_loss.mean()
		best_epoch = epoch
		bad_counter = 0
	else:
		bad_counter += 1

	if bad_counter == args.patience:
		break

	print 'Meta update', epoch, meta_train_loss.mean(), meta_train_corr.mean(), meta_val_loss.mean(), meta_val_corr.mean(), 'best epoch', best_epoch
	# Perform the meta update
	meta_update( observed_test_loader, grads )

print 'Best loss meta training:', test_corr[best_epoch]
#print 'Best loss meta training:', train_loss[best_epoch], train_corr[best_epoch], test_loss[best_epoch], test_corr[best_epoch] 
