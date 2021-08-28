import time
import argparse
import numpy as np
import random
import os
import glob
import sys
import pickle
import copy
from data_loading import *
from utils import *
from score import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from copy import deepcopy


# Training settings
parser = argparse.ArgumentParser()
#work_dic = '/share/data/jinbodata/siqi/Cancer_Drug_Xenograft/'
#data_dic = '/share/data/jinbodata/siqi/Cancer_Drug_Xenograft/tissue_test_data/'
#work_dic = '/cellar/users/samsonfong/Projects/tcrp-v2/from-ma/cell_line_lists/'
work_dic = '/mnt/beegfs/users/shfong/projects/TCRP-refactored/tcrp-original/data/cell_line_lists/'
#data_dic = '/cellar/users/samsonfong/Projects/tcrp-v2/from-ma/drug_feature/'
data_dic = '/mnt/beegfs/users/shfong/projects/TCRP-refactored/tcrp-original/data/drug_feature/'
filepath = os.path.realpath(__file__)
dir_name = os.path.dirname(filepath)

parser.add_argument('--tissue', type=str, default='UPPER_AERODIGESTIVE_TRACT', help='Validation tissue, using the rest tissues for training')
parser.add_argument('--drug', type=str, default='AC220', help='Treated drug')
parser.add_argument('--seed', type=int, default=19, help='Random seed.')
parser.add_argument('--K', type=int, default=5, help='Perform K shot learning')
parser.add_argument('--num_trials', type=int, default=10, help='Number of trials for unseen tissue')
#parser.add_argument('--tissue_list', type=str, default=work_dic + 'cell_line_data/tissue_cell_line_list.pkl', help='Cell line list for different tissues')
parser.add_argument('--log_folder', type=str, default=work_dic+'Log/', help='Log folder')
parser.add_argument('--tissue_num', type=int, default=13, help='Tissue number evolved in the inner update')
parser.add_argument('--run_name', type=str, default='run', help='Run name')
parser.add_argument('--fewshot_data_path', type=str, default=None, help='Path to fewshot data')

args = parser.parse_args()

job_directory = dir_name + '/output/{}/'.format(args.run_name)
if args.fewshot_data_path is None:
	fewshot_data_path = job_directory
else: 
	fewshot_data_path = args.fewshot_data_path

K = args.K
num_trials = args.num_trials

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

drug_tissue_list = work_dic + args.drug + '_tissue_cell_line_list.pkl'
with open(drug_tissue_list, 'rb') as f:
	drug_tissue_map = pickle.load(f)

# Load data
#train_feature, train_label, tissue_index_list, drug_test_feature, drug_test_label, _ = load_data( drug_tissue_map, args.tissue, args.drug, path=data_dic )
train_feature, train_label, tissue_index_list, drug_test_feature, drug_test_label, _ = load_data_cell_line( drug_tissue_map, args.drug, args.tissue, K, path=data_dic )
feature_dim = train_feature.shape[1]

# Here the training process starts

unseen_train_loader_list = []
unseen_test_loader_list = []

# testing_path_suffix = data_dic + args.drug + '/' + args.tissue + '/'
test_data_path = fewshot_data_path + "fewshot_data/" + args.drug + '/' + args.tissue + '/' 

unseen_train_loader_list, unseen_test_loader_list = [], []

for trial in range(num_trials):
	
	# Sample a few shot learning task. Here we use k training, and use the rest for testing. 
	#unseen_train_loader, unseen_test_loader = get_unseen_data_loader(drug_test_feature, drug_test_label, K, args.inner_batch_size)
	unseen_train, unseen_test = [], []

	for k in range(1,K+1):
		# # Sample a few shot learning task. Here we use k training, and use the rest for testing. 

		train_data = np.load(test_data_path + '{}_{}_{}-shot_{}-trial_train.npz'.format(args.drug, args.tissue, k, trial))
		train_X = train_data['train_X']
		train_y = train_data['train_y']
		unseen_train_loader = [(train_X, train_y)]
		
		test_data = np.load(test_data_path + '{}_{}_{}-trial_test.npz'.format(args.drug, args.tissue, trial))
		test_X = test_data['test_X']
		test_y = test_data['test_y']
		unseen_test_loader = [(test_X, test_y)]

		unseen_train.append( unseen_train_loader )
		unseen_test.append( unseen_test_loader )
	 
	unseen_train_loader_list.append(unseen_train)
	unseen_test_loader_list.append(unseen_test)


def train_linear_baseline(Regressor, train_X, train_y, zero_train, zero_test, 
	unseen_train, unseen_test, **kwargs): 

	model = Regressor(**kwargs)
	model.fit(train_X, train_y.ravel())
	zero_p = make_predictions(model, zero_train, zero_test)

	performances = []
	for nt in range(num_trials): 
		inner_p = []
		for k in range(K):
			tmp1 = unseen_train[nt][k]
			tmp2 = unseen_test[nt][k]

			fs_train_X, fs_train_y = tmp1[0]
			fs_test_X, fs_test_y = tmp2[0]

			X = np.vstack([fs_train_X, train_X])
			y = np.vstack([fs_train_y, train_y])

			model = Regressor(**kwargs)
			model.fit(X, y.ravel())
			out = make_predictions(model, fs_test_X, fs_test_y)
			
			inner_p.append(out)
		performances.append(inner_p)

	performances = np.vstack(performances)

	return zero_p, performances


def make_predictions(model, X, y): 
	predictions = model.predict(X)
	out = np.corrcoef(np.vstack([predictions.ravel(), y.ravel()]))

	return out[0,1]


base_line_outpath = job_directory + "baseline_performances/" + args.drug + '/' + args.tissue + '/'
os.system("mkdir -p {}".format(base_line_outpath))

models = [
	("linear", LinearRegression, {}), 
	("KNN", KNeighborsRegressor, {}), 
	("RF", RandomForestRegressor, {'n_estimators': 100, 'n_jobs': -1})
]

results = {}
for name, model, kwargs in models:
	print("Training...", name) 
	zero_p, p = train_linear_baseline(model, train_feature, train_label, drug_test_feature, drug_test_label, 
		unseen_train_loader_list, unseen_test_loader_list, **kwargs)

	results["{}-zero".format(name)] = np.array([zero_p])
	results["{}-fewshot".format(name)] = p

np.savez(
	base_line_outpath + "baseline_performance", 
	**results
)

