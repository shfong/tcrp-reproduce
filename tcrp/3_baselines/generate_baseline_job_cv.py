import time
import numpy as np
import random
import os
import sys
import pickle
import copy
import argparse

parser = argparse.ArgumentParser(description="Generate slurm scripts to run experiment")
parser.add_argument('--drug_list_file', default='selected_drug_list', help='list of drugs to generate code for')
parser.add_argument('--job', default='job', help='Job name for slurm script')
parser.add_argument('--job_id', default='', help='Jod ID to append to create a separate script if needed')
parser.add_argument('--run_name', default='run')
parser.add_argument('--fewshot_data_path', default=None)

args = parser.parse_args()
drug_list_file, job, job_id = args.drug_list_file, args.job, args.job_id


#work_dic = '/share/data/jinbodata/siqi/Drug_data/'
#work_dic = '/share/data/jinbodata/siqi/mut_exp_cnv_data/challenge_1b/'
#work_dic = '/cellar/users/samsonfong/Projects/tcrp-v2/from-ma/cell_line_lists/'
work_dic = '/mnt/beegfs/users/shfong/projects/TCRP-refactored/tcrp-original/data/cell_line_lists/'
filepath = os.path.realpath(__file__)
dir_name = os.path.dirname(filepath)
job_directory = dir_name + '/output/{}/'.format(args.run_name)

file_handle = open( drug_list_file )

cmd_folder = job_directory + 'baseline_cmd/'
os.system("mkdir -p {}".format(cmd_folder))
cmd_list = []

fewshot_data_path_str = ''
if args.fewshot_data_path is not None: 
	fewshot_data_path_str = ' --fewshot_data_path {}'.format(args.fewshot_data_path)

for line in file_handle:

	gene = line.rstrip()

	tissue_list = work_dic + gene + '_tissue_cell_line_list.pkl'

	with open(tissue_list, 'rb') as f:
		tissue_map = pickle.load(f)

	for tissue, tissue_cell_line in tissue_map.items():
		if len(tissue_cell_line) < 15:
			continue

		# Assuming fewshot samples already exist...
		# cmd_str = '$python ' + dir_name + '/' + 'generate_fewshot_samples.py ' + '--tissue {} --drug {} --K 10 --num_trials 20 --run_name {}'.format(tissue, gene, args.run_name)
		# cmd_list.append(cmd_str)
		
		cmd_str = '$python ' + dir_name + '/' + 'baseline_DRUG.py --tissue ' + tissue + ' --drug ' + gene + ' --K 10 --num_trials 20' + ' --run_name ' + args.run_name  + fewshot_data_path_str
		cmd_list.append( cmd_str )						

file_handle.close()


subcommand_directory = cmd_folder + "subcommands"
os.system("mkdir -p {}".format(subcommand_directory))
with open(subcommand_directory + '/' + 'subcommands_baseline_{}{}.sh'.format(job, job_id), 'w') as f:
	f.write('#!/bin/bash\n')
	f.writelines("python=/cellar/users/shfong/bin/miniconda3/envs/tcrp/bin/python\n")
	f.writelines('\n'.join(cmd_list) + '\n')


tempfile = cmd_folder + 'run_baseline_{}{}.sh'.format(job, job_id)

slurm_output = job_directory + "baseline-slurm-logs"
os.system("mkdir -p {}".format(slurm_output))

file_handle = open(tempfile,'w')

file_handle.writelines('#!/bin/bash\n')
file_handle.writelines("#SBATCH --job-name {}{}\n".format(job, job_id))
file_handle.writelines("#SBATCH --output={}/{}{}.%j\n".format(slurm_output, job, job_id))
file_handle.writelines("#SBATCH --cpus-per-task=16\n")
file_handle.writelines("#SBATCH --mem=64G\n")
file_handle.writelines("#SBATCH --partition=nrnb-compute\n")
# file_handle.writelines("#SBATCH --account=nrnb-gpu\n")
# file_handle.writelines("#SBATCH --gres=gpu:1\n\n")

#file_handle.writelines("python=/cellar/users/samsonfong/bin/miniconda/envs/tcrp/bin/python\n")
file_handle.writelines("/usr/bin/bash {}/subcommands_baseline_{}{}.sh\n".format(subcommand_directory, job, job_id))

file_handle.close()

