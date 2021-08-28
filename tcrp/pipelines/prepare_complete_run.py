from glob import glob
import os

def make_chunks(iterable, nchunks): 
    length = len(iterable)
    chunk_size = (length // nchunks) 
    remainder = length % nchunks

    chunks = []
    i = 0
    while i < length:
        if remainder != 0: 
            end = i + chunk_size + 1
            remainder -= 1
        else: 
            end = i + chunk_size

        chunks.append(iterable[i:end])
        i = end

    return chunks


filepath = os.path.realpath(__file__)
dirname = os.path.dirname(filepath)
home_dir = os.path.dirname(os.path.dirname(dirname))
print home_dir

n_gpus = 20

run_mode = "tcrp" # "tcrp" or "baseline"
run_name = "210803_drug-baseline-models"
out_directory = home_dir + '/output/' + run_name
task_directory = out_directory + '/' + "tasks"

os.system("mkdir -p {}".format(task_directory))

drugs = glob(home_dir + "/data/cell_line_lists/*.pkl")
drugs = [drug.split('/')[-1].split('_tissue_cell_line_list.pkl')[0] for drug in drugs]

with open('priority_drugs') as f: 
    priority = [i.strip() for i in f.readlines()]

missing = set(priority).difference(drugs)
print "Following drugs are missing: "
print missing
print len(missing)

remaining_drugs = list(set(drugs).difference(priority))
priority = list(set(priority).difference(missing))
remaining_drugs_chunked = make_chunks(remaining_drugs, n_gpus)
priority_chunked = make_chunks(priority, n_gpus)

print "Total number of drugs: {}".format(len(priority) + len(remaining_drugs))

fewshot_data_path = "/mnt/beegfs/users/shfong/projects/TCRP-refactored/tcrp-original/output/210726_complete-drug-run-v2/"

for i, (a, b) in enumerate(zip(priority_chunked, remaining_drugs_chunked)):
    _drugs = a + b
    drug_input_file = task_directory + '/drugs_input_{}'.format(i)
    with open(drug_input_file, 'w') as f: 
        f.write('\n'.join(_drugs))


    if run_mode == "tcrp": 
        cmd = ['python', 'generate_MAML_job_cv.py', '--run_name', run_name, '--drug_list_file', drug_input_file, '--job_id', str(i), '--job', 'drugs']
        cmd = ' '.join(cmd)
        print cmd
        os.system(cmd)

    elif run_mode == "baseline": 
        cmd = ['python', dirname + '/generate_baseline_job_cv.py', '--run_name', run_name, '--drug_list_file', drug_input_file, '--job_id', str(i), '--job', 'drugs', '--fewshot_data_path', fewshot_data_path]
        cmd = ' '.join(cmd)
        print cmd
        os.system(cmd)


    break
