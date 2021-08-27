# tcrp-original
TCRP codebase that adheres to the original code as much as possible

## Instructions for a complete run
The code should all be contained in `prepare_complete_run.py`. This script will create a directory that contains all of commands to sweep through all hyperpameter for all of the specific drugs. The drugs analyzed correspond to the pickle files in `data/cell_line_lists`. Code to generate the pickled files still need to be included in this repository. Feel free to edit the `run_name` variable to change the run name. 

After the code is generated, the slurm submission scrips are created in `output/{RUN NAME}/MAML_cmd`. To submit all of the slurm scripts you can run the following: 
```ls run_MAML_drugs*.sh | awk '{k = "sbatch "$0""; system(k); print(k)}'```

## Parsing results
The results are all embedded as logs in `output/{RUN NAME}/run-logs/{DRUG}/{TISSUE}`. The log will specify the selected epoch for that hyperparameter and the correspond test performance. Additional code will be needed to gather the best performance to select the final performance for task.



