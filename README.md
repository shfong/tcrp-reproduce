# tcrp-reproduce
Refactored TCRP codebase with improved organization and additional code for data transformation and model selection. The original codebase can be accessed [here](https://github.com/idekerlab/TCRP). For high level questions regarding TCRP, check out the [TCRP FAQ](https://github.com/shfong/tcrp-reproduce/blob/public/tcrp-faq.md). 

## Instructions for a complete run
End-to-end run is quite completely glued together yet and will require a little bit of manual work. 

### Gathering data

This part of the pipeline is not automated yet. The raw data will need to be downloaded from DepMap, and the transformed data are generated in with a jupyter notebook `tcrp/data_preparation/process_sanger_drug_cell_line.ipynb`. This notebook will generate a series of pickled files and numpy compressed files that the following steps will be dependent on. 

### TCRP complete run

The code should all be contained in `prepare_complete_run.py`. This script will create a directory that contains all of commands to sweep through all hyperpameter for all of the specific drugs. The drugs analyzed correspond to the pickle files in `data/cell_line_lists`. Code to generate the pickled files still need to be included in this repository. Feel free to edit the `run_name` variable to change the run name. 

After the code is generated, the slurm submission scrips are created in `output/{RUN NAME}/MAML_cmd`. To submit all of the slurm scripts you can run the following: 
```ls run_MAML_drugs*.sh | awk '{k = "sbatch "$0""; system(k); print(k)}'```

### Baseline run

Edit `prepare_complete_run.py`. Change `run_mode` variable to `baseline` to run `generate_baseline_jobs.py`. In addition, point to the correct `fewshot_data_path`. This is a directory that was created in the tcrp complete run. It's simply the fewshot training and testing dataset that was used in the complete run.


## Parsing results
The results are all embedded as logs in `output/{RUN NAME}/run-logs/{DRUG}/{TISSUE}`. The log will specify the selected epoch for that hyperparameter and the correspond test performance. Additional code will be needed to gather the best performance to select the final performance for task.

## Reusability efforts

Emily So, from the Haibe-Kains lab, has reproduced and extended TCRP. Their repository can be found [here](https://github.com/bhklab/TCRP_Reusability_Report). A runnable version is also available [here](https://codeocean.com/capsule/8411716/tree/v2).
