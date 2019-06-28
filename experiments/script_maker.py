""" This python script creates the bash files for running the experiments
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

CUDA_AVAILABLE = True
NUM_SPLITS = 1

# Helper Functions for creating Scripts
def convert_dict_to_arg_string(arg_dict, sep = " "):
    # Convert python dict to argparse string
    string_args = ""
    for k, v in arg_dict.items():
        if v is None:
            string_args += k + sep
        elif type(v) == bool:
            if v:
                string_args += k + sep
            else:
                # Skip if False
                continue
        elif pd.isna(v):
            # Skip value if missing
            continue
        else:
            string_args += k + sep + str(v) + sep
    return(string_args)

def create_script(arg_dicts, script_path, python_script_path,
        pre_script_commands=None, post_script_commands=None,
        splits=1):
    njobs = len(arg_dicts)
    if splits == 1:
        print("Creating Script at {0} with {1} jobs".format(script_path, njobs))
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("\n")
            if pre_script_commands is not None:
                f.write(pre_script_commands)

            for arg_dict in arg_dicts:
                python_script_args = convert_dict_to_arg_string(arg_dict)
                f.write("\n")
                f.write("python " + python_script_path + " " + python_script_args)
                f.write("\n")

            f.write("\n")
            if post_script_commands is not None:
                f.write(post_script_commands)
            f.write("\n#EOF")
        os.chmod(script_path, 0o755)
        print("... Done Creating Script at {0}!".format(script_path))
    else:
        print("Splitting {0} into {1} Scripts ...".format(njobs, splits))
        jobs_per_split = int(np.ceil(njobs/splits))
        for split_num in range(splits):
            job_path = script_path.rstrip('.sh') + '_{0}.sh'.format(split_num)
            if split_num < splits - 1:
                job_args = arg_dicts[split_num*jobs_per_split:
                                     (split_num+1)*jobs_per_split]
            else:
                job_args = arg_dicts[split_num*jobs_per_split:]
            create_script(
                arg_dicts=job_args,
                script_path=job_path,
                python_script_path=python_script_path,
                pre_script_commands=pre_script_commands,
                post_script_commands=post_script_commands,
                splits=1,
                )
        print("... Done Creating {0} Scripts!".format(splits))
    return


# Create Bash Script Files To Run
def setup_copy_fixed_memory_script(path_to_scripts, experiment_id=0, num_splits=1):
    experiment_name = 'copy_fixed-lstm'
    experiment_folder = './output/{0}'.format(experiment_name)
    python_script_path = "./experiments/synth_script.py"
    script_path = "./scripts/{0}.sh".format(experiment_name)

    common_args = {
        '--experiment_folder': [experiment_folder],
        '--data_name': ['./data/copy_fixed'],
        '--data_type': ['copy'],
        '--data_lag': [10],
        '--data_minlag': [10],
        '--emsize': [6],
        '--model' : ['LSTM'],
        '--nhid': [50],
        '--nlayers': [2],
        '--weight_decay': [10**-5],
        '--cuda': [True] if CUDA_AVAILABLE else [False],
        '--dropout': [0.0],
        '--lr': [1.0],
        '--scale_lr_K': [True],
        '--decay_lr': [False],
        '--init_num': [0,1,2,3,4,5,6,7,8],
        }
    arg_dicts = []
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [5, 10, 15, 20, 30],
        '--tbptt_style': ['original-buffer'],
        '--init_num': [0],
        **common_args,
        }))))
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [15],
        '--adaptive_K': [True],
        '--beta_estimate_method': ['ols'],
        '--tbptt_style': ['original-buffer'],
        '--delta': [0.9, 0.5, 0.1],
        '--init_num': [0],
        **common_args,
        }))))

    arg_df = pd.concat(arg_dicts, ignore_index=True, sort=True)
    arg_df = arg_df.sort_values(['--tbptt_style', '--K', '--delta', '--lr'],
            ascending=[False, True, False, False])
    arg_df['--experiment_id'] = experiment_id + np.arange(arg_df.shape[0])
    arg_dicts = arg_df.to_dict('records')

    pre_script_commands = ""
    post_script_commands = "python ./experiments/aggregate_output.py --path_to_data {0}".format(experiment_folder)

    create_script(arg_dicts=arg_dicts, script_path=script_path, python_script_path=python_script_path, pre_script_commands=pre_script_commands, post_script_commands=post_script_commands, splits=num_splits)
    return len(arg_dicts)

def setup_copy_variable_memory_script(path_to_scripts, experiment_id=0, num_splits=1):
    experiment_name = 'copy_variable-lstm'
    experiment_folder = './output/{0}'.format(experiment_name)
    python_script_path = "./experiments/synth_script.py"
    script_path = "./scripts/{0}.sh".format(experiment_name)

    common_args = {
        '--experiment_folder': [experiment_folder],
        '--data_name': ['./data/copy_variable'],
        '--data_type': ['copy'],
        '--data_lag': [10],
        '--data_minlag': [5],
        '--emsize': [6],
        '--model' : ['LSTM'],
        '--nhid': [50],
        '--nlayers': [2],
        '--weight_decay': [10**-5],
        '--cuda': [True] if CUDA_AVAILABLE else [False],
        '--dropout': [0.0],
        '--lr': [1.0],
        '--scale_lr_K': [True],
        '--decay_lr': [False],
        '--init_num': [0,1,2,3,4,5,6,7,8],
        }
    arg_dicts = []
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [5, 10, 15, 20, 30],
        '--tbptt_style': ['original-buffer'],
        '--init_num': [0],
        **common_args,
        }))))
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [15],
        '--adaptive_K': [True],
        '--beta_estimate_method': ['ols'],
        '--tbptt_style': ['original-buffer'],
        '--delta': [0.9, 0.5, 0.1],
        '--init_num': [0],
        **common_args,
        }))))

    arg_df = pd.concat(arg_dicts, ignore_index=True, sort=True)
    arg_df = arg_df.sort_values(['--tbptt_style', '--K', '--delta', '--lr'],
            ascending=[False, True, False, False])
    arg_df['--experiment_id'] = experiment_id + np.arange(arg_df.shape[0])
    arg_dicts = arg_df.to_dict('records')

    pre_script_commands = ""
    post_script_commands = "python ./experiments/aggregate_output.py --path_to_data {0}".format(experiment_folder)

    create_script(arg_dicts=arg_dicts, script_path=script_path, python_script_path=python_script_path, pre_script_commands=pre_script_commands, post_script_commands=post_script_commands)
    return len(arg_dicts)

def setup_ptb_script(path_to_scripts, experiment_id=0, num_splits=1):
    experiment_name = 'ptb-lstm'
    experiment_folder = './output/{0}'.format(experiment_name)
    script_path = "./scripts/{0}.sh".format(experiment_name)
    python_script_path = "./experiments/language_script.py"

    common_args = {
        '--experiment_folder': [experiment_folder],
        '--data': ['./data/ptb'],
        '--model' : ['LSTM'],
        '--emsize': [900],
        '--nhid': [900],
        '--nlayers': [1],
        '--weight_decay': [10**-7],
        '--cuda': [True] if CUDA_AVAILABLE else [False],
        '--dropout': [0.2],
        '--lr': [1, 10.0],
        '--scale_lr_K': [True],
        '--decay_lr': [False],
        '--tied': [True],
        '--batch_size': [32],
        '--init_num': [0,1,2,3,4,5,6,7,8],
        }
    arg_dicts = []
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [10, 50, 100, 200, 300],
        '--tbptt_style': ['original-buffer'],
        '--init_num': [0],
        **common_args,
        }))))
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [100],
        '--adaptive_K': [True],
        '--beta_estimate_method': ['ols'],
        '--tbptt_style': ['original-buffer'],
        '--delta': [0.9, 0.5, 0.1],
        '--init_num': [0],
        **common_args,
        }))))

    arg_df = pd.concat(arg_dicts, ignore_index=True, sort=True)
    arg_df = arg_df.sort_values(['--tbptt_style', '--K', '--delta', '--lr'],
            ascending=[False, True, False, False])
    arg_df['--experiment_id'] = experiment_id + np.arange(arg_df.shape[0])
    arg_dicts = arg_df.to_dict('records')

    pre_script_commands = ""
    post_script_commands = "python ./experiments/aggregate_output.py --path_to_data {0}".format(experiment_folder)


    create_script(arg_dicts=arg_dicts, script_path=script_path, python_script_path=python_script_path, pre_script_commands=pre_script_commands, post_script_commands=post_script_commands, splits=num_splits)
    return len(arg_dicts)

def setup_wiki_script(path_to_scripts, experiment_id=0, num_splits=1):
    # Setup Lag 10 Repeat Script
    experiment_name = 'wiki2-lstm'
    experiment_folder = './output/{0}'.format(experiment_name)
    script_path = "./scripts/{0}.sh".format(experiment_name)
    python_script_path = "./experiments/language_script.py"

    common_args = {
        '--experiment_folder': [experiment_folder],
        '--data': ['./data/wikitext-2'],
        '--epoch': [50],
        '--model' : ['LSTM'],
        '--emsize': [512],
        '--nhid': [512],
        '--nlayers': [1],
        '--weight_decay': [10**-7],
        '--cuda': [True] if CUDA_AVAILABLE else [False],
        '--dropout': [0.2],
        '--lr': [10.0],
        '--scale_lr_K': [True],
        '--decay_lr': [False],
        '--tied': [True],
        '--max_train_time': [3*3600],
        '--linear_scale': [True],
        '--init_num': [0,1,2,3,4,5,6,7,8],
        }
    arg_dicts = []
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [10, 50, 100, 200, 300],
        '--tbptt_style': ['original-buffer'],
        '--init_num': [0],
        **common_args,
        }))))
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [100],
        '--adaptive_K': [True],
        '--beta_estimate_method': ['ols'],
        '--tbptt_style': ['original-buffer'],
        '--delta': [0.9, 0.5, 0.1],
        '--init_num': [0],
        **common_args,
        }))))

    arg_df = pd.concat(arg_dicts, ignore_index=True, sort=True)
    arg_df = arg_df.sort_values(['--tbptt_style', '--K', '--delta', '--lr'],
            ascending=[False, True, False, False])
    arg_df['--experiment_id'] = experiment_id + np.arange(arg_df.shape[0])
    arg_dicts = arg_df.to_dict('records')

    pre_script_commands = ""
    post_script_commands = "python ./experiments/aggregate_output.py --path_to_data {0}".format(experiment_folder)

    create_script(arg_dicts=arg_dicts, script_path=script_path, python_script_path=python_script_path, pre_script_commands=pre_script_commands, post_script_commands=post_script_commands, splits=num_splits)
    return len(arg_dicts)

def setup_book_order_script(path_to_scripts, experiment_id=0, num_splits=1):
    experiment_name = 'book_order-lstm'
    experiment_folder = './output/{0}'.format(experiment_name)
    script_path = "./scripts/{0}.sh".format(experiment_name)
    python_script_path = "./experiments/temporal_pp_script.py"

    common_args = {
        '--experiment_folder': [experiment_folder],
        '--data_path': ['./data/tpp-book-order'],
        '--data_split_number': [1],
        '--data_time_scale': [1],
        '--time_onehot_split': [True],
        '--emsize': [128],
        '--model' : ['LSTM'],
        '--epochs': [51],
        '--nhid': [128],
        '--nlayers': [2],
        '--weight_decay': [10**-3],
        '--cuda': [True] if CUDA_AVAILABLE else [False],
        '--dropout': [0.0],
        '--lr': [0.1],
        '--decay_lr': [False],
        '--optim': ['SGD'],
        '--init_num': [0,1,2,3,4,5,6,7,8],
        }
    arg_dicts = []
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [3,6,9,15,21],
        '--tbptt_style': ['original-buffer'],
        '--scale_lr_K': [False],
        **common_args,
        }))))
    arg_dicts.append(pd.DataFrame(list(ParameterGrid({
        '--K': [6],
        '--adaptive_K': [True],
        '--beta_estimate_method': ['ols'],
        '--tbptt_style': ['original-buffer'],
        '--delta': [0.1, 0.5, 0.9],
        '--scale_lr_K': [True],
        **common_args,
        }))))

    arg_df = pd.concat(arg_dicts, ignore_index=True, sort=True)
    arg_df = arg_df.sort_values(['--init_num', '--tbptt_style', '--K', '--delta', '--lr'],
            ascending=[True, False, True, False, False])
    arg_df['--experiment_id'] = experiment_id + np.arange(arg_df.shape[0])
    arg_dicts = arg_df.to_dict('records')

    pre_script_commands = ""
    post_script_commands = "python ./experiments/aggregate_output.py --path_to_data {0}".format(experiment_folder)
    create_script(arg_dicts=arg_dicts, script_path=script_path, python_script_path=python_script_path, pre_script_commands=pre_script_commands, post_script_commands=post_script_commands, splits=num_splits)
    return len(arg_dicts)



if __name__ == "__main__":
    print("Generating Scripts...")

    path_to_scripts = './scripts'
    if not os.path.isdir(path_to_scripts):
        os.makedirs(path_to_scripts)

    # Setup Experiment Scripts
    setup_copy_fixed_memory_script(path_to_scripts, num_splits=NUM_SPLITS)
    setup_copy_variable_memory_script(path_to_scripts, num_splits=NUM_SPLITS)
    setup_ptb_script(path_to_scripts, num_splits=NUM_SPLITS)
    setup_wiki_script(path_to_scripts, num_splits=NUM_SPLITS)
    setup_book_order_script(path_to_scripts, num_splits=NUM_SPLITS)

    print("...Done")

#EOF
