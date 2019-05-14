import pandas as pd
import joblib
import os
import sys
import argparse
from tqdm import tqdm

def setup_argparser():
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--path_to_data", type=str, help="path_to_output")
    return argparser


def main(args):
    path_to_data = os.path.join(args.path_to_data, "out")
    metric_dfs = []
    options_dfs = []
    folders = os.listdir(path_to_data)
    for folder in tqdm(folders, total=len(folders)):
        if not os.path.isdir(os.path.join(path_to_data, folder)):
            continue
        if not os.path.isfile(os.path.join(path_to_data, folder, "metrics.p.gz")):
            print("missing file for {0}".format(folder))
            continue
        metric_df = joblib.load(
                os.path.join(path_to_data, folder, "metrics.p.gz")
                )
        metric_df['experiment_id'] = folder.split('_')[0]
        metric_dfs.append(metric_df)
        options = joblib.load(
                os.path.join(path_to_data, folder, "options.p")
                )
        options_dfs.append(options)

    all_metrics_df = pd.concat(metric_dfs, ignore_index=True)
    all_metrics_df.to_csv(os.path.join(path_to_data, "metrics.csv.gz"),
            compression="gzip")
    all_options_df = pd.concat(options_dfs, ignore_index=True, sort=True)
    all_options_df.to_csv(os.path.join(path_to_data, "options.csv"))
    return



if __name__ == "__main__":
    argparser = setup_argparser()
    args = argparser.parse_args()
    print(args)
    main(args)

