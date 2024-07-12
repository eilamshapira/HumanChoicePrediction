import pandas as pd 
import wandb
from tqdm import tqdm
import requests
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics

class wandb_results:
    def __init__(self, project_id, wandb_username="eilamshapira"): 
        self.api = wandb.Api(timeout=60)
        self.project_id = project_id
        self.wandb_username = wandb_username

    def get_sweep_results(self, sweep_id, metric="accuracy_all", best_epoch=False, get_servers=False, reset_api=False, read_csv_if_exist=True, save_to_csv=True):
        if reset_api:
            self.reset_api()

        print(f"Download {sweep_id=} data...")
        runs = self.api.sweep(f"{self.wandb_username}/{self.project_id}/{sweep_id}").runs
        n_runs = len(runs)
        path = f"sweeps_csvs/{sweep_id}_{n_runs}.csv"
        if read_csv_if_exist and os.path.exists(path):
            return pd.read_csv(path, index_col=0)
        summary_list, config_list, name_list = [], [], []
        for run in tqdm(runs): 
            summary_list.append(run.summary._json_dict)
            config_list.append(
                {k: v for k,v in run.config.items()
                  if not k.startswith('_')})
            name_list.append(run.name)

        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
            })
        config_cols = pd.json_normalize(runs_df['config'])
        config_cols.columns = [f"config_{c}" for c in config_cols.columns]
        summary_cols = pd.json_normalize(runs_df['summary'])
        runs_df = pd.concat([runs_df, config_cols, summary_cols], axis=1)
        runs_df.drop(['config', 'summary'], axis=1, inplace=True)
        hpt = [c for c in config_cols.columns if c not in ["config_seed", "config_run_hash"]]
        if save_to_csv: runs_df.to_csv(path)
        return runs_df
    
    def get_sweeps_results(self, sweeps, metric="accuracy_all", best_epoch=False, get_servers=False,  read_csv_if_exist=True, save_to_csv=True):
        print("Total number of sweeps:", len(sweeps))
        j = pd.concat([self.get_sweep_results(sweep, metric=metric, best_epoch=best_epoch,  get_servers=get_servers, save_to_csv=save_to_csv, read_csv_if_exist=read_csv_if_exist) for sweep in sweeps])
        j = j.reset_index(drop=True)
        return j
    
    def reset_api(self):
        self.api = wandb.Api()

def result_metric(sweeps, group_name, drop_list=[0], drop_HPT=False, metric="accuracy_per_mean_user_and_bot", epoch="best"):
    api = wandb_results("NLP2024_PROJECT_BellaPerel", wandb_username="bellatomer")
    df = api.get_sweeps_results(sweeps, metric=metric)

    config_cols = [c for c in df.columns if "config_" in c and c!="config_wandb_run_id" and c!="config_online_simulation_size"]
    HPT_cols = [col for col in config_cols if df[col].nunique() > 1]
    print(f"Hyperparameter columns (HPT_cols): {HPT_cols}")

    if drop_HPT:
        df = df.drop([c for c in HPT_cols if not c in ["config_LLM_SIM_SIZE", "config_seed"]], axis=1)
        HPT_cols = ["config_LLM_SIM_SIZE", "config_seed"]

    # Remove non-numeric columns before computing mean and std
    numeric_cols = df.select_dtypes(include=np.number).columns
    df_numeric = df[numeric_cols]
    print(f"Numeric columns: {numeric_cols}")

    # Check if the columns to group by exist in df_numeric
    groupby_cols = [c for c in HPT_cols if c in df_numeric.columns and c != "config_seed"]
    print(f"Grouping by columns: {groupby_cols}")

    if not groupby_cols:
        raise KeyError("None of the HPT columns are present in the numeric DataFrame")

    grouped = df_numeric.groupby(groupby_cols)

    mean_df = grouped.mean()
    std_df = grouped.std()

    # Re-add non-numeric columns before computing best_col
    for col in config_cols:
        if col not in mean_df.columns:
            mean_df[col] = df[col]

    # Print columns that include the metric
    if epoch == "best":
        metric_columns = [c for c in mean_df.columns if metric in c]
        print(f"Metric columns that include '{metric}': {metric_columns}")
        if not metric_columns:
            raise ValueError("No columns match the specified metric for the 'best' epoch")
        best_col = mean_df[metric_columns].idxmax(axis=1)
    else:
        metric_columns = [c for c in mean_df.columns if f"{metric}_epoch{epoch}" in c]
        print(f"Metric columns for epoch {epoch} that include '{metric}': {metric_columns}")
        if not metric_columns:
            raise ValueError(f"No columns match the specified metric for epoch {epoch}")
        best_col = mean_df[metric_columns].idxmax(axis=1)

    result = grouped.apply(lambda x: x[best_col.loc[x.name]].values)
    means = grouped.apply(lambda x: x[best_col.loc[x.name]].mean())
    stds = grouped.apply(lambda x: x[best_col.loc[x.name]].std())

    df_cols = {'mean': means, 'std': stds, 'values': result.values}
    if epoch == "best":
        df_cols['epoch'] = best_col.apply(lambda x: int(x.split("epoch")[1]) if "epoch" in x else "last")

    df_cols['CI'] = result.apply(lambda x: bootstrap_ci(x))

    summary_df = pd.DataFrame(df_cols, index=best_col.index)
    for d in drop_list:
        if d in summary_df.index:
            summary_df = summary_df.drop(d)
    if len(summary_df.index.names) == 1:
        return summary_df.rename_axis(group_name)
    else:
        return summary_df

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound

BASE_METRIC = "accuracy_per_mean_user_and_bot"


def result_metric(sweeps, group_name, metric="accuracy_per_mean_user_and_bot"):
    api = wandb_results("NLP2024_PROJECT_BellaPerel", wandb_username="bellatomer")
    df = api.get_sweeps_results(sweeps, metric=metric)

    config_cols = [c for c in df.columns if "config_" in c and c != "config_wandb_run_id" and c != "config_online_simulation_size"]
    HPT_cols = ['config_seed', 'config_features', 'config_input_dim', 'config_REVIEW_DIM', 'config_architecture', 'config_FEATURES_PATH', 'config_ENV_LEARNING_RATE', 'config_offline_train_test_datasets']
    print(f"Hyperparameter columns (HPT_cols): {HPT_cols}")

    # Check if the columns to group by exist in df
    groupby_cols = [c for c in HPT_cols if c in df.columns]
    print(f"Grouping by columns: {groupby_cols}")

    if not groupby_cols:
        raise KeyError("None of the HPT columns are present in the DataFrame")

    # Extract metric columns for all epochs
    metric_columns = [col for col in df.columns if metric in col]
    print(f"Metric columns: {metric_columns}")

    # Select relevant columns
    selected_cols = groupby_cols + metric_columns
    df_selected = df[selected_cols]

    # Group by all hyperparameters including seed
    grouped_df = df_selected.groupby(groupby_cols).mean().reset_index()

    return grouped_df

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))
    lower_bound = np.percentile(bootstrapped_means, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_means, (1 + ci) / 2 * 100)
    return lower_bound, upper_bound

# Example usage
sweeps = ["e1rtm6h3"]  # Replace with your actual sweeps IDs
group_name = "Group"
result_df = result_metric(sweeps, group_name, metric="accuracy_per_mean_user_and_bot")

# Save the DataFrame to a CSV file
csv_path = 'all_epochs_params.csv'
result_df.to_csv(csv_path, index=False)

# Print the DataFrame
print("DataFrame with all configs and all epochs:")
print(result_df)

import pandas as pd

# Define the hyperparameter columns
HPT_cols = ['config_features', 'config_input_dim', 'config_REVIEW_DIM', 'config_architecture', 'config_FEATURES_PATH', 'config_ENV_LEARNING_RATE', 'config_offline_train_test_datasets']

# Read the CSV file
df = pd.read_csv('all_epochs_params.csv')

# Identify the metric columns that contain "accuracy_per_mean_user_and_bot"
metric_columns = [col for col in df.columns if "accuracy_per_mean_user_and_bot" in col]
print(f"Metric columns: {metric_columns}")

# Group by the specified hyperparameters (excluding seed) and calculate the mean for each group
grouped_df = df.groupby(HPT_cols + ['config_seed'])[metric_columns].mean().reset_index()

# Group again by the specified hyperparameters to average over seeds
final_grouped_df = grouped_df.groupby(HPT_cols)[metric_columns].mean().reset_index()

# Save the averaged DataFrame to a new CSV file
csv_path_avg = 'averaged_params.csv'
final_grouped_df.to_csv(csv_path_avg, index=False)

# Print the DataFrame
print("Averaged DataFrame:")
print(final_grouped_df)
