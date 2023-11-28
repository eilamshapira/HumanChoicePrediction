import pandas as pd
import os
import sys

sys.path.append("../")
from consts import *

class ResultsSaver:
    def __init__(self, basic_params, results_dir=RESULTS_DIR):
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        run_hash = basic_params["run_hash"]
        self.path = f"{results_dir}/{run_hash}.csv"
        self.basic_params = basic_params
        self.df = pd.DataFrame()

    def log(self, epoch, metrics):
        row = {}
        for key, value in self.basic_params.items():
            row[key] = [value]
        row["epoch"] = epoch
        for metric in metrics.keys():
            for phase, phase_metric in metrics[metric].items():
                row[f"{metric}_{phase}"] = phase_metric[-1]

        self.df = pd.concat([self.df, pd.DataFrame.from_dict(row)], ignore_index=True)
        self.df.to_csv(self.path)
