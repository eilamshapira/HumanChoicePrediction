import matplotlib.pyplot as plt


def update_metrics(metrics, epoch_metrics):
    for metric in epoch_metrics.keys():
        if metric not in metrics.keys():
            metrics[metric] = {
                phase: [] for phase in epoch_metrics[metric].keys()
            }
        for phase in epoch_metrics[metric].keys():
            if phase not in metrics[metric].keys():
                metrics[metric][phase] = []
            metrics[metric][phase].append(epoch_metrics[metric][phase])
    return metrics


def metrics_for_wandb(metrics, mode="nested"):
    wandb_metrics = {}
    if mode == "nested":
        for metric in metrics:
            wandb_metrics[metric] = {}
            for phase, phase_metric in metrics[metric].items():
                wandb_metrics[metric][phase] = {}
                wandb_metrics[metric][phase]["max"] = max(phase_metric)
                wandb_metrics[metric][phase]["min"] = min(phase_metric)
                wandb_metrics[metric][phase]["val"] = phase_metric[-1]
                for i, val in enumerate(phase_metric):
                    wandb_metrics[metric][phase][f"epoch{i+1}"] = val
    else:
        for metric in metrics:
            for phase, phase_metric in metrics[metric].items():
                wandb_metrics[f"{metric}_{phase}_max"] = max(phase_metric)
                wandb_metrics[f"{metric}_{phase}_min"] = min(phase_metric)
                for i, val in enumerate(phase_metric):
                    wandb_metrics[f"{metric}_{phase}_epoch{i + 1}"] = val
    return wandb_metrics


def plot_metrics_graph(metrics, config=""):
    # summarize history for accuracy
    for metric in metrics.keys():
        metric_data = metrics[metric]
        for phase in metric_data.keys():
            plt.plot(range(1, len(metric_data[phase])+1), metric_data[phase], label=phase)
        plt.title(f'model {metric} {str(config)}')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend()
        plt.show()