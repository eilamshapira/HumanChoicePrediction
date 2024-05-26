import wandb
YOUR_WANDB_USERNAME = "beneliav1"
project = "Strategy_Transfer_TACL"

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "Hyperparameter tuning for transformer model",
    "method": "random",
    "metric": {
        "goal": "maximize",
        "name": "ENV_test_accuracy_per_mean_user_and_bot"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["LSTM"]},
        "combine_features": {"values": [True]},
        "seed": {"values": [1, 2, 3]},
        "feature_combination": {"values": [["EFs", "GPT4"], ["EFs", "BERT"]]},
        "hidden_dim": {"values": [32, 64, 128]},
        "ENV_LEARNING_RATE": {"values": [1e-3, 1e-4, 1e-5]},
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
