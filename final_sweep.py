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
    "name": "LSTM: SimFactor=0/4 for any features representation",
    "method": "random",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["transformer"]},
        "seed": {"values": list(range(1, 6))},
        "save_previous_games": {"values": [True]},
        "combine_features": {"values": [True]},
        "seed": {"values": [1, 2, 3]},
        "feature_combination": {"values": ["EFs_GPT4", "EFs_BERT"]},
        "transformer_nheads": {"values": [2, 4, 8]},
        "hidden_dim": {"values": [32, 64, 128]},
        "layers": {"values": [1, 2]},
        "ENV_LEARNING_RATE": {"values": [1e-3, 1e-4, 1e-5]},
    },
    "command": command
}

# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
