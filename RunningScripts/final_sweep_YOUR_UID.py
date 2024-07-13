import wandb
import subprocess
import sys
YOUR_WANDB_USERNAME = "bellatomer"
project = "NLP2024_PROJECT_BellaPerel" #todo: rename in wandb to this (nm it created it by itself)

command = [
        "${ENVIRONMENT_VARIABLE}",
        "${interpreter}",
        "StrategyTransfer.py",
        "${project}",
        "${args}"
    ]

sweep_config = {
    "name": "LSTM: SimFactor=0/4 for any features representation",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    }, #1*1*5*2*3
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["CNN"]},
        "seed": {"values": [0, 2, 4]},
        "online_simulation_factor": {"values": [0]},
        "features": {"values": ["EFs"]},
        "ENV_LEARNING_RATE": {"values": [1e-3]},
        "offline_train_test_datasets": {"values": ["original_data"]} #todo: can add more parameters in ST
    },
    "command": command
}

"""
sweep_config = {
    "name": "LSTM: SimFactor=0/4 for any features representation",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    }, #1*1*5*2*3
    "parameters": {
        "ENV_HPT_mode": {"values": [False]},
        "architecture": {"values": ["CNN", "LSTM"]},
        "seed": {"values": [0, 2, 4]},
        "online_simulation_factor": {"values": [0]},
        "features": {"values": ["EFs", "GPT4", "BERT"]},
        "ENV_LEARNING_RATE": {"values": [1e-3, 3e-3]},
        "offline_train_test_datasets": {"values": ["key_word_tagging", "transformer_tagged"]} #todo: can add more parameters in ST
    },
    "command": command
}
"""
# Initialize a new sweep
sweep_id = wandb.sweep(sweep=sweep_config, project=project)
print("run this line to run your agent in a screen:")
print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")


# Run the sweep agent and log output to a file
log_file = "sweep_agent_output.log"
with open(log_file, "w") as f:
    process = subprocess.Popen(
        ["wandb", "agent", f"{YOUR_WANDB_USERNAME}/{project}/{sweep_id}"],
        stdout=f,
        stderr=subprocess.STDOUT
    )
    process.communicate()


