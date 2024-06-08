import wandb

YOUR_WANDB_USERNAME = "eden-hindi"
#YOUR_WANDB_USERNAME = "guytechnion-org"

project = "NLP2024_PROJECT"

command = [
    "${ENVIRONMENT_VARIABLE}",
    "${interpreter}",
    "StrategyTransfer.py",
    "${project}",
    "${args}"
]
sweep_config_1 = {
    "name": "LSTM: eps incorrect oracle for different agent learning rate",
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "AUC.test.max"
    },
    "parameters": {
        "seed": {"values": list(range(1, 6))},
        "eps_incorrect": {"values": [0, 0.1, 0.2]}, # At the first sweep we ran with this line and at the second without this
        "learning_rate_gb": {"values": [0.5, 0.1, 0.02]}
    },
    "command": command
}
# sweep_config_2 = {
#     "name": "LSTM: eps incorrect oracle for different agent learning rate",
#     "method": "grid",
#     "metric": {
#         "goal": "maximize",
#         "name": "AUC.test.max"
#     },
#     "parameters": {
#         "seed": {"values": list(range(1, 6))},
#         "learning_rate_gb": {"values": [0.5, 0.1, 0.02]}
#     },
#     "command": command
# }

# Initialize a new sweep
# sweep_id = wandb.sweep(sweep=sweep_config, project=project)
# print("run this line to run your agent in a screen:")
# print(f"screen -dmS \"sweep_agent\" wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
sweep_id = wandb.sweep(sweep=sweep_config_1, project=project)

print("Run these lines to run your agent in a screen:")
parallel_num = 6

if parallel_num > 10:
    print('Are you sure you want to run more than 10 agents in parallel? It would result in a CPU bottleneck.')
for i in range(parallel_num):
    print(f"screen -dmS \"final_sweep_agent_{i}\" nohup wandb agent {YOUR_WANDB_USERNAME}/{project}/{sweep_id}")
