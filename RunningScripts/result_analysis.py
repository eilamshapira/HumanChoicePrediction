import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
"""
# Initialize the W&B API
api = wandb.Api()

# Replace with your own sweep ID and project name
sweep_id = "LKLK"

# Fetch the sweep
sweep = api.sweep('bellatomer/Strategy_Transfer_TACL/jgfzuppf')

# Define patterns for the different metrics
patterns = {
    'TotalLoss': re.compile(r"ENV_Train_TotalLoss(?:_epoch(\d+))?"),
    'Probability to choose the right action': re.compile(r"ENV_Train_Probability to choose the right action(?:_epoch(\d+))?"),
    'Weighted probability to choose the right action': re.compile(r"ENV_Train_Weighted probability to choose the right action:_epoch(\d+)")
}

# Collect data for each pattern across all runs in the sweep
data = []

for run in sweep.runs:
    metrics = run.summary
    for key, value in metrics.items():
        for metric_type, pattern in patterns.items():
            match = pattern.match(key)
            if match:
                epoch = int(match.group(1)) if match.group(1) is not None else 0
                data.append((epoch, metric_type, value))

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=["epoch", "type", "value"])

# Plot the loss data separately
loss_df = df[df['type'] == 'TotalLoss']

plt.figure(figsize=(10, 5))
sns.lineplot(data=loss_df, x="epoch", y="value", marker='o', color='b')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

# Print the min value
"""
import re
import pandas as pd
import matplotlib.pyplot as plt

# Read the log file with the correct encoding
log_file_path = '../wandb/run-20240705_173544-yceko246/files/output.log'
with open(log_file_path, 'r', encoding='utf-8') as file:
    log_data = file.read()

# Define regex patterns
epoch_pattern = re.compile(r'# Epoch (\d+)')
accuracy_pattern = re.compile(r'accuracy_per_mean_user_and_bot:\s+(\d+\.\d+)')
proba_accuracy_pattern = re.compile(r'proba_accuracy_per_mean_user_and_bot:\s+(\d+\.\d+)')

# Initialize lists to store extracted data
epochs = []
accuracies = []
proba_accuracies = []

# Extract data from log file
for match in epoch_pattern.finditer(log_data):
    epoch = int(match.group(1))
    epochs.append(epoch)

    subsequent_data = log_data[match.end():]

    # Find the first occurrence of proba accuracy pattern
    proba_accuracy_match = proba_accuracy_pattern.search(subsequent_data)
    if proba_accuracy_match:
        proba_accuracy = float(proba_accuracy_match.group(1))
        proba_accuracies.append(proba_accuracy)
    else:
        proba_accuracies.append(None)

    # Find the second occurrence of accuracy pattern
    accuracy_matches = list(accuracy_pattern.finditer(subsequent_data))
    if len(accuracy_matches) >= 2:
        accuracy = float(accuracy_matches[1].group(1))
        accuracies.append(accuracy)
    else:
        accuracies.append(None)

# Create DataFrame
data = {
    'Epoch': epochs,
    'Accuracy': accuracies,
    'Proba Accuracy': proba_accuracies
}
df = pd.DataFrame(data)

# Display DataFrame
print(df)

# Plot accuracies as a function of the epoch
plt.figure(figsize=(10, 5))
plt.plot(df['Epoch'], df['Accuracy'], marker='o', label='Accuracy')
plt.plot(df['Epoch'], df['Proba Accuracy'], marker='o', label='Proba Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy and Proba Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
