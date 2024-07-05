import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Initialize the W&B API
api = wandb.Api()

# Fetch the specific run
run = api.run("bellatomer/Strategy_Transfer_TACL/jgfzuppf")

# Get the summary metrics
metrics = run.summary

# Filter and organize metrics that contain "ENV_Test_accuracy_per_mean_user_and_bot_epoch" and "ENV_Test_proba_accuracy_per_mean_user_and_bot_epoch"
pattern_accuracy = re.compile(r"ENV_Test_accuracy_per_mean_user_and_bot_epoch(\d+)")
pattern_proba = re.compile(r"ENV_Test_proba_accuracy_per_mean_user_and_bot_epoch(\d+)")
data = []

for key, value in metrics.items():
    match_accuracy = pattern_accuracy.match(key)
    match_proba = pattern_proba.match(key)
    if match_accuracy:
        epoch = int(match_accuracy.group(1))
        data.append((epoch, 'Accuracy', value))
    elif match_proba:
        epoch = int(match_proba.group(1))
        data.append((epoch, 'Proba Accuracy', value))

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=["epoch", "type", "value"])

# Plot the data using seaborn
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x="epoch", y="value", hue="type", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Test Metrics Over Epochs")
plt.grid(True)
plt.legend(title="Metric Type")
plt.show()

# Find and print the max value for each type
for metric_type in df['type'].unique():
    filtered_df = df[df['type'] == metric_type]
    max_row = filtered_df.loc[filtered_df['value'].idxmax()]
    print(f"Max value for {metric_type}: {max_row['value']:.03} at epoch {max_row['epoch']+1}")