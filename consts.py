import torch
import pandas as pd
import numpy as np
import json

class Configurations:
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config

    def get_config(self):
        return self.config  # Access config here


config_instance = Configurations()
config = config_instance.get_config()

DATA_GAME_REVIEWS_PATH = "data/game_reviews"
if config.get("offline_train_test_datasets") == "key_word_tagging":
    DATA_CLEAN_ACTION_PATH_X = "data/games_clean_X.csv"
    DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS = 210
    DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS = 35
    DATA_CLEAN_ACTION_PATH_Y = "data/games_clean_Y.csv"
elif config.get("offline_train_test_datasets") == "original_data":
    DATA_CLEAN_ACTION_PATH_X = "data/games_clean_X_original.csv"
    DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS = 210
    DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS = 35
    DATA_CLEAN_ACTION_PATH_Y = "data/games_clean_Y_original.csv"
else:
    DATA_CLEAN_ACTION_PATH_X = "data/games_clean_X_trans_tagging.csv"
    DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS = 210
    DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS = 35
    DATA_CLEAN_ACTION_PATH_Y = "data/games_clean_Y_trans_tagging.csv"

OFFLINE_SIM_DATA_PATH = "data/LLM_games_personas.csv"

REVIEW_ENCODERS_PATH = "models/reviews_encoders"
REVIEW_VECTORS_PATH = "data/reviews_vectors"

USING_REACTION_TIME = True
reaction_time_bins = [(0, 400),
                      (400, 800),
                      (800, 1200),
                      (1200, 1600),
                      (1600, 2500),
                      (2500, 4000),
                      (4000, 6000),
                      (6000, 12000),
                      (12000, 20000),
                      (20000, np.inf)]
reaction_time_columns_names = [f"last_reaction_time_{lower}_{upper}" for lower, upper in reaction_time_bins]

if config.get("offline_train_test_datasets") == "original_data":
    STRATEGIC_FEATURES_ORDER = ['roundNum', 'user_points', 'bot_points',
                                'last_didGo_True', 'last_didGo_False',
                                'last_didWin_True', 'last_didWin_False',
                                'last_last_didGo_True', 'last_last_didGo_False',
                                'last_last_didWin_True', 'last_last_didWin_False',
                                "user_earned_more", "user_not_earned_more"]
else:
    STRATEGIC_FEATURES_ORDER = ['roundNum', 'user_points', 'bot_points',
                                'last_didGo_True', 'last_didGo_False',
                                'last_didWin_True', 'last_didWin_False',
                                'last_last_didGo_True', 'last_last_didGo_False',
                                'last_last_didWin_True', 'last_last_didWin_False',
                                "user_earned_more", "user_not_earned_more", 'Staff (Pos)', 'Facilities (Pos)', 'Cleanliness (Pos)', 'Location (Pos)', 'Food (Pos)', 'Staff (Neg)', 'Facilities (Neg)', 'Cleanliness (Neg)', 'Location (Neg)', 'Food (Neg)']


if USING_REACTION_TIME:
    STRATEGIC_FEATURES_ORDER += reaction_time_columns_names

N_HOTELS = 1068

STRATEGY_DIM = len(STRATEGIC_FEATURES_ORDER)

DEEPRL_LEARNING_RATE = 4e-4

DATA_ROUNDS_PER_GAME = 10
SIMULATION_BATCH_SIZE = 4
ENV_BATCH_SIZE = 4

SIMULATION_MAX_ACTIVE_USERS = 2000
SIMULATION_TH = 9

DATA_N_BOTS = 1179

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

DATA_BLANK_ROW_DF = lambda s: pd.DataFrame.from_dict({"user_id": [-100],
                                                      "strategy_id": [s],
                                                      "gameId": [-100],
                                                      "roundNum": [-1],
                                                      "hotelId": [-1],
                                                      "reviewId": [-1],
                                                      "hotelScore": [-1],
                                                      "didGo": [-100],
                                                      "didWin": [-100],
                                                      "correctAnswers": [-1],
                                                      "reaction_time": [1],
                                                      "review_positive": [""],
                                                      "review_negative": [""],
                                                      "last_didGo_True": [0],
                                                      "last_didWin_True": [0],
                                                      "last_didGo_False": [0],
                                                      "last_didWin_False": [0],
                                                      "last_last_didGo_True": [0],
                                                      "last_last_didWin_True": [0],
                                                      "last_last_didGo_False": [0],
                                                      "last_last_didWin_False": [0],
                                                      "last_reaction_time": [-1],
                                                      "user_points": [-1],
                                                      "bot_points": [-1],
                                                      "is_sample": [False],
                                                      "weight": 0,
                                                      "action_id": [-1],
                                                      'Staff (Pos)': [0], 'Facilities (Pos)': [0], 'Cleanliness (Pos)': [0], 'Location (Pos)': [0], 'Food (Pos)': [0],
                                                      'Staff (Neg)': [0], 'Facilities (Neg)': [0], 'Cleanliness (Neg)': [0], 'Location (Neg)': [0], 'Food (Neg)': [0]})

bot2strategy_X = {0: 3, 1: 0, 2: 2, 3: 5, 4: 59, 5: 19}
bot2strategy_Y = {0: 132, 1: 23, 2: 107, 3: 43, 4: 17, 5: 93}

bot_thresholds_X = {0: 10, 1: 7, 2: 9, 3: 8, 4: 8, 5: 9}
bot_thresholds_Y = {0: 10, 1: 9, 2: 9, 3: 9, 4: 9, 5: 9}

AGENT_LEARNING_TH = 8
"""
#concat all of the files in game_reviews folder to one csv file, and join it with the EFs_by_GPT35.csv on "hotel number" column
import pandas as pd
import os

# Define the directory containing game review files and the file to join
game_reviews_dir = 'data/game_reviews'
ef_file_path = 'data/EFs_by_GPT35.csv'

# Read and concatenate all game review files
game_reviews = pd.concat([pd.read_csv(os.path.join(game_reviews_dir, f)) for f in os.listdir(game_reviews_dir) if f.endswith('.csv')])

# Read the EF file
ef_data = pd.read_csv(ef_file_path)

# Join the concatenated game reviews with the EF data on "hotel number" column
merged_data = game_reviews.merge(ef_data, on='hotel number')

# Save the merged data to a new CSV file
output_file_path = 'game_reviews_EFs_extra_features.csv'
merged_data.to_csv(output_file_path, index=False)

print(f"Merged data saved to {output_file_path}")
"""
