import torch
import warnings
import pandas as pd
import os
import numpy as np

# Data Pre-Processing
DATA_FLURRY_DIR_PATH = "data/flurry_data"
DROP_INCOMPLETE_GAMES = False
DROP_NOT_WINNERS = True

Y_VERSIONS = ["2.0.1", "67", "2.0.2", "71", "72", "2.0.3", "73"]
OUT_VERSIONS = []

DATA_CLEAN_ACTION_PATH_X = "data/games_clean_X.csv"
DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS = 211
DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS = 34

DATA_CLEAN_ACTION_PATH_Y = "data/games_clean_Y.csv"

ONE_HOT_BOT = False
BOT_TRUST = True
BOT_EMBEDDING = False
assert int(ONE_HOT_BOT) + int(BOT_TRUST) + int(BOT_EMBEDDING) < 2

if ONE_HOT_BOT:
    DATA_BOTS_VECTORS_PATH = "StrategySpace/one_hot_bot_vectors.csv"
    BOT_DIM = 48
else:
    if BOT_TRUST:
        DATA_BOTS_VECTORS_PATH = "StrategySpace/bot_trust_vector.csv"
    else:
        DATA_BOTS_VECTORS_PATH = "StrategySpace/bot_vectors.csv"
    BOT_DIM = 16

DATA_N_BOTS = 1179

DATA_GAME_REVIEWS_PATH = "data/game_reviews"
DATA_N_HOTELS = 1068
DATA_ROUNDS_PER_GAME = 10
DATA_HEADER = ["user_id", "strategy_id", "gameId", "roundNum", "hotelId", "reviewId", "hotelScore", "didGo", "didWin",
               "correctAnswers", "last_reaction_time", "last_didGo_True", "last_didWin_True",
               "last_didGo_False", "last_didWin_False",
               "user_points", "bot_points"]

DATA_BLANK_ROW_DF = lambda s: pd.DataFrame.from_dict({"user_id": [-1],
                                            "strategy_id": [s],
                                            "gameId": [-1],
                                            "roundNum": [-1],
                                            "hotelId": [-1],
                                            "reviewId": [-1],
                                            "hotelScore": [-1],
                                            "didGo": [False],
                                            "didWin": [False],
                                            "correctAnswers": [-1],
                                            "reaction_time": [1],
                                            "review_positive": [""],
                                            "review_negative": [""],
                                            "weight": [0],
                                            "last_didGo_True": [0],
                                            "last_didWin_True": [0],
                                            "last_didGo_False": [0],
                                            "last_didWin_False": [0],
                                            "user_points": [-1],
                                            "bot_points": [-1]})

bot2strategy_X = {0: 3, 1: 0, 2: 2, 3: 5, 4: 59, 5: 19}
bot2strategy_Y = {0: 132, 1: 23, 2: 107, 3: 43, 4: 17, 5: 93}

bot_thresholds_X = {0: 10, 1: 7, 2: 9, 3: 8, 4: 8, 5: 9}
bot_thresholds_Y = {0: 10, 1: 9, 2: 9, 3: 9, 4: 9, 5: 9}

SAVE_CLEAN_DATA = True

# Fine Tune BERT
MODELS_PATH = "models"
if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)
FINE_TUNED_MODELS_PATH = f"{MODELS_PATH}/fine_tuned"
FINE_TUNED_DATASET_PATH = "data/reviews_only.csv"

FINE_TUNING_LR = 1e-4
FINE_TUNING_MAX_EPOCHS = 0

LEARN_REVIEWS_VECTORS_IN_ADVANCE = True
BERT_MAX_LENGTH = 150

FINE_TUNING_DEBUG_FLAG = False
if FINE_TUNING_DEBUG_FLAG:
    warnings.warn("fine tuning debug flag is true!")

# Architecture
N_CLASSES = 2

BERT_DIM = 768
REVIEW_FEATURES = True
REVIEW_FEATURES_PATH = "data/37HCF"
assert LEARN_REVIEWS_VECTORS_IN_ADVANCE - REVIEW_FEATURES == 0

if REVIEW_FEATURES:
    REVIEW_DIM = 37
else:
    # use bert pretrained vectors
    REVIEW_DIM = 48

TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
SIMULATION_BATCH_SIZE = 4
SIMULATION_MAX_ACTIVE_USERS = 2000
SIMULATION_SIGNAL_ERROR = 0.3
SIMULATION_TH = 9
ADVANCED_REACTION_TIME_IN_SIMULATION = True
SIMULATION_USER_IMPROVE = 0.01

# Running
MAX_EPOCHS = 20
PROBLEM_1_P_USERS_TEST = 0.2
CONFUSION_MATRIX_REACTION_TIMES = [0, 1000, 2000]
# SCHEDULER_LR_GAMMA = 0.95

USER_DIM_SimpleLSTM = 16
USER_DIM_TRANSFORMER = 32
BINARY_BEHAVIORAL_FEATURES = []
DO_SIM_ANYWAY = True

reaction_time_bins = [(0, 500),
                      (500, 1000),
                      (1000, 2000),
                      (2000, 3000),
                      (3000, 4000),
                      (4000, 6500),
                      (6500, 12000),
                      (12000, 20000),
                      (20000, np.inf)]
reaction_time_columns_names = [f"last_reaction_time_{lower}_{upper}" for lower, upper in reaction_time_bins]

NUMERICAL_BEHAVIORAL_FEATURES = ["last_didGo_True", "last_didWin_True", "last_didGo_False", "last_didWin_False",
                                 "user_points", "bot_points",
                                 "user_earned_more", "user_not_earned_more"] + reaction_time_columns_names

CHECKPOINT_DIR = "checkpoints"
SAVE_TAGS = True
PREDS_PATH = "results_log"
RESULTS_DIR = "results_10_02"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
