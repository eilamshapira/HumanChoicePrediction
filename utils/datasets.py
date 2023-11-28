import torch
from torch.utils.data import Dataset
import numpy as np
from consts import *
from transformers import BertTokenizer
import os
from collections import defaultdict
from utils.functions import learn_sigmoid_weighting_by_reaction_time, get_model_name, move_to
import Simulation.strategies_code as bot_strategies
import Simulation.basice_users_code as user_strategies
import random
import utils.basic_nature_options
from sklearn.linear_model import LogisticRegression
import pickle
from tqdm import trange


class RealDataSet(Dataset):
    def __init__(self, user_groups, bots_path=DATA_BOTS_VECTORS_PATH, reviews_path=DATA_GAME_REVIEWS_PATH,
                 strategies=None, users=None, weighting_function=None, bert_params=None,
                 x_path=DATA_CLEAN_ACTION_PATH_X, y_path=DATA_CLEAN_ACTION_PATH_Y):
        self.actions_df = None
        if "X" in user_groups:
            self.actions_df = pd.read_csv(x_path)
        if "Y" in user_groups:
            Y_dataset = pd.read_csv(y_path)
            Y_dataset.user_id += DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS
            if self.actions_df is None:
                self.actions_df = Y_dataset
            else:
                self.actions_df = pd.concat([self.actions_df, Y_dataset])

        if strategies is not None:
            self.actions_df = self.actions_df[self.actions_df["strategy_id"].isin(strategies)]
            strategies_in_data = self.actions_df["strategy_id"].drop_duplicates().tolist()
            for strategy in strategies:
                assert strategy in strategies_in_data, f"You have no games against strategy #{strategy} " \
                                                       f"in the entire dataset!"

        if users is not None:
            self.actions_df = self.actions_df[self.actions_df["user_id"].isin(users.tolist())]

        self.weighting_function = self.set_weighting(weighting_function)
        action_weighting = self.actions_df.groupby("didGo").sum()["weight"]
        action_weighting = action_weighting[False] / action_weighting
        action_weighting_function = lambda weight, did_go: weight * action_weighting[did_go]

        self.actions_df["weight"] = self.actions_df.apply(
            lambda row: action_weighting_function(row["weight"], row["didGo"]), axis=1)

        self.actions_df = self.actions_df.groupby(["user_id", "gameId"])
        self.bot_vectors = np.genfromtxt(bots_path, delimiter=",")
        self.idx_to_group = list(self.actions_df.indices.keys())
        self.group_to_idx = {g: i for i, g in enumerate(self.idx_to_group)}
        self.n_groups_by_user_id = defaultdict(list)
        for u, i in sorted(self.actions_df.indices.keys()):
            self.n_groups_by_user_id[u].append(i)

        if LEARN_REVIEWS_VECTORS_IN_ADVANCE:
            if REVIEW_FEATURES:
                self.review_reduced = pd.read_csv(REVIEW_FEATURES_PATH, index_col=0).astype(int).to_dict(orient='list')
                self.review_reduced = {int(rid): torch.Tensor(vec) for rid, vec in self.review_reduced.items()}
                self.review_reduced[-1] = torch.zeros(REVIEW_DIM)
            else:
                if REVIEW_ONE_HOT:
                    reviews_path = REVIEW_ONE_HOT_PATH
                else:
                    model_name = get_model_name(bert_params, "BertFineTuned")
                    model_path = FINE_TUNED_MODELS_PATH + "/" + model_name
                    reviews_path = model_path + "_reviews"
                self.review_reduced = torch.load(reviews_path)
                for r in self.review_reduced:
                    self.review_reduced[r] = self.review_reduced[r].to(device)
                self.review_reduced[-1] = torch.zeros(REVIEW_DIM).to(device)
        else:
            self.reviews = {}
            for h in range(1, DATA_N_HOTELS + 1):
                hotel_df = pd.read_csv(os.path.join(reviews_path, f"{h}.csv"),
                                       header=None)
                for review in hotel_df.iterrows():
                    self.reviews[review[1][0]] = {"positive": review[1][2],
                                                  "negative": review[1][3],
                                                  "score": review[1][4]}
                self.reviews[-1] = {"positive": "",
                                    "negative": "",
                                    "score": 8}
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True,
                                                                max_length=BERT_MAX_LENGTH)

    def set_weighting(self, weighting_function=None):
        if weighting_function is None:
            weighting_function = learn_sigmoid_weighting_by_reaction_time(self.actions_df)
        self.actions_df["weight"] = self.actions_df["reaction_time"].apply(weighting_function)
        return weighting_function

    def __len__(self):
        return len(self.idx_to_group)

    def __getitem__(self, item):
        if isinstance(item, int):
            group = self.idx_to_group[item]
        else:
            group = item
        game = self.actions_df.get_group(group).reset_index()
        user_id = game["user_id"][0]
        n_rounds = len(game)

        if n_rounds < DATA_ROUNDS_PER_GAME:
            game = pd.concat([game] + [DATA_BLANK_ROW_DF(game["strategy_id"][0])] * (DATA_ROUNDS_PER_GAME - n_rounds), ignore_index=True)

        bot_strategy = game["strategy_id"].to_numpy()
        bot_vector = self.bot_vectors[bot_strategy[0]]

        hotels_scores = game["hotelScore"].to_numpy()

        action_taken = game["didGo"].to_numpy()
        is_hotel_good = (game["didGo"] == game["didWin"]).to_numpy()
        weight = game["weight"].to_numpy()

        reaction_time = game["reaction_time"].to_numpy()
        last_reaction_time = game["last_reaction_time"].to_numpy()

        user_points = game["user_points"].to_numpy()
        bot_points = game["bot_points"].to_numpy()
        user_earned_more = user_points >= bot_points
        user_not_earned_more = user_points < bot_points

        reviewId = game["reviewId"]
        game = {"user_id": user_id,
                "bot_strategy": bot_strategy,
                "bot_vector": bot_vector,
                "n_rounds": n_rounds,
                "hotels_scores": hotels_scores,
                "action_taken": action_taken,
                "is_hotel_good": is_hotel_good,
                "weight": weight,
                "reaction_time": reaction_time,
                "last_reaction_time": last_reaction_time,
                "last_didGo_True": game["last_didGo_True"].to_numpy(),
                "last_didWin_True": game["last_didWin_True"].to_numpy(),
                "last_didGo_False": game["last_didGo_False"].to_numpy(),
                "last_didWin_False": game["last_didWin_False"].to_numpy(),
                "user_points": user_points /10, "bot_points": bot_points /10,
                "user_earned_more": user_earned_more, "user_not_earned_more": user_not_earned_more}
        for column_name, (lower, upper) in zip(reaction_time_columns_names, reaction_time_bins):
            game[column_name] = (lower <= last_reaction_time) & (last_reaction_time < upper)
        if SAVE_TAGS:
            game["group"] = tuple(group)

        if LEARN_REVIEWS_VECTORS_IN_ADVANCE:
            game["review_vector"] = reviewId.apply(lambda r: self.review_reduced[r]).tolist()
            game["review_vector"] = torch.stack(game["review_vector"])
        else:
            review_positive = reviewId.apply(lambda r: self.reviews[r]["positive"]).fillna("").tolist()
            review_negative = reviewId.apply(lambda r: self.reviews[r]["negative"]).fillna("").tolist()
            game["review_encoded_positive"] = self.bert_tokenizer(review_positive, add_special_tokens=True,
                                                                  return_tensors='pt', padding="max_length",
                                                                  max_length=BERT_MAX_LENGTH, truncation=True)
            game["review_encoded_negative"] = self.bert_tokenizer(review_negative, add_special_tokens=True,
                                                                  return_tensors='pt', padding="max_length",
                                                                  max_length=BERT_MAX_LENGTH, truncation=True)
        return game


class FineTuningDataSet(Dataset):
    def __init__(self, data_type="FineTuning"):
        if data_type == "FineTuning":
            self.reviews = pd.read_csv(FINE_TUNED_DATASET_PATH)
        elif data_type == "GameReviews":
            self.reviews = {}
            for h in range(1, DATA_N_HOTELS + 1):
                hotel_df = pd.read_csv(os.path.join(DATA_GAME_REVIEWS_PATH, f"{h}.csv"),
                                       header=None)
                for review in hotel_df.iterrows():
                    self.reviews[review[1][0]] = {"review_id": review[1][0],
                                                  "Positive_Review": review[1][2],
                                                  "Negative_Review": review[1][3],
                                                  "Reviewer_Score": review[1][4]}
            self.reviews = pd.DataFrame.from_dict(self.reviews, orient='index').fillna("")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                            truncation=True, max_length=BERT_MAX_LENGTH)

    def __len__(self):
        return 1068 * 7 if FINE_TUNING_DEBUG_FLAG else len(self.reviews)

    def __getitem__(self, item):
        review_row = self.reviews.iloc[item]
        review = {}
        review["review_positive"] = review_row["Positive_Review"]
        review["review_negative"] = review_row["Negative_Review"]
        review["score"] = torch.tensor([review_row["Reviewer_Score"]], dtype=torch.float)

        review["review_positive"] = self.bert_tokenizer(review["review_positive"], add_special_tokens=True,
                                                        return_tensors='pt', padding="max_length",
                                                        max_length=BERT_MAX_LENGTH,
                                                        truncation=True)
        review["review_negative"] = self.bert_tokenizer(review["review_negative"], add_special_tokens=True,
                                                        return_tensors='pt', padding="max_length",
                                                        max_length=BERT_MAX_LENGTH,
                                                        truncation=True)
        if "review_id" in review_row:
            review["review_id"] = review_row["review_id"]
        return review


class SimulationDataSet(Dataset):
    def __init__(self,
                 n_users,
                 simulation_th=SIMULATION_TH,
                 signal_error=SIMULATION_SIGNAL_ERROR,
                 max_active=SIMULATION_MAX_ACTIVE_USERS,
                 bert_params=None,
                 bots_per_user=6,
                 basic_nature=24,
                 strategies_for_reaction_time=None,
                 advanced_reaction_time=ADVANCED_REACTION_TIME_IN_SIMULATION,
                 user_improve=SIMULATION_USER_IMPROVE,
                 zero_knowledge=True,
                 problem=-100):

        self.bots_per_user = bots_per_user
        self.n_users = int(n_users / self.bots_per_user * 6)
        max_active = int(max_active / self.bots_per_user * 6)

        self.SIMULATION_TH = simulation_th
        self.SIMULATION_SIGNAL_EPSILON = signal_error
        self.users = defaultdict(list)
        self.n_games_per_user = {}
        self.active_users = []
        self.next_user = 0
        self.total_games_created = 0
        self.user_improve = user_improve
        self.zero_knowledge = zero_knowledge
        self.problem = problem

        self.hotels = [np.array([0] * 7)]
        self.reviews_id = [np.array([0] * 7)]
        for hotel in range(1, DATA_N_HOTELS + 1):
            hotel_path = f"{DATA_GAME_REVIEWS_PATH}/{hotel}.csv"
            hotel_csv = pd.read_csv(hotel_path, header=None)
            self.hotels.append(hotel_csv.iloc[:, 4].to_numpy())
            self.reviews_id.append(hotel_csv.iloc[:, 0].to_numpy())
        self.hotels = np.array(self.hotels)
        self.reviews_id = np.array(self.reviews_id)
        if LEARN_REVIEWS_VECTORS_IN_ADVANCE:
            if REVIEW_FEATURES:
                self.review_reduced = pd.read_csv(REVIEW_FEATURES_PATH, index_col=0).astype(int).to_dict(orient='list')
                self.review_reduced = {int(rid): torch.Tensor(vec) for rid, vec in self.review_reduced.items()}
                self.review_reduced[-1] = torch.zeros(REVIEW_DIM)
            else:
                model_name = get_model_name(bert_params, "BertFineTuned")
                model_path = FINE_TUNED_MODELS_PATH + "/" + model_name
                reviews_path = model_path + "_reviews"
                self.review_reduced = torch.load(reviews_path)
                for r in self.review_reduced:
                    self.review_reduced[r] = self.review_reduced[r].to(device)
                self.review_reduced[-1] = torch.zeros(REVIEW_DIM).to(device)
        else:
            self.reviews = {}
            for h in range(1, DATA_N_HOTELS + 1):
                hotel_df = pd.read_csv(os.path.join(DATA_GAME_REVIEWS_PATH, f"{h}.csv"),
                                       header=None)
                for review in hotel_df.iterrows():
                    self.reviews[review[1][0]] = {"positive": review[1][2],
                                                  "negative": review[1][3],
                                                  "score": review[1][4]}
                self.reviews[-1] = {"positive": "",
                                    "negative": "",
                                    "score": 8}
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True,
                                                                max_length=BERT_MAX_LENGTH)

        self.bot_vectors = np.genfromtxt(DATA_BOTS_VECTORS_PATH, delimiter=",")
        self.gcf = pd.read_csv("data/37HCF", index_col=0, dtype=int)
        self.gcf.columns = self.gcf.columns.astype(int)

        self.basic_nature = utils.basic_nature_options.pers[basic_nature]
        # self.ACTIONS = {0: ("correct", 0, user_strategies.correct_action),
        #                 1: ("random", 1 + basic_nature[0], user_strategies.random_action, (None)),
        #                 2: ("history_and_review_quality", 1 + basic_nature[1], user_strategies.history_and_review_quality, ([0, 5], [7, 9])),
        #                 3: ("topics", 1 + basic_nature[4], user_strategies.topics, ([2, 3, 5]))}

        # print(self.ACTIONS)
        #
        # self.advanced_reaction_time = advanced_reaction_time
        # self.reaction_time_bins_simulation = [0, 500, 1000, 1500, 3000, 15_000, np.inf]
        # self.basic_x_cols_simulation = ['roundNum', 'correctAnswers', 'user_points', 'bot_points', 'last_didGo_True', 'last_didGo_False', 'last_didWin_True', 'last_didWin_False']
        # self.strategies_for_reaction_time = strategies_for_reaction_time
        # self.reaction_time_model = self.load_reaction_time_model()

        self.max_active = min(max_active, self.n_users)
        pbar = trange(self.max_active)
        for i in pbar:
            self.new_user()
            pbar.set_description(f"mean games/user: {self.total_games_created / self.next_user}")

    class SimulatedUser:
        def __init__(self, user_improve, basic_nature):
            history_window = np.random.negative_binomial(2, 1/2) + np.random.randint(0, 2)
            quality_threshold = np.random.normal(8, 0.5)
            positive_topics = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 19, 28, 42], 3)
            negative_topics = np.random.choice([11, 20, 21, 22, 23, 24, 25, 26, 27, 36, 40], 3)
            self.ACTIONS = {0: ("correct", 0, user_strategies.correct_action),
                            1: ("random", basic_nature[0], user_strategies.random_action),
                            2: ("history_and_review_quality", basic_nature[1],
                                user_strategies.history_and_review_quality(history_window, quality_threshold)),
                            3: ("topic based", basic_nature[2], user_strategies.topic_based(positive_topics,
                                                                                                negative_topics,
                                                                                                quality_threshold))}
            self.nature = np.random.rand(len(self.ACTIONS)) * np.array([v[1] for v in self.ACTIONS.values()])
            self.nature = self.nature / sum(self.nature)
            self.user_proba = self.nature.copy()
            self.user_improve = user_improve

        def return_to_init_proba(self):
            self.user_proba = self.nature.copy()

        def update_proba(self):
            reduce_feelings = np.random.rand(len(self.ACTIONS) - 1) * self.user_improve - (self.user_improve/10)
            total_reduced = self.user_proba[1:] * reduce_feelings
            self.user_proba[1:] -= total_reduced
            self.user_proba[1:] = np.maximum(0, self.user_proba[1:])
            self.user_proba[0] = 1 - self.user_proba[1:].sum()

    def play_round(self, bot_message, user, previous_rounds, hotel, review_id):
        user_strategy = self.sample_from_probability_vector(user.user_proba)
        user_strategy_function = user.ACTIONS[user_strategy][2]
        review_features = self.gcf[review_id]
        information = {"bot_message": bot_message,
                       "previous_rounds": previous_rounds,
                       "hotel_value": hotel.mean(),
                       "review_features": review_features}
        user_action = user_strategy_function(information)
        return user_action

    def load_reaction_time_model(self):
        model_hash = "all" if self.strategies_for_reaction_time is None else "_".join([str(s) for s in (sorted(self.strategies_for_reaction_time))])
        model_path = MODELS_PATH + "/reaction_time_" + model_hash + ".pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                log_reg_model = pickle.load(f)
        else:
            log_reg_model = self.reaction_time_init()
            with open(model_path, "wb") as f:
                pickle.dump(log_reg_model, f)
        return log_reg_model

    def reaction_time_init(self):
        train_data = pd.read_csv("data/games_clean_X.csv")
        if self.strategies_for_reaction_time:
            train_data = train_data[train_data.strategy_id.isin(self.strategies_for_reaction_time)]
        gcf_df = pd.read_csv("data/37HCF", index_col=0, dtype=int)
        gcf_df.columns = gcf_df.columns.astype(int)
        G = gcf_df.T
        train_data[[f"gcf_{c}" for c in G.columns.tolist()]] = G.loc[train_data["reviewId"]].to_numpy()
        bins = self.reaction_time_bins_simulation
        last_reaction_time_cat = []
        reaction_time_cat = []
        for lb, ub in zip(bins[:-1], bins[1:]):
            for main_cat in ["reaction_time", "last_reaction_time"]:
                cat = f"{main_cat}{lb}_{ub}"
                train_data[cat] = (lb <= train_data[main_cat]) & (train_data[main_cat] < ub)
                if main_cat == "reaction_time":
                    reaction_time_cat += [cat]
                else:
                    last_reaction_time_cat += [cat]
        x_cols = self.basic_x_cols_simulation + last_reaction_time_cat
        y_cols = reaction_time_cat
        x_train = train_data[x_cols].to_numpy()
        y_train = train_data[y_cols].to_numpy().argmax(1)

        action_prediction_model = LogisticRegression(max_iter=10_000, class_weight="balanced")
        action_prediction_model.fit(x_train, y_train)
        return action_prediction_model

    def get_reaction_time(self, row):
        bins = self.reaction_time_bins_simulation
        last_reaction_time_cat = []
        reaction_time_cat =[]
        for lb, ub in zip(bins[:-1], bins[1:]):
            main_cat = "last_reaction_time"
            cat = f"{main_cat}{lb}_{ub}"
            row[cat] = (lb <= row[main_cat]) & (row[main_cat] < ub)
            if main_cat == "reaction_time":
                reaction_time_cat += [cat]
            else:
                last_reaction_time_cat += [cat]
        x_cols = self.basic_x_cols_simulation + last_reaction_time_cat
        x = np.array([[row[col] for col in x_cols]])
        y_proba = self.reaction_time_model.predict_proba(x)
        y_cat = self.sample_from_probability_vector(y_proba.flatten())
        y = bins[y_cat] + 1
        return y

    @staticmethod
    def sample_from_probability_vector(probabilities):
        # Select a random number between 0 and 1
        random_num = np.random.rand()

        # Initialize a variable to keep track of the cumulative probability
        cumulative_probability = 0

        # Iterate through the probabilities
        for i, probability in enumerate(probabilities):
            # Add the probability to the cumulative probability
            cumulative_probability += probability

            # If the random number is less than the cumulative probability, return the index
            if random_num < cumulative_probability:
                return i

    def get_hotel(self, hotel_id=None):
        if hotel_id is None:
            hotel_id = np.random.randint(DATA_N_HOTELS) + 1
        hotel = self.hotels[hotel_id]
        return hotel_id, hotel

    def get_review_id(self, hotel_id, review_number):
        return self.reviews_id[hotel_id, review_number]

    @staticmethod
    def bot_plays(bot_strategy, hotel, previous_rounds):
        bot_message = bot_strategy(hotel, previous_rounds)
        return bot_message

    @staticmethod
    def check_choice(hotel, action):
        return (hotel.mean() >= 8) == action

    def add_game(self, user, game):
        game = pd.DataFrame.from_records(game)
        self.users[user].append(game)

    def sample_bots(self):
        if self.zero_knowledge == False:
            if self.problem == 0:
                return [3, 0, 2, 5, 19, 59]
            else:
                return [132, 23, 107, 43, 17, 93]
        return random.sample(range(DATA_N_BOTS), self.bots_per_user)

    def new_user(self):
        user_id = self.next_user
        assert user_id < self.n_users
        user = self.SimulatedUser(user_improve=self.user_improve, basic_nature=self.basic_nature)
        bots = self.sample_bots()
        game_id = 0
        for bot in bots:
            user.return_to_init_proba()
            bot_strategy = getattr(bot_strategies, f"strategy_{bot}")
            correct_answers = 0
            games_until_winning = 0
            while (correct_answers < self.SIMULATION_TH) and not (self.user_improve == 0 and (games_until_winning == 100)):  # start a new game
                correct_answers = 0
                games_until_winning += 1
                previous_rounds = []
                game = []
                user_points, bot_points = 0, 0
                last_didGo, last_didWin = -1, -1
                last_reaction_time = -1
                for round_number in range(1, DATA_ROUNDS_PER_GAME + 1):  # start a new round
                    hotel_id, hotel = self.get_hotel()  # get a hotel

                    bot_message = self.bot_plays(bot_strategy, hotel, previous_rounds)  # expert plays
                    review_id = self.get_review_id(hotel_id, np.argmax(hotel == bot_message))

                    signal_error = np.random.normal(0, self.SIMULATION_SIGNAL_EPSILON)
                    user_action = self.play_round(bot_message + signal_error, user, previous_rounds,
                                                  hotel, review_id)  # DM plays
                    round_result = self.check_choice(hotel, user_action)  # round results
                    correct_answers += round_result

                    user.update_proba()  # update user vector
                    previous_rounds += [(hotel, bot_message, user_action)]

                    last_didGo_True = last_didGo == 1
                    last_didWin_True = last_didWin == 1
                    last_didGo_False = last_didGo == 0
                    last_didWin_False = last_didWin == 0

                    row = {"user_id": user_id, "strategy_id": bot, "gameId": game_id, "roundNum": round_number,
                           "hotelId": hotel_id, "reviewId": review_id, "hotelScore": float(f"{hotel.mean():.2f}"),
                           "didGo": user_action, "didWin": round_result, "correctAnswers": correct_answers,
                           "last_reaction_time": last_reaction_time,
                           "last_didWin_True": last_didWin_True, "last_didGo_True": last_didGo_True,
                           "last_didWin_False": last_didWin_False, "last_didGo_False": last_didGo_False,
                           "user_points": user_points, "bot_points": bot_points}

                    # if self.advanced_reaction_time:
                    #     last_reaction_time = self.get_reaction_time(row)

                    user_points += round_result
                    bot_points += user_action

                    last_didGo, last_didWin = int(user_action), int(round_result)
                    game.append(row)
                self.add_game(user_id, game)
                game_id += 1
        self.next_user += 1
        self.n_games_per_user[user_id] = game_id
        self.total_games_created += game_id
        self.active_users.append(user_id)

    def __len__(self):
        if self.next_user <= 50:
            return self.n_users * 42
        else:
            return int(self.n_users * self.total_games_created / self.next_user)

    def get_game(self, user_id):
        game = self.users[user_id].pop(0)
        if not len(self.users[user_id]):
            self.active_users.remove(user_id)
            if self.next_user < self.n_users:
                self.new_user()
        return game

    def __getitem__(self, user_id):
        game = self.get_game(user_id)

        user_id = game["user_id"][0]
        n_rounds = len(game)

        if n_rounds < DATA_ROUNDS_PER_GAME:
            game = pd.concat([game] + [DATA_BLANK_ROW_DF(game["strategy_id"][0])] * (DATA_ROUNDS_PER_GAME - n_rounds), ignore_index=True)

        bot_strategy = game["strategy_id"].to_numpy()
        bot_vector = self.bot_vectors[bot_strategy[0]]
        hotels_scores = game["hotelScore"].to_numpy()

        action_taken = game["didGo"].to_numpy()
        is_hotel_good = (game["didGo"] == game["didWin"]).to_numpy()
        weight = np.ones_like(game["didGo"])
        last_reaction_time = game["last_reaction_time"].to_numpy()

        user_points = game["user_points"].to_numpy()
        bot_points = game["bot_points"].to_numpy()
        user_earned_more = user_points >= bot_points
        user_not_earned_more = user_points < bot_points

        reviewId = game["reviewId"]
        game = {"user_id": user_id,
                "bot_strategy": bot_strategy,
                "bot_vector": bot_vector,
                "n_rounds": n_rounds,
                "hotels_scores": hotels_scores,
                "action_taken": action_taken,
                "is_hotel_good": is_hotel_good,
                "weight": weight,
                "last_reaction_time": last_reaction_time,
                "log_last_reaction_time": np.log(last_reaction_time + 2),
                "last_didGo_True": game["last_didGo_True"].to_numpy(),
                "last_didWin_True": game["last_didWin_True"].to_numpy(),
                "last_didGo_False": game["last_didGo_False"].to_numpy(),
                "last_didWin_False": game["last_didWin_False"].to_numpy(),
                "user_points": user_points / 10, "bot_points": bot_points /10,
                "user_earned_more": user_earned_more, "user_not_earned_more": user_not_earned_more,
                }
        for column_name, (lower, upper) in zip(reaction_time_columns_names, reaction_time_bins):
            game[column_name] = (lower <= last_reaction_time) & (last_reaction_time < upper)

        if LEARN_REVIEWS_VECTORS_IN_ADVANCE:
            game["review_vector"] = reviewId.apply(lambda r: self.review_reduced[r]).tolist()
            game["review_vector"] = torch.stack(game["review_vector"])
        else:
            review_positive = reviewId.apply(lambda r: self.reviews[r]["positive"]).fillna("").tolist()
            review_negative = reviewId.apply(lambda r: self.reviews[r]["negative"]).fillna("").tolist()
            game["review_encoded_positive"] = self.bert_tokenizer(review_positive, add_special_tokens=True,
                                                                  return_tensors='pt', padding="max_length",
                                                                  max_length=BERT_MAX_LENGTH, truncation=True)
            game["review_encoded_negative"] = self.bert_tokenizer(review_negative, add_special_tokens=True,
                                                                  return_tensors='pt', padding="max_length",
                                                                  max_length=BERT_MAX_LENGTH, truncation=True)
        return game
