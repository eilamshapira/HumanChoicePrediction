import sklearn
import torch.nn as nn
from torch.utils.data import DataLoader
from consts import *
from transformers import BertModel
import os
from tqdm import tqdm
from utils.datasets import RealDataSet, FineTuningDataSet, SimulationDataSet
from utils.samplers import SimulationSampler, NewUserBatchSampler
from utils.evaluation import update_metrics, plot_metrics_graph, metrics_for_wandb
from utils.usersvectors import UsersVectors
from utils.functions import rounds_mask, move_to, get_model_name, set_global_seed
import wandb
from utils.results_saver import ResultsSaver
from Architectures.mainnetlstm import MainNetLSTM
from Architectures.mainnet_attention import MainNetAttention
from Architectures.mainnettransformer import MainNetTransformer


main_run = wandb.init(project='MSC_simulations')
config = wandb.config

# config = {"seed": 23,
#           "architecture": "Transformer",
#           "model_n_layers": 4,
#           "lr": 4e-4,
#           "hidden_dim": 64,
#           "include_bot_vector": False,
#           "simulation_size": 0,
#           "simulation_bot_per_user": 6,
#           "is_test": False,
#           "online_simulation_size": 100
# }

default_values = {
    "online_simulation_size": 0,
    "basic_nature": 24,
    "problem": 3,
    "advanced_reaction_time_in_simulation": False,
    "training_just_over_ms": 0,
    "simulation_user_improve": 0.01,
    "simulation_signal_error": 0.3,
    "dropout": 0.2,
    "n_head": 4,
    "zero_knowledge": True
}

for param, value in default_values.items():
    if param not in config.keys():
        config[param] = value

if config["training_just_over_ms"]:
    over = config["training_just_over_ms"]
    DATA_CLEAN_ACTION_PATH_X = f"data/games_clean_X_reactiontime_over_{over}.csv"
    DATA_CLEAN_ACTION_PATH_Y = f"data/games_clean_Y_reactiontime_over_{over}.csv"

if config["architecture"] == "LSTM":
    USER_DIM = config["hidden_dim"]
elif config["architecture"] == "SimpleLSTM":
    # USER_DIM = USER_DIM_SimpleLSTM
    raise NotImplementedError
elif config["architecture"] == "Transformer":
    USER_DIM = 1 # I'll don't use it
    #raise NotImplementedError
else:
    raise ValueError

BERT_PARAMS = {key: val for key, val in config.items() if key in ["seed", "problem", "ffcv", "basic_nature"]}
SIMULATION_PARAMS = {key: val for key, val in config.items() if key in ["architecture",
                                                                        "model_n_layers",
                                                                        "lr",
                                                                        "hidden_dim",
                                                                        "include_bot_vector",
                                                                        "simulation_size",
                                                                        "is_test",
                                                                        "simulation_bot_per_user",
                                                                        "online_simulation_size",
                                                                        "advanced_reaction_time_in_simulation"
                                                                        ]}
SIMULATION_PARAMS.update(BERT_PARAMS)

print(f"{TRAIN_BATCH_SIZE=}, {TEST_BATCH_SIZE=}, {SIMULATION_BATCH_SIZE=}")
print(f"{config=}, {BERT_PARAMS=}, {SIMULATION_PARAMS=}")
print(f"{hash(str(config))}")


# set seed
set_global_seed(config["seed"])


class ReduceBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.reduce_positive_dim = nn.Sequential(nn.Linear(BERT_DIM, REVIEW_DIM))
        self.reduce_negative_dim = nn.Sequential(nn.Linear(BERT_DIM, REVIEW_DIM))
        self.reduce_review_dim = nn.Sequential(nn.Linear(REVIEW_DIM * 2, REVIEW_DIM))

    def forward(self, review):
        review_encoded_positive, review_encoded_negative = review

        positive_vector = self.bert(**review_encoded_positive)["pooler_output"]
        negative_vector = self.bert(**review_encoded_negative)["pooler_output"]

        positive_vector = self.reduce_positive_dim(positive_vector)
        negative_vector = self.reduce_negative_dim(negative_vector)

        review_vector = torch.cat((positive_vector, negative_vector), dim=1)
        review_vector = self.reduce_review_dim(review_vector)
        return review_vector


class FineTuneBert(nn.Module):
    def __init__(self, return_review=False):
        super().__init__()
        self.bert_reducer = ReduceBert()
        self.linear = nn.Linear(REVIEW_DIM, 1)
        self.loss = nn.MSELoss()
        self.return_review = return_review

    def forward(self, review):
        for d in [review["review_positive"], review["review_negative"]]:
            for k, v in d.items():
                d[k] = d[k].squeeze()
        review_vector = self.bert_reducer((review['review_positive'], review['review_negative']))
        predicted_score = self.linear(review_vector)
        loss = self.loss(predicted_score, review["score"])
        if self.return_review:
            return review_vector
        return predicted_score, loss.type(torch.DoubleTensor)


class MainNetSimpleLSTM(nn.Module):
    def __init__(self, review_dim=REVIEW_DIM, n_layers=config["model_n_layers"]):
        super().__init__()
        if not LEARN_REVIEWS_VECTORS_IN_ADVANCE:
            self.bert_reducer = ReduceBert()

        input_dim = REVIEW_DIM + \
                    BOT_DIM * config["include_bot_vector"] + \
                    USER_DIM_SimpleLSTM * config["include_user_vector"] + \
                    len(BINARY_BEHAVIORAL_FEATURES) + \
                    len(NUMERICAL_BEHAVIORAL_FEATURES)
        self.n_layers = n_layers
        # self.ff1 = nn.Sequential(nn.Linear(input_dim, 4 * config["hidden_dim"]),
        #                          nn.ReLU(),
        #                          nn.Linear(4 * config["hidden_dim"], 2 * config["hidden_dim"]),
        #                          nn.ReLU(),
        #                          nn.Linear(2 * config["hidden_dim"], config["hidden_dim"]),
        #                          ).double()

        self.ff1 = nn.Sequential(nn.Linear(input_dim, 4 * config["hidden_dim"]),
                                 nn.ReLU(),
                                 nn.Linear(4 * config["hidden_dim"], 2 * config["hidden_dim"]),
                                 nn.ReLU(),
                                 nn.Linear(2 * config["hidden_dim"], config["hidden_dim"]),
                                 ).double()

        self.main_task = nn.LSTM(input_size=config["hidden_dim"],
                                 hidden_size=config["hidden_dim"],
                                 batch_first=True,
                                 num_layers=self.n_layers).double()
        self.main_task_classifier = nn.Sequential(nn.ReLU(),
                                                  nn.Linear(config["hidden_dim"], N_CLASSES),
                                                  nn.LogSoftmax(dim=1)).double()

        self.softmax = nn.Softmax(dim=1)

        self.main_task_loss = nn.NLLLoss(reduction="none")

        self.init_history_vector = torch.rand(self.n_layers,
                                              config["hidden_dim"],
                                              dtype=torch.double,
                                              requires_grad=True)

        self.init_cell_vector = torch.rand(self.n_layers,
                                           config["hidden_dim"],
                                           dtype=torch.double,
                                           requires_grad=True)

    def init_history(self, batch_size):
        return torch.stack([self.init_history_vector] * batch_size, dim=1)

    def init_cell(self, batch_size):
        return torch.stack([self.init_cell_vector] * batch_size, dim=1)

    def forward(self, game):
        loss = 0.0
        batch_size, _ = game["hotels_scores"].shape
        game["bot_vector"] = game["bot_vector"].reshape(batch_size, 1, -1)

        if LEARN_REVIEWS_VECTORS_IN_ADVANCE:
            review_vector = game["review_vector"]
        else:
            # get BERT vectors
            for d in [game["review_encoded_positive"], game["review_encoded_negative"]]:
                for k, v in d.items():
                    d[k] = d[k].reshape((batch_size * DATA_ROUNDS_PER_GAME, -1))
            review = game["review_encoded_positive"], game["review_encoded_negative"]
            review_vector = self.bert_reducer(review)
            review_vector = review_vector.reshape(batch_size, DATA_ROUNDS_PER_GAME, 1, -1)

        # reshape entities vectors
        game["bot_vector"] = game["bot_vector"].repeat((1, review_vector.shape[1], 1))
        game["user_vector"] = game["user_vector"].repeat((1, review_vector.shape[1], 1))
        history_vector = self.init_history(batch_size=batch_size).to(device)
        cell_vector = self.init_history(batch_size=batch_size).to(device)

        review_vector = review_vector.reshape(batch_size, DATA_ROUNDS_PER_GAME, -1)

        lstm_input = [review_vector]
        if config["include_bot_vector"]:
            lstm_input.append(game["bot_vector"])
        if config["include_user_vector"]:
            lstm_input.append((game["user_vector"]))
        lstm_input = torch.cat(lstm_input, dim=2).double()

        add_features = []
        if BINARY_BEHAVIORAL_FEATURES:
            add_features += [game[feature].unsqueeze(-1).to(torch.long)
                             for feature in BINARY_BEHAVIORAL_FEATURES]
        if NUMERICAL_BEHAVIORAL_FEATURES:
            add_features += [game[feature].unsqueeze(-1) for feature in NUMERICAL_BEHAVIORAL_FEATURES]
        if add_features:
            lstm_input = torch.cat([lstm_input] + add_features, dim=2)

        lstm_input = self.ff1(lstm_input)

        # output, (history_vector, cell_vector) = self.main_task(lstm_input.contiguous(),
        #                                                        (history_vector.contiguous(),
        #                                                         cell_vector.contiguous()))
        # output = output.reshape(batch_size * DATA_ROUNDS_PER_GAME, -1)
        # predicted_result = self.main_task_classifier(output)[game["mask"].reshape(-1)]
        predicted_result = self.main_task_classifier(lstm_input.reshape(batch_size * DATA_ROUNDS_PER_GAME, -1))[
            game["mask"].reshape(-1)]

        loss += (self.main_task_loss(predicted_result, game["y_true"].reshape(-1)[game["mask"].reshape(-1)])
                 * game["weight"].flatten()[game["mask"].reshape(-1)]).sum()
        y_pred = torch.argmax(predicted_result, dim=-1)

        history_vector = history_vector.squeeze()
        return y_pred, loss, game["user_vector"], history_vector


class ConfusionMatricesByCondition():
    def __init__(self, conditions, ordinal=True):
        self.ordinal = ordinal
        self.confusion_matrix = {"all": torch.zeros((N_CLASSES, N_CLASSES))}
        for condition in conditions:
            self.confusion_matrix[condition] = torch.zeros((N_CLASSES, N_CLASSES))

    def update(self, y_pred, y_true, condition):
        for _y_pred, _y_true, _condition in zip(y_pred, y_true, condition):
            for _matrix_condition in self.confusion_matrix.keys():
                if _matrix_condition == "all":
                    continue
                if (self.ordinal and (_matrix_condition <= _condition)) or \
                        (not self.ordinal and (_matrix_condition == _condition)):
                    self.confusion_matrix[_matrix_condition][_y_pred.item()][_y_true.item()] += 1
            self.confusion_matrix["all"][_y_pred.item()][_y_true.item()] += 1

    def get_matrices(self):
        results = {}
        for condition, matrix in self.confusion_matrix.items():
            precision = torch.diag(matrix) / matrix.sum(axis=0)
            recall = torch.diag(matrix) / matrix.sum(axis=1)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            results[f"accuracy_{condition}"] = np.diag(matrix).sum() / matrix.sum()
            results[f"macro_f1_{condition}"] = f1_scores.mean()

            results[f"accuracy_{condition}"][results[f"accuracy_{condition}"] != results[f"accuracy_{condition}"]] = 0
            results[f"macro_f1_{condition}"][results[f"macro_f1_{condition}"] != results[f"macro_f1_{condition}"]] = 0

            results[f"accuracy_{condition}"] = results[f"accuracy_{condition}"].item()
            results[f"macro_f1_{condition}"] = results[f"macro_f1_{condition}"].item()
        return results

    def __str__(self):
        string = ""
        metrics = self.get_matrices()
        for condition in self.confusion_matrix.keys():
            string += f"reaction time >= {condition}:\n{self.confusion_matrix[condition].numpy()}"
            for metric in metrics.keys():
                if f"_{condition}" in metric:
                    string += f"\n{metric}: {metrics[metric]}"
            string += "\n\n"
        return string


def train_loop(model, optimizer, train_dataset, val_dataset=None, epoch=-1, train_strategies=[], test_strategies=[]):
    phases = []
    if train_dataset is not None:
        phases.append("train")
    if val_dataset is not None:
        phases.append("test")

    cac_metrics = ConfusionMatricesByCondition(conditions=train_strategies + test_strategies,
                                               ordinal=False).get_matrices().keys()
    metrics = {metric: {p: [] for p in phases} for metric in list(cac_metrics)+["AUC"]}

    all_user_vectors = UsersVectors(user_dim=USER_DIM,
                                    n_layers=config["model_n_layers"] if config["architecture"] == "LSTM" else 1)

    pred_log = {}
    for phase in phases:

        # confusion_matrix_by_reaction_time = ConfusionMatricesByCondition(conditions=CONFUSION_MATRIX_REACTION_TIMES)
        all_cm = []
        if phase == "train":
            model.train()
            dataloader = train_dataset
        elif phase == "test":
            model.eval()
            dataloader = val_dataset
        else:
            raise ValueError
        confusion_matrix_by_strategy = ConfusionMatricesByCondition(conditions=train_strategies + test_strategies,
                                                                    ordinal=False)
        all_cm += [confusion_matrix_by_strategy]

        results_log = []
        all_y_true = []

        t_bar = tqdm(dataloader)
        total_loss = 0
        for game in t_bar:
            t_bar.total = dataloader.__len__()
            game["user_vector"] = all_user_vectors[game["user_id"]]
            game["mask"] = rounds_mask(game["n_rounds"])
            game["y_true"] = game["action_taken"].type(torch.LongTensor)
            game = move_to(game, device)

            if phase == "train":
                y_pred, loss, user_vector, history_vector, y_logproba = model(game)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                y_pred = y_pred.to("cpu").reshape(-1)
            elif phase == "test":
                with torch.no_grad():
                    y_pred, loss, user_vector, history_vector, y_logproba = model(game)
                    y_pred = y_pred.to("cpu").reshape(-1)
                    y_logproba = y_logproba.to("cpu").reshape(-1)
                    if "group" in game.keys():
                        i = 0
                        for x in range(len(game["n_rounds"])):
                            for y in range(max(game["n_rounds"])):
                                if game["n_rounds"][x] > y:
                                    results_log += [(game["group"][0][x].item(), game["group"][1][x].item(),
                                                     y, y_pred[i].item(), y_logproba[i].item())]
                                    i += 1

            total_loss += loss.item()
            game = move_to(game, "cpu")
            all_user_vectors[game["user_id"]] = user_vector.to("cpu")

            y_true = game["y_true"].reshape(-1)[game["mask"].reshape(-1)]
            all_y_true += [_y.item() for _y in list(y_true)]
            # if "reaction_time" in game.keys():
            #     confusion_matrix_by_reaction_time.update(y_pred, y_true, game["reaction_time"].reshape(-1)[game["mask"].reshape(-1)])

            confusion_matrix_by_strategy.update(y_pred, y_true,
                                                game["bot_strategy"].reshape(-1)[game["mask"].reshape(-1)])
            for _cm in all_cm:
                for metric, metric_value in _cm.get_matrices().items():
                    metrics[metric][phase] = metric_value
            del game
        print("total loss: ", total_loss)

        results_df = pd.DataFrame(results_log, columns=["user_id", "gameId", "roundNum", "pred", "proba"])

        def get_auc_score(y, y_log_proba):
            pred = np.e ** np.array(y_log_proba)
            y = np.array(y)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, pred)
            auc = sklearn.metrics.auc(fpr, tpr)
            return auc

        if len(results_log):
            auc = get_auc_score(all_y_true, [l[4] for l in results_log])
            metrics["AUC"][phase] = auc
            print("auc score:", auc)
        else:
            metrics["AUC"][phase] = -1

        if SAVE_TAGS and len(results_log):
            p=config["run_hash"]
            results_df.to_csv(f"{PREDS_PATH}/{p}_epoch{epoch}{phase}")
            print("results_saved.")

    return metrics


def get_fine_tuned_bert(bert_params):
    def training_loop_bert(model, optimizer, train_dataloader, val_dataloader=None):
        phases = []
        if train_dataloader is not None:
            phases.append("train")
        if val_dataloader is not None:
            phases.append("test")

        cac_metrics = ["loss"]
        metrics = {metric: {p: [] for p in phases} for metric in cac_metrics}

        for phase in phases:
            if phase == "train":
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader

            total_loss = 0.0
            n_samples = 0
            t_bar = tqdm(dataloader)

            for review in t_bar:
                review = move_to(review, device)
                n_samples += len(review["score"])

                if phase == "train":
                    predicted_score, loss = model(review)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif phase == "test":
                    with torch.no_grad():
                        predicted_score, loss = model(review)

                total_loss += loss.item()
                mean_loss = total_loss / n_samples
                t_bar.set_description(f"loss={mean_loss}")
                metrics["loss"][phase] = mean_loss
        return metrics

    if not os.path.exists(FINE_TUNED_MODELS_PATH):
        os.makedirs(FINE_TUNED_MODELS_PATH)
    model_name = get_model_name(bert_params, "BertFineTuned")
    model_path = FINE_TUNED_MODELS_PATH + "/" + model_name
    reviews_path = model_path + "_reviews"

    # training
    fine_tuning_model = FineTuneBert()
    fine_tuning_model.to(device)

    if os.path.exists(model_path):
        fine_tuned_bert = fine_tuning_model.bert_reducer
        fine_tuned_bert.load_state_dict(torch.load(model_path))
    else:
        fine_tuning_optimizer = torch.optim.Adam(fine_tuning_model.parameters(), lr=FINE_TUNING_LR)
        fine_tuning_dataset = FineTuningDataSet()
        fine_tuning_dataloader = DataLoader(fine_tuning_dataset, batch_size=16)

        metrics = {}
        for epoch in range(FINE_TUNING_MAX_EPOCHS):
            print(f"-- Fine-Tuning: epoch {epoch} --")
            epoch_metrics = training_loop_bert(fine_tuning_model, fine_tuning_optimizer, fine_tuning_dataloader)
            metrics = update_metrics(metrics, epoch_metrics)
        plot_metrics_graph(metrics)

        fine_tuned_bert = fine_tuning_model.bert_reducer
        torch.save(fine_tuned_bert.state_dict(), model_path)

    # reduce all reviews
    if not os.path.exists(reviews_path):
        reviews_dataset = FineTuningDataSet(data_type="GameReviews")
        reviews_dataloader = DataLoader(reviews_dataset, batch_size=8)

        reduced_reviews = {}
        fine_tuning_model.return_review = True
        fine_tuning_model.eval()
        with torch.no_grad():
            t_bar = tqdm(reviews_dataloader)
            for review in t_bar:
                review_verbal_features = fine_tuning_model(move_to(review, device))
                for rid, vector in zip(review["review_id"], review_verbal_features):
                    reduced_reviews[rid.item()] = vector
        torch.save(reduced_reviews, reviews_path)
    return fine_tuned_bert


def train_test_split(user_ids, test_p=PROBLEM_1_P_USERS_TEST):
    n_users = len(user_ids)
    set_global_seed(42)
    idx = torch.randperm(n_users)[:int(n_users * test_p)]
    set_global_seed(config["seed"])
    test_users = user_ids[idx]
    train_users = user_ids[[i for i in range(n_users) if i not in idx]]
    return train_users, test_users


def get_problem_groups(problem, is_test):
    train_users = None
    test_users = None
    if config["problem"] == 0:
        train_users_groups = ["X"]
        test_users_groups = ["X"]
        train_users = torch.arange(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS)
        train_users, test_users = train_test_split(train_users)
        if config["is_test"]:
            train_strategies = list(bot2strategy_X.values())
            test_strategies = list(bot2strategy_X.values())
        else:
            train_strategies = list(bot2strategy_X.values())[:4]
            test_strategies = list(bot2strategy_X.values())[:4]
            train_users, test_users = train_test_split(train_users)
    elif config["problem"] == 1:
        train_users_groups = ["X"]
        test_users_groups = ["X"]
        train_users = torch.arange(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS)
        train_users, test_users = train_test_split(train_users)
        if config["is_test"]:
            train_strategies = list(bot2strategy_X.values())[:4]
            test_strategies = list(bot2strategy_X.values())[4:]
        else:
            train_strategies = list(bot2strategy_X.values())[:2]
            test_strategies = list(bot2strategy_X.values())[2:4]
            train_users, test_users = train_test_split(train_users)
    elif config["problem"] == 2:
        train_users_groups = ["X"]
        test_users_groups = ["X"]
        if config["is_test"]:
            train_strategies = list(bot2strategy_X.values())[:4]
            test_strategies = list(bot2strategy_X.values())[4:]
        else:
            train_strategies = list(bot2strategy_X.values())[:2]
            test_strategies = list(bot2strategy_X.values())[2:4]
    elif config["problem"] == 3:
        train_users_groups = ["X"]
        if config["is_test"]:
            test_users_groups = ["Y"]
            train_strategies = list(bot2strategy_X.values())
            test_strategies = list(bot2strategy_Y.values())
        else:
            test_users_groups = ["X"]
            train_strategies = list(bot2strategy_X.values())[:4]
            test_strategies = list(bot2strategy_X.values())[4:]
            train_users = torch.arange(DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS)
            train_users, test_users = train_test_split(train_users)
    elif config["problem"] == 4:
        if config["is_test"]:
            test_users_groups = ["Y"]
            train_users_groups = ["Y"]
            test_users = torch.arange(DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS)[config["ffcv"]::5]
            train_users = torch.tensor([u for u in torch.arange(DATA_CLEAN_ACTION_PATH_Y_NUMBER_OF_USERS) if u not in test_users])
            train_users += DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS
            test_users += DATA_CLEAN_ACTION_PATH_X_NUMBER_OF_USERS
            train_strategies = list(bot2strategy_Y.values())
            test_strategies = list(bot2strategy_Y.values())
        else:
            raise ValueError
    else:
        raise ValueError
    return train_users_groups, test_users_groups, train_strategies, test_strategies, train_users, test_users


if config["architecture"] == "LSTM":
    model = MainNetLSTM(n_layers=config["model_n_layers"],
                        include_bot_vector=config["include_bot_vector"],
                        hidden_dim=config["hidden_dim"],
                        dropout=config["dropout"])
elif config["architecture"] == "Attention":
    model = MainNetAttention(n_layers=config["model_n_layers"],
                             include_bot_vector=config["include_bot_vector"],
                             hidden_dim=config["hidden_dim"])
elif config["architecture"] == "SimpleLSTM":
    model = MainNetSimpleLSTM()
elif config["architecture"] == "Transformer":
    model = MainNetTransformer(n_layers=config["model_n_layers"],
                               include_bot_vector=config["include_bot_vector"],
                               hidden_dim=config["hidden_dim"],
                               nhead=config["n_head"],
                               dropout=config["dropout"])
else:
    raise ValueError

# Stage 1: Fine tune BERT
if not REVIEW_FEATURES:
    print("#" * 18, "\n# Fine tune BERT #", "\n" + "#" * 18)
    model.bert_reducer = get_fine_tuned_bert(BERT_PARAMS)

# Stage 2: Simulation
print("#" * 14, "\n# Simulation #", "\n" + "#" * 14)

simulation_hash = hash(str(SIMULATION_PARAMS))
simulation_path = f"{CHECKPOINT_DIR}/{simulation_hash}.ckpt"

train_users_groups, test_users_groups, train_strategies, test_strategies, train_users, test_users = \
    get_problem_groups(config["problem"], config["is_test"])

if not os.path.exists(simulation_path) or DO_SIM_ANYWAY:
    simulation_dataset = SimulationDataSet(n_users=config["simulation_size"], bert_params=BERT_PARAMS,
                                           bots_per_user=config["simulation_bot_per_user"],
                                           basic_nature=config["basic_nature"],
                                           strategies_for_reaction_time=train_strategies,
                                           advanced_reaction_time=config["advanced_reaction_time_in_simulation"],
                                           signal_error=config["simulation_signal_error"],
                                           user_improve=config["simulation_user_improve"],
                                           zero_knowledge=config["zero_knowledge"],
                                           problem=config["problem"])
    simulation_sampler = SimulationSampler(simulation_dataset, SIMULATION_BATCH_SIZE)
    simulation_dataloader = DataLoader(simulation_dataset, batch_sampler=simulation_sampler, shuffle=False)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    train_loop(model, optimizer, simulation_dataloader)

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    orig_model = model.to("cpu")
    torch.save(orig_model.state_dict(), simulation_path)

# Stage 3: Training
print("#" * 12, "\n# Training #", "\n" + "#" * 12)

weighting_reaction_time= False
online_simulation_size = config["online_simulation_size"]
config["weighting_reaction_time"] = weighting_reaction_time
print(config)
model.load_state_dict(torch.load(simulation_path))
model.to(device)

if BOT_EMBEDDING:
    model.bot_embedding.weight.requires_grad = False
else:
    print("There is no bot embedding.")

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["lr"])
print("Number of model's parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHEDULER_LR_GAMMA)

config["run_hash"] = hash(str(config))
result_saver = ResultsSaver(config)

train_dataset = RealDataSet(user_groups=train_users_groups,
                            strategies=train_strategies,
                            users=train_users,
                            weighting_function=None if config["weighting_reaction_time"] else lambda x: 1,
                            bert_params=BERT_PARAMS,
                            x_path=DATA_CLEAN_ACTION_PATH_X,
                            y_path=DATA_CLEAN_ACTION_PATH_Y
                            )
weighting_function = train_dataset.weighting_function

test_dataset = RealDataSet(user_groups=test_users_groups,
                           strategies=test_strategies,
                           users=test_users,
                           weighting_function=weighting_function,
                           bert_params=BERT_PARAMS,
                           x_path=DATA_CLEAN_ACTION_PATH_X,
                           y_path=DATA_CLEAN_ACTION_PATH_Y)


train_sampler = NewUserBatchSampler(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
test_sampler = NewUserBatchSampler(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_sampler=test_sampler, shuffle=False)

metrics = {}
# Train and Evaluation
no_training_metrics = train_loop(model, optimizer, train_dataset=None, val_dataset=test_dataloader,
                                 train_strategies=train_strategies, test_strategies=test_strategies)
metrics = update_metrics(metrics, no_training_metrics)
for k in metrics.keys():
    for pashe in metrics[k].keys():
        print(k, pashe, metrics[k][pashe][-1], "best so far: ", max(metrics[k][pashe]))
result_saver.log(-1, metrics)

for epoch in range(MAX_EPOCHS):
    print(f"-- epoch {epoch} --")
    if config["online_simulation_size"]:
        print("Online Simulation")
        online_simulation_dataset = SimulationDataSet(n_users=config["online_simulation_size"], bert_params=BERT_PARAMS,
                                                      bots_per_user=config["simulation_bot_per_user"],
                                                      strategies_for_reaction_time=train_strategies,
                                                      advanced_reaction_time=config["advanced_reaction_time_in_simulation"],
                                                      signal_error=config["simulation_signal_error"],
                                                      user_improve=config["simulation_user_improve"],
                                                      zero_knowledge=config["zero_knowledge"])
        online_simulation_sampler = SimulationSampler(online_simulation_dataset, SIMULATION_BATCH_SIZE)
        online_simulation_dataloader = DataLoader(online_simulation_dataset,
                                                  batch_sampler=online_simulation_sampler, shuffle=False)
        train_loop(model, optimizer, online_simulation_dataloader)

    epoch_metrics = train_loop(model, optimizer, train_dataloader, test_dataloader, epoch=epoch,
                               train_strategies=train_strategies, test_strategies=test_strategies)
    metrics = update_metrics(metrics, epoch_metrics)
    try:
        wandb.log(metrics_for_wandb(metrics))
    except:
        print("No logging to W&B.")
    result_saver.log(epoch, metrics)
    # scheduler.step()
    for k in metrics.keys():
        for pashe in metrics[k].keys():
            if max(metrics[k][pashe]) > 0 and ("accuracy_all" in k) or ("AUC" in k):
                print(k, pashe, metrics[k][pashe][-1], "best so far: ", max(metrics[k][pashe]))
plot_metrics_graph(metrics)
model.to("cpu")
os.remove(simulation_path)
