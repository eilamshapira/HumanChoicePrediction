import sys
import torch
from torch import nn
from Architectures.reducebert import ReduceBert
import numpy as np

sys.path.append("../")
from consts import *


class MainNetLSTM(nn.Module):
    def __init__(self, n_layers, hidden_dim, include_bot_vector, dropout):
        super().__init__()
        if not LEARN_REVIEWS_VECTORS_IN_ADVANCE:
            self.bert_reducer = ReduceBert()

        self.include_bot_vector = include_bot_vector
        self.hidden_dim = hidden_dim

        situation_dim = BOT_DIM * self.include_bot_vector + \
                        len(BINARY_BEHAVIORAL_FEATURES) * 2 + \
                        len(NUMERICAL_BEHAVIORAL_FEATURES)

        input_dim = REVIEW_DIM + situation_dim

        self.n_layers = n_layers

        if BOT_EMBEDDING:
            self.bot_embedding = torch.nn.Embedding(DATA_N_BOTS, BOT_DIM)
            self.bot_embedding.weight.data += torch.Tensor(np.genfromtxt(DATA_BOTS_VECTORS_PATH, delimiter=","))

        self.fc = nn.Sequential(nn.Linear(input_dim, input_dim * 2),
                                nn.Dropout(dropout),
                                nn.ReLU(),
                                nn.Linear(input_dim * 2, self.hidden_dim),
                                nn.Dropout(dropout),
                                nn.ReLU()).double()

        self.main_task = nn.LSTM(input_size=self.hidden_dim,
                                 hidden_size=self.hidden_dim,
                                 batch_first=True,
                                 num_layers=self.n_layers,
                                 dropout=dropout).double()

        self.main_task_classifier = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                                                  nn.ReLU(),
                                                  nn.Linear(self.hidden_dim // 2, N_CLASSES),
                                                  nn.LogSoftmax(dim=1)).double()


        self.softmax = nn.Softmax(dim=1)

        self.main_task_loss = nn.NLLLoss(reduction="none")

        self.init_history_vector = torch.rand(self.n_layers,
                                              self.hidden_dim,
                                              dtype=torch.double,
                                              requires_grad=True)

        self.init_cell_vector = torch.rand(self.n_layers,
                                              self.hidden_dim,
                                              dtype=torch.double,
                                              requires_grad=True)

    def init_history(self, batch_size):
        return torch.stack([self.init_history_vector] * batch_size, dim=1)

    def init_cell(self, batch_size):
        return torch.stack([self.init_history_vector] * batch_size, dim=1)

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
        if BOT_EMBEDDING:
            bot_vector = self.bot_embedding(game["bot_strategy"])
        else:
            bot_vector = game["bot_vector"].repeat((1, review_vector.shape[1], 1))

        history_vector = self.init_history(batch_size=batch_size).to(device)
        #game_cell = self.init_cell(batch_size=batch_size).to(device)

        user_cell = game["user_vector"].transpose(0, 1)

        #cell_vector = torch.cat((user_cell[:1], game_cell[1:]), dim=0)
        cell_vector = user_cell

        review_vector = review_vector.reshape(batch_size, DATA_ROUNDS_PER_GAME, -1)

        lstm_input = [review_vector]
        situation_features = []

        if self.include_bot_vector:
            situation_features.append(bot_vector)
        if BINARY_BEHAVIORAL_FEATURES:
            situation_features += [torch.nn.functional.one_hot(game[feature].to(torch.long), 2)
                                   for feature in BINARY_BEHAVIORAL_FEATURES]
        if NUMERICAL_BEHAVIORAL_FEATURES:
            situation_features += [game[feature].unsqueeze(-1) for feature in NUMERICAL_BEHAVIORAL_FEATURES]
        # situation = torch.cat(situation_features, dim=2).double()
        # situation = self.fc_situation(situation)
        lstm_input += [*situation_features]

        lstm_input = torch.cat(lstm_input, dim=2).double()
        lstm_input = self.fc(lstm_input)

        output, (history_vector, cell_vector) = self.main_task(lstm_input.contiguous(),
                                                                       (history_vector.contiguous(),
                                                                        cell_vector.contiguous()))
        output = output.reshape(batch_size * DATA_ROUNDS_PER_GAME, -1)
        predicted_result = self.main_task_classifier(output)[game["mask"].reshape(-1)]
        loss += (self.main_task_loss(predicted_result, game["y_true"].reshape(-1)[game["mask"].reshape(-1)])
                 * game["weight"].flatten()[game["mask"].reshape(-1)]).mean()
        y_pred = torch.argmax(predicted_result, dim=-1)
        y_logproba = predicted_result[:, 1]

        game["user_vector"] = cell_vector.transpose(0, 1)
        history_vector = history_vector.squeeze()
        return y_pred, loss.double(), game["user_vector"], history_vector, y_logproba
