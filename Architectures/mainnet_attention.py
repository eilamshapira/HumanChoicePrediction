import sys
import torch
from torch import nn
from Architectures.reducebert import ReduceBert

sys.path.append("../")
from consts import *


class MainNetAttention(nn.Module):
    def __init__(self, n_layers, hidden_dim, include_bot_vector):
        super().__init__()
        if not LEARN_REVIEWS_VECTORS_IN_ADVANCE:
            self.bert_reducer = ReduceBert()

        self.include_bot_vector = include_bot_vector
        self.hidden_dim = hidden_dim

        input_dim = REVIEW_DIM +\
                    BOT_DIM * self.include_bot_vector + \
                    len(BINARY_BEHAVIORAL_FEATURES) + \
                    len(NUMERICAL_BEHAVIORAL_FEATURES)
        self.n_layers = n_layers

        self.fc = nn.Sequential(nn.Linear(input_dim, self.hidden_dim*2),
                                nn.Dropout(0.2),
                                nn.ReLU(),
                                nn.Linear(self.hidden_dim*2, self.hidden_dim),
                                nn.Dropout(0.2),
                                nn.ReLU()).double()

        self.main_task = nn.LSTM(input_size=self.hidden_dim,
                                 hidden_size=self.hidden_dim,
                                 batch_first=True,
                                 num_layers=self.n_layers,
                                 dropout=0.2).double()
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



    def init_history(self, batch_size):
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
        game["bot_vector"] = game["bot_vector"].repeat((1, review_vector.shape[1], 1))
        history_vector = self.init_history(batch_size=batch_size).to(device)
        game["user_vector"] = game["user_vector"].transpose(0, 1)

        review_vector = review_vector.reshape(batch_size, DATA_ROUNDS_PER_GAME, -1)

        lstm_input = [review_vector]
        if self.include_bot_vector:
            lstm_input.append(game["bot_vector"])
        if BINARY_BEHAVIORAL_FEATURES:
            lstm_input += [game[feature].unsqueeze(-1).to(torch.long)
                             for feature in BINARY_BEHAVIORAL_FEATURES]
        if NUMERICAL_BEHAVIORAL_FEATURES:
            lstm_input += [game[feature].unsqueeze(-1) for feature in NUMERICAL_BEHAVIORAL_FEATURES]
        lstm_input = torch.cat(lstm_input, dim=2).double()
        lstm_input = self.fc(lstm_input)
        output, (history_vector, game["user_vector"]) = self.main_task(lstm_input.contiguous(),
                                                                       (history_vector.contiguous(),
                                                                        game["user_vector"].contiguous()))
        output = output.reshape(batch_size * DATA_ROUNDS_PER_GAME, -1)
        predicted_result = self.main_task_classifier(output)[game["mask"].reshape(-1)]
        loss += (self.main_task_loss(predicted_result, game["y_true"].reshape(-1)[game["mask"].reshape(-1)])
                 * game["weight"].flatten()[game["mask"].reshape(-1)]).sum()
        y_pred = torch.argmax(predicted_result, dim=-1)

        game["user_vector"] = game["user_vector"].transpose(0, 1)
        history_vector = history_vector.squeeze()
        return y_pred, loss.double(), game["user_vector"], history_vector
