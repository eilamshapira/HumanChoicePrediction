import torch
import torch.nn as nn
from utils.usersvectors import UsersVectors

class CausalConv1d(nn.Conv1d):
    def forward(self, x):
        padding = (self.kernel_size[0] - 1, 0)
        x = nn.functional.pad(x, padding)
        return super().forward(x)

class SpecialCNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, logsoftmax=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.last_train_input = None

        self.conv_layers = nn.Sequential(
            CausalConv1d(in_channels=input_dim, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        if logsoftmax:
            self.fc_layers.add_module("log_softmax", nn.LogSoftmax(dim=-1))

        self.user_vectors = UsersVectors(user_dim=64, n_layers=2)
        self.game_vectors = UsersVectors(user_dim=64, n_layers=2)

    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)

    def forward(self, input_vec, game_vector, user_vector):
        #print(f"Stage: {'Training' if self.training else 'Testing'}")

        #print(f"Input vector shape: {input_vec.shape}")  # [batch_size, seq_len, features]
        input_vec = input_vec.permute(0, 2, 1)  # Convert to [batch_size, channels, seq_length] for Conv1d
        #print(f"Permuted input vector shape: {input_vec.shape}")  # [batch_size, channels, seq_length]

        conv_output = self.conv_layers(input_vec)
        batch_size, channels, seq_len = conv_output.size()
        #print(f"Conv output shape: {conv_output.shape}")  # [batch_size, channels, seq_length]

        conv_output = conv_output.permute(0, 2, 1).reshape(batch_size * seq_len, channels)
        #print(f"Reshaped conv output shape: {conv_output.shape}")  # [batch_size * seq_len, channels]

        user_vector = user_vector.reshape(batch_size, self.user_vectors.n_layers, -1)
        game_vector = game_vector.reshape(batch_size, self.game_vectors.n_layers, -1)
        #print(f"User vector shape: {user_vector.shape}")  # [batch_size, n_layers, 64]
        #print(f"Game vector shape: {game_vector.shape}")  # [batch_size, n_layers, 64]

        output = self.fc_layers(conv_output)
        output = output.view(batch_size, seq_len, -1)
        #print(f"Final output shape: {output.shape}")  # [batch_size, seq_len, output_dim]

        if self.training:
            return {"output": output, "game_vector": game_vector, "user_vector": user_vector}
        else:
            return {"output": output, "game_vector": game_vector.detach(), "user_vector": user_vector.detach()}

from environments import environment
import torch
from SpecialCNN import SpecialCNN
from consts import *

class CNN_env_ARC(SpecialCNN):
    def forward(self, vectors, **kwargs):
        data = super().forward(vectors["x"], vectors["game_vector"], vectors["user_vector"])
        return data

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        assert not update_vectors
        if vectors_in_input:
            output = self(data)
        else:
            output = self({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        return output

class CNN_env(environment.Environment):
    def init_model_arc(self, config):
        self.model = CNN_env_ARC(input_dim=config['input_dim'], output_dim=config["output_dim"], dropout=config["dropout"]).double()

    def predict_proba(self, data, update_vectors: bool, vectors_in_input=False):
        if vectors_in_input:
            output = self.model(data)
        else:
            output = self.model({**data, "user_vector": self.currentDM, "game_vector": self.currentGame})
        output["proba"] = torch.exp(output["output"].flatten())
        if update_vectors:
            self.currentDM = output["user_vector"]
            self.currentGame = output["game_vector"]
        return output

    def init_user_vector(self):
        self.currentDM = self.model.init_user()

    def init_game_vector(self):
        self.currentGame = self.model.init_game()

    def get_curr_vectors(self):
        return {"user_vector": self.currentDM, "game_vector": self.currentGame}