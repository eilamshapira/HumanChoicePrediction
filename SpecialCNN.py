import torch
import torch.nn as nn
from utils.usersvectors import UsersVectors

class SpecialCNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, logsoftmax=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
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
        input_vec = input_vec.permute(0, 2, 1)  # Convert to [batch_size, channels, seq_length] for Conv1d
        conv_output = self.conv_layers(input_vec)
        batch_size, channels, seq_len = conv_output.size()
        conv_output = conv_output.permute(0, 2, 1).reshape(batch_size * seq_len, channels)

        user_vector = user_vector.reshape(batch_size, self.user_vectors.n_layers, -1)
        game_vector = game_vector.reshape(batch_size, self.game_vectors.n_layers, -1)

        output = self.fc_layers(conv_output)
        output = output.view(batch_size, seq_len, -1)

        # Debugging: print shapes
        """
        print(f"conv_output shape: {conv_output.shape}")
        print(f"output shape before view: {output.shape}")
        print(f"user_vector shape: {user_vector.shape}")
        print(f"game_vector shape: {game_vector.shape}")
        """
        if self.training:
            return {"output": output, "game_vector": game_vector, "user_vector": user_vector}
        else:
            return {"output": output, "game_vector": game_vector.detach(), "user_vector": user_vector.detach()}
