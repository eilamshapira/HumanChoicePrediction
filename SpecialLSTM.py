import torch
import torch.nn as nn
from utils.usersvectors import UsersVectors


class SpecialLSTM(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dropout, logsoftmax=True, input_twice=False):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.input_twice = input_twice

        self.input_fc = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(input_dim * 2, self.hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.main_task = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=self.n_layers,
            dropout=dropout
        )

        seq = [nn.Linear(self.hidden_dim + (input_dim if self.input_twice else 0), self.hidden_dim // 2),
               nn.ReLU(),
               nn.Linear(self.hidden_dim // 2, self.output_dim)]
        if logsoftmax:
            seq += [nn.LogSoftmax(dim=-1)]

        self.output_fc = nn.Sequential(*seq)

        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)

    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)

    def forward(self, input_vec, game_vector, user_vector):
        #print(f"Input vector shape: {input_vec.shape}")  # [batch_size, seq_len, features]
        lstm_input = self.input_fc(input_vec)
        #print(f"LSTM input shape after input_fc: {lstm_input.shape}")  # [batch_size, seq_len, hidden_dim]

        lstm_shape = lstm_input.shape
        shape = user_vector.shape
        assert game_vector.shape == shape

        if len(lstm_shape) != len(shape):
            lstm_input = lstm_input.reshape((1,) * (len(shape) - 1) + lstm_input.shape)

        #print(f"LSTM input shape after possible reshape: {lstm_input.shape}")
        #print(f"User vector shape before reshape: {user_vector.shape}")
        #print(f"Game vector shape before reshape: {game_vector.shape}")

        user_vector = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        game_vector = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))

        #print(f"User vector shape after reshape: {user_vector.shape}")
        #print(f"Game vector shape after reshape: {game_vector.shape}")

        lstm_output, (game_vector, user_vector) = self.main_task(
            lstm_input.contiguous(),
            (game_vector.contiguous(), user_vector.contiguous())
        )

        #print(f"LSTM output shape: {lstm_output.shape}")
        #print(f"User vector shape after LSTM: {user_vector.shape}")
        #print(f"Game vector shape after LSTM: {game_vector.shape}")

        user_vector = user_vector.reshape(shape)
        game_vector = game_vector.reshape(shape)

        #print(f"User vector shape after final reshape: {user_vector.shape}")
        #print(f"Game vector shape after final reshape: {game_vector.shape}")

        if hasattr(self, "input_twice") and self.input_twice:
            lstm_output = torch.cat([lstm_output, input_vec], dim=-1)

        output = self.output_fc(lstm_output)

        #print(f"Output shape before final reshape: {output.shape}")

        if len(output.shape) != len(lstm_shape):
            output = output.reshape(-1, output.shape[-1])

        #print(f"Final output shape: {output.shape}")

        if self.training:
            return {"output": output, "game_vector": game_vector, "user_vector": user_vector}
        else:
            return {"output": output, "game_vector": game_vector.detach(), "user_vector": user_vector.detach()}


# Example usage:
# model = SpecialLSTM(n_layers=2, input_dim=60, hidden_dim=128, output_dim=10, dropout=0.5)
# input_vec = torch.randn(4, 10, 60)  # Example input
# game_vector = model.init_game(batch_size=4)
# user_vector = model.init_user(batch_size=4)
# output = model(input_vec, game_vector, user_vector)
