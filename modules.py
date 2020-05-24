"""
Implementation of https://arxiv.org/abs/1704.02971
"""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf


def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    return Variable(torch.zeros(1, x.size(0), hidden_size))


class Encoder(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        # Steps in order of execution
        self.attn_linear1 = nn.Linear(in_features=2 * hidden_size, out_features=T - 1)
        self.attn_linear2 = nn.Linear(in_features=T - 1, out_features=T - 1)
        self.tanh = nn.Tanh()
        self.attn_linear3 = nn.Linear(in_features=T - 1, out_features=1)
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)

    def forward(self, input_data):
        """
        input_data: (batch_size, T - 1, input_size)
        hidden, cell: initial states with dimension hidden_size
        """
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))
        hidden = init_hidden(input_data, self.hidden_size)
        cell = init_hidden(input_data, self.hidden_size)

        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden and cell
            z1_ = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                             cell.repeat(self.input_size, 1, 1).permute(1, 0, 2)), dim=2)
            z1 = self.attn_linear1(z1_)
            # Eqn. 8: input part
            z2 = self.attn_linear2(input_data.permute(0, 2, 1))
            x = z1 + z2
            # Eqn. 8: merging all together
            z3 = self.attn_linear3(self.tanh(x.view(-1, self.T - 1)))

            # Eqn. 9: Softmax the attention weights
            attn_weights = tf.softmax(z3.view(-1, self.input_size), dim=1)

            # Eqn. 10: LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]

            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()

        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer1 = nn.Linear(2 * decoder_hidden_size, encoder_hidden_size)
        self.attn_layer2 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.tanh = nn.Tanh()
        self.attn_layer3 = nn.Linear(encoder_hidden_size, out_feats)

        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        """
        input_encoded: (batch_size, T - 1, encoder_hidden_size)
        y_history: (batch_size, (T-1))
        Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        """
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.T - 1):
            # Eqn. 12: Concat hidden and cell
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2)), dim=2)
            z1 = self.attn_layer1(x)
            # Eqn. 12: Adding the input
            z2 = self.attn_layer2(input_encoded)
            # Eqn. 12: Combining them
            x = z1 + z2
            z3 = self.tanh(x)
            # Eqn. 12
            z4 = self.attn_layer3(z3)

            # Eqn. 13: softmax on the computed attention weights
            x = tf.softmax(z4.view(-1, self.T - 1), dim=1)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]

            # Eqn. 15: y^tilde
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))

            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]
            cell = lstm_output[1]

        # Eqn. 22: final output
        output = self.fc_final(torch.cat((hidden[0], context), dim=1))
        return output
