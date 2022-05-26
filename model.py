import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

Feature_DIM = 64

# Hyper Parameters
TIME_STEP = 25  # rnn time step / music step length
INPUT_SIZE = 201  # rnn input size / fft window


class CGN(nn.Module):
    def __init__(self, hidden_size):
        super(CGN, self).__init__()
        self.hidden_size = hidden_size
        self.transform = nn.Linear(INPUT_SIZE, 32)
        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=32,
            hidden_size=256,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(256, 80)

    def forward(self, data):
        data = self.transform(data)
        output, (h_n, h_c) = self.rnn(data, None)
        output = self.out(output[:, -1, :])
