import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd

class LSTM_Text(nn.Module):
    def __init__(self, input_size, hidden_size, share_size):
        super(LSTM_Text, self).__init__()
        self.hidden_size = hidden_size
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        # The linear layer that maps from hidden state space to tag space
        self.i2o = nn.Linear(hidden_size, share_size)
        C = 1
        self.fc1 = nn.Linear(share_size, C)
        self.fc2 = nn.Linear(share_size, C)

    def init_hidden(self):
        return autograd.Variable(torch.zeros(1, self.hidden_size))

    def forward(self, input, hidden, cell):
        hidden, cell = self.lstm(input, (hidden, cell))
        output = self.i2o(hidden)
        logit1 = self.fc1(output)
        logit2 = self.fc2(output)
        return logit1, logit2, hidden, cell