import torch
from torch import nn
import numpy as np


class ClassifierRNN(nn.Module):

    def __init__(self, num_classes, num_step, ckpt, input_size, hidden_size, num_layers, drop_rate):
        super().__init__()

        self.ckpt = ckpt

        self.num_classes = num_classes
        self.num_step = num_step

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  #, dropout=drop_rate)

        self.classifier = nn.Sequential(
            nn.Linear(num_step * hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(64, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def save(self):
        state_dict = self.state_dict()
        torch.save(state_dict, self.ckpt)
        print("Classifier RNN was saved.")

    def load(self):
        state_dict = torch.load(self.ckpt)
        self.load_state_dict(state_dict)
        print("Classifier RNN was loaded.")

    def forward(self, x, hidden):
        assert(x.shape[1] == self.num_step)

        res_out, res_hidden = self.lstm(x)
        res_out = res_out.contiguous()
        res_out = res_out.view(-1, self.num_step * self.hidden_size)

        x = self.classifier(res_out)

        return x, res_hidden