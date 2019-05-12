import torch
from torch import nn
import numpy as np
import sys
from ai.features.FeatureCNN import FeatureCNN
from ai.ClassifierRNN import ClassifierRNN

# FEATURE_CNN_CKPT = "D:/ckpts/capstone/torch/feature_cnn.pth"
# FEATURE_CNN_CKPT = "ckpts/feature_cnn.pth"
FEATURE_CNN_CKPT = "../ai/ckpts/feature_cnn_train4.pth"
# CLASSIFIER_RNN_CKPT = "D:/ckpts/capstone/torch/classifier_rnn.pth"
CLASSIFIER_RNN_CKPT = "../ai/ckpts/classifier4.pth"
NUM_CLASSES = 3
NUM_STEP = 8


class Classifier(nn.Module):

    def __init__(self, num_classes, drop_rate=0.5):
        super().__init__()

        self.num_classes = num_classes

        self.input_size = 256
        self.hidden_size = 32
        self.num_layers = 1
        self.drop_rate = drop_rate
        # self.hidden = None

        self.features = FeatureCNN(self.num_classes, drop_rate)
        self.features.load(FEATURE_CNN_CKPT)
        for param in self.features.parameters():
            param.requires_grad_(False)
        self.features.eval()

        self.classifier = ClassifierRNN(
            NUM_CLASSES, NUM_STEP, CLASSIFIER_RNN_CKPT,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.drop_rate
        )

    def save(self, ckpt):
        model = {
            # "short_term": self.hidden[0],
            # "long_term": self.hidden[1],
            "state_dict": self.state_dict()
        }
        torch.save(model, ckpt)
        print("Classifier was saved.")

    def load(self, ckpt):
        model = torch.load(ckpt)
        # self.hidden = (model["short_term"], model["long_term"])
        self.load_state_dict(model["state_dict"])
        print("Classifier was loaded.")

    def score(self, logps, y):
        ps = torch.exp(logps)
        _, topk = ps.topk(dim=1)
        equal = topk == y.view(*topk.shape)
        acc = torch.mean(equal.type(torch.FloatTensor))
        return acc

    def forward(self, x):
        self.features.eval()

        n_b = x.size(0)
        n_s = x.size(1)
        n = n_b * n_s

        x = x.view(n, 3, 128, 128)

        # print(self.features(x))
        x = self.features.get_features(x)
        x = x.view(n_b, n_s, -1)

        # print(x.size())

        x, hidden = self.classifier(x, None)
        # self.hidden = (hidden[0].data, hidden[1].data)

        return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            ps = torch.exp(x)
            print(ps)
            cls_ps, top_k = ps.topk(1, dim=1)
            return top_k.squeeze().data

    def score(self, logps, y):
        top_k = logps.topk(1, dim=1)[1]
        equal = top_k == y.view(*top_k.shape)
        score = torch.mean(equal.type(torch.FloatTensor))
        return score

