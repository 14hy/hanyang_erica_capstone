import torch
from torch import nn
import numpy as np
import sys
from ai_torch_ver.features.FeatureCNNv2 import FeatureCNN
from ai_torch_ver.ClassifierRNN import ClassifierRNN

# FEATURE_CNN_CKPT = "D:/ckpts/capstone/torch/feature_cnn.pth"
FEATURE_CNN_CKPT = "../ai_torch_ver/ckpts/feature_cnn.pth"
# CLASSIFIER_RNN_CKPT = "D:/ckpts/capstone/torch/classifier_rnn.pth"
CLASSIFIER_RNN_CKPT = "../ai_torch_ver/ckpts/classifier_rnn.pth"
NUM_CLASSES = 4
NUM_STEP = 8


class Classifier(nn.Module):

    def __init__(self, num_classes, drop_rate=0.5):
        super().__init__()

        self.num_classes = num_classes

        self.input_size = 256
        self.hidden_size = 64
        self.num_layers = 2
        self.drop_rate = drop_rate
        self.hidden = (
            torch.randn(self.num_layers, 1, self.hidden_size).data,
            torch.randn(self.num_layers, 1, self.hidden_size).data
        )

        cnn = FeatureCNN(self.num_classes, 0.5)
        cnn.load(FEATURE_CNN_CKPT)
        
        if torch.cuda.device_count() > 1:
            self.features = nn.DataParallel(FeatureMap(cnn.conv1, cnn.conv2, cnn.features))
        else:
            self.features = FeatureMap(cnn.conv1, cnn.conv2, cnn.features)

        # for param in pcnn.parameters():
        #     param.requires_grad_(False)
        # self.load_feature_cnn(pcnn)

        # for m in pcnn.modules():
        #     if type(m) is FeatureCNN:
        #         self.features = m
        #         break

        self.classifier = ClassifierRNN(
            NUM_CLASSES, NUM_STEP, CLASSIFIER_RNN_CKPT,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.drop_rate
        )

    # def load_feature_cnn(self, cnn):
    #     state_dict = torch.load(FEATURE_CNN_CKPT)
    #     cnn.load_state_dict(state_dict)

    # def transform(self, x):
    #     x1 = self.features.conv1(x)
    #     x2 = self.features.conv2(x)

    #     x1 = x1.view(-1, 8*8*64)
    #     x2 = x2.view(-1, 8*8*64)

    #     x = torch.cat([x1, x2], dim=1)
    #     x = self.features.features(x)
    #     return x

    def save(self, ckpt):
        model = {
            "short_term": self.hidden[0],
            "long_term": self.hidden[1],
            "state_dict": self.state_dict()
        }
        torch.save(model, ckpt)
        print("Classifier was saved.")

    def load(self, ckpt):
        model = torch.load(ckpt)
        self.hidden = (model["short_term"], model["long_term"])
        self.load_state_dict(model["state_dict"])
        print("Classifier was loaded.")

    def score(self, logps, y):
        ps = torch.exp(logps)
        _, topk = ps.topk(dim=1)
        equal = topk == y.view(*topk.shape)
        acc = torch.mean(equal.type(torch.FloatTensor))
        return acc

    def forward(self, x):
        # if rearange:
        #     x = self.rearange_image(x)

        # new_x = []

        # for b in range(x.shape[0]):
        #     _x = x[b].view(NUM_STEP, 3, 128, 128)
        #     _x = self.transform(_x)
        #     _x = _x.view(-1, NUM_STEP, 256)
        #     _x, hidden = self.classifier(_x.cuda(0), self.hidden)
        #     new_x.append(_x)

        #     self.hidden = (hidden[0].data, hidden[1].data)

        # x = torch.cat(new_x, dim=0)

        # return x

        # if rearange:
        #     x = self.rearange_image(x)

        n_b = x.size(0)
        n_s = x.size(1)
        n = n_b * n_s

        x = x.view(n, 3, 128, 128)

        x = self.features(x)
        x = x.view(n_b, n_s, -1)

        x, hidden = self.classifier(x, self.hidden)
        self.hidden = (hidden[0].data, hidden[1].data)

        return x

    # def rearange_image(self, x):
    #     x = x.view(-1, NUM_STEP, 128, 128, 3)
    #     x = x.permute(4, 2, 3)
    #     return x

    def predict(self, x):
        with torch.no_grad():
            x = self.forward(x)
            cls_ps, top_k = x.topk(1, dim=1)
            return top_k.squeeze().data

    def score(self, logps, y):
        top_k = logps.topk(1, dim=1)[1]
        equal = top_k == y.view(*top_k.shape)
        score = torch.mean(equal.type(torch.FloatTensor))
        return score


class FeatureMap(nn.Module):

    def __init__(self, conv1, conv2, features):
        super().__init__()

        self.conv1 = conv1
        self.conv2 = conv2
        self.features = features

    def forward(self, x):
        n = x.size(0)

        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x1 = x1.view(n, -1)
        x2 = x2.view(n, -1)

        x = torch.cat([x1, x2], dim=1)
        x = self.features(x)

        return x