import torch
from torch import nn
from torchvision.models import vgg16_bn
import numpy as np
import sys
from collections import Counter


NUM_CLASSES = 4
NUM_STEP = 8


class ClassifierVGG16(nn.Module):

    def __init__(self):
        super().__init__()

        num_classes = 4
        drop_rate = 0.5
        num_step = 8
        input_size = 4*4*512
        hidden_size = 128
        num_layers = 2

        vgg = vgg16_bn(pretrained=True)

        device = torch.device("cuda:0")

        if torch.cuda.device_count() > 1:
            self.features = nn.DataParallel(vgg.features).to(device)
        else:
            self.features = vgg.features.to(device)

        for param in self.features.parameters():
            param.requires_grad_(False)
        self.features.eval()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=drop_rate if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_step*2*hidden_size, 128),
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),

            nn.Linear(128, 4),
            nn.LogSoftmax(dim=1)
        )

    def score(self, logps, y):
        ps = torch.exp(logps)
        _, topk = ps.topk(1, dim=1)
        equal = topk == y.view(*topk.shape)
        acc = torch.mean(equal.type(torch.FloatTensor))
        return acc

    def forward(self, x):
        self.features.eval()

        n_b = x.size(0)
        n_s = x.size(1)
        n = n_b * n_s

        x = x.view(n, 3, 128, 128)

        x = self.features(x)
        x = x.view(n_b, n_s, -1)
        x, _ = self.lstm(x, None)
        x = x.contiguous().view(n_b, -1)
        x = self.classifier(x)

        return x

    def save(self, ckpt):
        model = {
            "state_dict": self.state_dict()
        }
        torch.save(model, ckpt)
        print("Classifier was saved.")

    def load(self, ckpt):
        model = torch.load(ckpt)
        self.load_state_dict(model["state_dict"])
        print("Classifier was loaded.")

    def predict(self, x):
        with torch.no_grad():
            logps = self.forward(x)
            ps = torch.exp(logps)
            print(ps)
            cls_ps, top_k = ps.topk(1, dim=1)
            return top_k

