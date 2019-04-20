import torch
from torch import nn
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import cv2
try:
    from ai_torch_ver.prepare_data import get_image_loader
except:
    from prepare_data import get_image_loader

DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/detector"


class TrashDetector(nn.Module):

    def __init__(self):
        super(TrashDetector, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 16, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), stride=2, padding=0),

            nn.Conv2d(16, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), stride=1, padding=0),

            nn.Conv2d(32, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, (5, 5), stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d((2, 2), stride=1, padding=0), # 16 16 64
        )
        
        nothing_path = DATA_PATH + "/nothing"
        trash_path = DATA_PATH + "/trash"

        self.nothing = []
        self.trash = []

        for f in pathlib.Path(nothing_path).glob("*.jpg"):
            self.nothing.append(f)
        for f in pathlib.Path(trash_path).glob("*.jpg"):
            self.trash.append(f)
        
    def forward(self, x):
        source = x[:, 0]
        positive = x[:, 1]
        negative = x[:, 2]

        source = self.features(source)
        positive = self.features(positive)
        negative = self.features(negative)

        pos = torch.mean((source - positive)**2)
        neg = torch.mean((source - negative)**2)

        return pos, neg

    def save(self, ckpt):
        torch.save(self.state_dict(), ckpt)
        print("TrashDetector was saved.")

    def load(self, ckpt):
        self.load_state_dict(torch.load(ckpt))
        print("TrashDetector was loaded.")

    def predict(self, x):
        np.random.shuffle(self.nothing)
        np.random.shuffle(self.trash)

        device = torch.device("cuda")

        nothing = self.nothing[:8]
        trash = self.trash[:8]

        no_imgs = []
        tr_imgs = []

        for no, tr in zip(nothing, trash):
            no_img = cv2.resize(plt.imread(no), dsize=(128, 128)).astype(np.float32) / 255
            tr_img = cv2.resize(plt.imread(tr), dsize=(128, 128)).astype(np.float32) / 255

            no_imgs.append(torch.FloatTensor(no_img).permute(2, 0, 1).view(1, 3, 128, 128))
            tr_imgs.append(torch.FloatTensor(tr_img).permute(2, 0, 1).view(1, 3, 128, 128))

        nothing = torch.cat(no_imgs, dim=0)
        trash = torch.cat(tr_imgs, dim=0)

        x_fts = self.features(x.to(device))
        no_fts = self.features(nothing.to(device))
        tr_fts = self.features(trash.to(device))

        pos = torch.mean((x_fts - tr_fts)**2)
        neg = torch.mean((x_fts - no_fts)**2)

        if pos > neg:
            return 1
        else:
            return 0