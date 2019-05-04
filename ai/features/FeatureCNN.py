import torch
import torch.nn.functional as F
from torch import nn


class FeatureCNN(nn.Module):

    def __init__(self, num_classes, drop_rate, load=True):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), stride=1, padding=1), # 128
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.MaxPool2d((2, 2), stride=2, padding=0), # 64

            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), # 64
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.MaxPool2d((2, 2), stride=2, padding=0), # 32

            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), # 32
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.MaxPool2d((2, 2), stride=2, padding=0), # 16

            nn.Conv2d(32, 32, (3, 3), stride=1, padding=1), # 16
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.MaxPool2d((2, 2), stride=2, padding=0) # 8
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 32, (7, 7), stride=1, padding=3), # 128
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.MaxPool2d((2, 2), stride=2, padding=0), # 64

            nn.Conv2d(32, 32, (7, 7), stride=1, padding=3), # 64
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.MaxPool2d((2, 2), stride=2, padding=0), # 32

            nn.Conv2d(32, 32, (7, 7), stride=1, padding=3), # 32
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.MaxPool2d((2, 2), stride=2, padding=0), # 16

            nn.Conv2d(32, 32, (7, 7), stride=1, padding=3), # 16
            nn.BatchNorm2d(32),
            nn.Tanh(),
            # nn.Dropout(drop_rate),
            nn.MaxPool2d((2, 2), stride=2, padding=0) # 8
        )

        self.features = nn.Sequential(
            # nn.Linear(2*8*8*64, 256),
            nn.Linear(2*8*8*32, 256),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),

            nn.Linear(256, 32),
            nn.Tanh(),
            nn.Dropout(drop_rate),

            nn.Linear(32, num_classes),
            nn.LogSoftmax(dim=1)
        )

    # def transform(self, x):
    # 	x1 = self.conv1(x)
    # 	x2 = self.conv2(x)

    # 	x1 = x1.view(-1, 8*8*64)
    # 	x2 = x2.view(-1, 8*8*64)

    # 	x = torch.cat([x1, x2], dim=1)
    # 	x = self.features(x)

    # 	return x

    def save(self, ckpt):
        torch.save(self.state_dict(), ckpt)
        print("Feature CNN was saved.")

    def load(self, ckpt):
        self.load_state_dict(torch.load(ckpt))
        print("Feature CNN was loaded.")

    def forward(self, x):
        x = self.get_features(x)
        x = self.classifier(x)

        return x

    def get_features(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x1 = x1.view(-1, 8*8*32)
        x2 = x2.view(-1, 8*8*32)

        x = torch.cat([x1, x2], dim=1)
        x = self.features(x)

        return x

    def predict(self, x):
        logps = self.forward(x)
        ps = torch.exp(x)
        _, topk = ps.topk(1, dim=1)

        return topk.cpu().numpy().squeeze()
        
