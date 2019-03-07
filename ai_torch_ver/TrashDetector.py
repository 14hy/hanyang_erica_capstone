import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class TrashDetector(nn.Module):

	def __init__(self, drop_rate):
		super().__init__()

		self.features = nn.Sequential(
			nn.Conv2d(3, 32, (5, 5), stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d((2, 2), stride=2, padding=0), # 64

			nn.Conv2d(32, 32, (5, 5), stride=1, padding=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.MaxPool2d((2, 2), stride=2, padding=0), # 32

			nn.Conv2d(32, 64, (5, 5), stride=1, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d((2, 2), stride=2, padding=0), # 16

			nn.Conv2d(64, 64, (5, 5), stride=1, padding=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.MaxPool2d((2, 2), stride=2, padding=0) # 8
		)

		self.classifier = nn.Sequential(
			nn.Linear(8*8*64, 256),
			nn.ReLU(),
			nn.Dropout(drop_rate),

			nn.Linear(256, 64),
			nn.ReLU(),
			nn.Dropout(drop_rate),

			nn.Linear(64, 2),
			nn.LogSoftmax(dim=1)
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 8*8*64)
		x = self.classifier(x)

		return x