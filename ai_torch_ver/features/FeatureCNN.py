import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import models

class FeatureCNN(nn.Module):

	def __init__(self, drop_rate=0.5, num_classes=4):
		super().__init__()

		vgg19 = models.vgg19(pretrained=True).features

		self.features = vgg19
		self.classifier = nn.Sequential(
			nn.Linear(7*7*512, 512),
			nn.ReLU(),
			nn.Dropout(drop_rate),

			nn.Linear(512, 128),
			nn.ReLU(),
			nn.Dropout(drop_rate),

			nn.Linear(128, num_classes),
			nn.LogSoftmax(dim=1)
		)

	def forward(self, x):
		x = self.features(x)
		x = self.classifier(x)

		return x