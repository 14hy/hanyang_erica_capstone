
import sys
import torch
import numpy as np
from torch import nn
from torchvision import transforms
from ai.TrashDetector import TrashDetector
from ai.Classifier import Classifier

THRESHOLD = 0.6


class AI():

    def __init__(self):
        self.classifier = None
        self.trash_detector = None

    def build(self):
        self.classifier = Classifier(num_classes=4, drop_rate=0.5).cuda()
        self.classifier.load("../ai/ckpts/classifier1.pth")
        for param in self.classifier.parameters():
            param.requires_grad_(False)

        self.classifier.eval()

        self.trash_detector = TrashDetector(0.0).cuda()
        self.trash_detector.load("../ai/ckpts/detector.pth")
        for param in self.trash_detector.parameters():
            param.requires_grad_(False)

        self.trash_detector.eval()

    def predict(self, images):
        images = (images.astype(np.float32) - 128) / 256
        print(images.shape, images.min(), images.max())

        torch_images = torch.FloatTensor(images).cuda()
        
        with torch.no_grad():
            preds = self.trash_detector.predict(torch_images)
            print(preds)
            if preds.float().mean() > THRESHOLD:
                print("This is a trash!")
                torch_images = torch_images.unsqueeze(dim=0)
                pred = self.classifier.predict(torch_images)
                return int(pred)
            else:
                print("This is not a trash.")
                return -1

            # torch_images = torch_images.unsqueeze(dim=0)
            # pred = self.classifier.predict(torch_images)
            # return int(pred)
