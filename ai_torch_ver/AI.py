from TrashDetector import TrashDetector
from Classifier import Classifier
import torch
from torch import nn
from torchvision import transforms

THRESHOLD = 0.6


class AI():

    def __init__(self):
        self.classifier = None
        self.trash_detector = None

    def build(self):
        self.classifier = Classifier(num_classes=4, drop_rate=0.5).cuda()
        self.classifier.load("ckpts/classifier.pth")
        for param in self.classifier.parameters():
            param.requires_grad_(False)

        self.classifier.eval()

        self.trash_detector = TrashDetector(drop_rate=0.5).cuda()
        self.trash_detector.module.load("ckpts/detector.pth")
        for param in self.trash_detector.parameters():
            param.requires_grad_(False)

        self.trash_detector.eval()

    def predict(self, images):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        n = len(images)
        torch_images = []
        for i in range(n):
            img = images[i]
            torch_images.append(transform(img).unsqueeze(dim=0))

        torch_images = torch.stack(torch_images, dim=0).cuda()
        
        with torch.no_grad():
            preds = self.trash_detector.predict(torch_images)
            if self.is_trash(preds):
                print("This is a trash!")
                torch_images = torch_images.unsqueeze(dim=0)
                pred = self.classifier.predict(torch_images)
                return int(pred)
            else:
                print("This is not a trash.")
                return 0

    def is_trash(self, preds):
        if torch.mean(preds) >= THRESHOLD:
            return True
        else:
            return False
