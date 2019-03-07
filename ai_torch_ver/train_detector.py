import torch
from torch import optim, nn
import numpy as np

from TrashDetector import TrashDetector
from prepare_data import image_loader

VM = False
DATA_PATH = "D:/Users/jylee/Dropbox/Files/Datasets/nothing_or_trash"
ETA = 1e-3
BATCH_SIZE = 128
EPOCHS = 50
DROP_RATE = 0.4

def score(logps, labels):
	ps = torch.exp(logps)
	cls_ps, top_k = ps.topk(1, dim=1)
	equal = top_k == labels.view(*top_k.shape)
	acc = torch.mean(equal.type(torch.FloatTensor))
	return acc

def train_detector():
	
	device = torch.device("cuda:0")
	cnn = nn.DataParallel(TrashDetector(DROP_RATE)).to(device)
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(cnn.parameters(), lr=ETA)

	train_loader, valid_loader, test_loader = image_loader(DATA_PATH, BATCH_SIZE)

	val_losses = []

	for e in range(EPOCHS):
		
		train_loss = 0.0
		train_acc = 0.0

		for x_batch, y_batch in train_loader:
			x_batch = x_batch.to(device)
			y_batch = y_batch.to(device)

			logps = cnn(x_batch)
			loss = criterion(logps, y_batch)

			train_loss += loss.item()
			train_acc += score(logps, y_batch).item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			cnn.eval()

			val_loss = 0.0
			val_acc = 0.0

			for x_batch, y_batch in valid_loader:
				x_batch = x_batch.to(device)
				y_batch = y_batch.to(device)

				logps = cnn(x_batch)
				loss = criterion(logps, y_batch)

				val_loss += loss.item()
				val_acc += score(logps, y_batch)

			train_loss /= len(train_loader)
			train_acc /= len(train_loader)
			val_loss /= len(valid_loader)
			val_acc /= len(valid_loader)

			print(f"Epochs {e+1}/{EPOCHS}")
			print(f"Train loss: {train_loss:.6f}")
			print(f"Train acc: {train_acc:.6f}")
			print(f"Valid loss: {val_loss:.6f}")
			print(f"Valid acc: {val_acc:.6f}")

			val_losses.append(val_loss)

			if np.min(val_losses) > val_loss:
				torch.save(cnn.state_dict(), "D:/ckpts/capstone/torch/trash_detector.pth")

			cnn.train()


if __name__ == "__main__":
	train_detector()