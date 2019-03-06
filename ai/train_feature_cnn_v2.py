
import numpy as np
import matplotlib.pyplot as plt

from features.FeatureCNNv2 import FeatureCNN
from datautils import trash_data_generator

VM = True
BATCH_SIZE = 128
KEEP_PROB = 0.4
EPOCHS = 50
ETA = 1e-3
NUM_CLASSES = 4

if VM:
	CKPT = "ckpts/capstone/feature_cnn.ckpt"
else:
	CKPT = "D:/ckpts/capstone/feature_cnn.ckpt"

def train_feature_cnn():

	cnn = FeatureCNN(ETA, CKPT, NUM_CLASSES, "/gpu:0")
	cnn.build()

	for e in range(EPOCHS):
		train_loader = iter(trash_data_generator(VM, BATCH_SIZE, "train"))

		# train
		for X_batch, Y_batch, _ in train_loader:
			cnn.step(X_batch, Y_batch, KEEP_PROB)

		# compute train loss ans accuracy
		train_loss = 0.0
		train_acc = 0.0

		train_loader = iter(trash_data_generator(VM, BATCH_SIZE, "train"))

		cnt = 0
		for X_batch, Y_batch, _ in train_loader:
			train_loss += cnn.compute_loss(X_batch, Y_batch)
			train_acc += cnn.score(X_batch, Y_batch)
			cnt += 1

		train_loss /= cnt
		train_acc /= cnt

		# compute valid loss and accuracy
		val_loss = 0.0
		val_acc = 0.0

		valid_loader = iter(trash_data_generator(VM, BATCH_SIZE, "valid"))

		cnt = 0

		for X_batch, Y_batch, _ in valid_loader:
			val_loss += cnn.compute_loss(X_batch, Y_batch)
			val_acc += cnn.score(X_batch, Y_batch)
			cnt += 1

		val_loss /= cnt
		val_acc /= cnt

		print(f"Epochs {e+1}/{EPOCHS}")
		print(f"Train loss: {train_loss:.6f}")
		print(f"Train acc: {train_acc:.6f}")
		print(f"Valid loss: {val_loss:.6f}")
		print(f"Valid acc: {val_acc:.6f}")

	cnn.save()

if __name__ == "__main__":
	train_feature_cnn()
