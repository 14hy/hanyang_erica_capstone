import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading as th

from datautils import trash_data_generator, FMD_data_generate

VM = True


def train_FMD_cnn(drop_rate, gpu=0):
	epochs = 150
	batch_size = 128
	num_classes = 6
	if VM:
		ckpt_file = "ckpts/feature_cnn.ckpt"
	else:
		ckpt_file = "D:/ckpts/capstone/feature_cnn.ckpt"
	
	from features.FeatureCNN import FeatureCNN

	cnn = FeatureCNN(num_classes, ckpt_file, f"/gpu:{gpu}", batch_size=batch_size, eta=1e-3)
	cnn.build((128, 128, 3))

	for e in range(epochs):

		train_loader = iter(FMD_data_generate(VM, batch_size, "train"))
		
		train_loss = 0.0
		cnt = 0

		for X_batch, Y_batch in train_loader:
			cnn.fit(X_batch, Y_batch, drop_rate)
			train_loss += cnn.compute_loss(X_batch, Y_batch)

			cnt += 1

		train_loss /= cnt

		valid_loader = iter(FMD_data_generate(VM, batch_size, "valid"))
		valid_acc = 0.0
		valid_loss = 0.0

		cnt = 0

		for X_batch, Y_batch in valid_loader:
			valid_loss += cnn.compute_loss(X_batch, Y_batch)
			valid_acc += cnn.score(X_batch, Y_batch)

			cnt += 1

		valid_loss /= cnt
		valid_acc /= cnt

		print("=== FMD  ===")
		print("Epochs {}/{}".format(e+1, epochs))
		print("Train loss: {:.6f}".format(train_loss))
		print("Valid loss: {:.6f}".format(valid_loss))
		print("Valid acc: {:.6f}".format(valid_acc))

	cnn.save()

def train_trash_cnn(drop_rate, gpu=0):
	epochs = 150
	batch_size = 128
	num_classes = 4
	drop_rate = 0.5

	if VM:
		ckpt_file = "ckpts/capstone/feature_cnn.ckpt"
	else:
		ckpt_file = "D:/ckpts/capstone/feature_cnn.ckpt"

	from features.FeatureCNN import FeatureCNN

	cnn = FeatureCNN(num_classes, ckpt_file, f"/gpu:{gpu}", batch_size=batch_size, eta=1e-2)
	cnn.build((128, 128, 3))

	for e in range(epochs):
		train_loader = iter(trash_data_generator(VM, batch_size, "train"))

		for X_batch, Y_batch, _ in train_loader:
			cnn.fit(X_batch, Y_batch, drop_rate)

		train_loader = iter(trash_data_generator(VM, batch_size, "train"))
		train_loss = 0.0
		train_acc = 0.0

		cnt = 0
		for X_batch, Y_batch, _ in train_loader:
			train_loss += cnn.compute_loss(X_batch, Y_batch)
			train_acc += cnn.score(X_batch, Y_batch)
			cnt += 1


		train_loss /= cnt
		train_acc /= cnt

		val_loader = iter(trash_data_generator(VM, batch_size, "valid"))
		val_loss = 0.0
		val_acc = 0.0

		cnt = 0
		for X_batch, Y_batch, _ in val_loader:
			val_loss += cnn.compute_loss(X_batch, Y_batch)
			val_acc += cnn.score(X_batch, Y_batch)
			cnt += 1

		val_loss /= cnt
		val_acc /= cnt

		print("=== trash ===")
		print(f"Epoch {e+1}/{epochs}")
		print(f"Train loss: {train_loss:.6f}")
		print(f"Train acc: {train_acc:.6f}")
		print(f"Val loss: {val_loss:.6f}")
		print(f"Val acc: {val_acc:.6f}")

	cnn.save()

if __name__ == "__main__":
	t1 = th.Thread(target=train_FMD_cnn, args=(0.3, 0))
	t2 = th.Thread(target=train_trash_cnn, args=(0.3, 1))

	t1.start()
	t2.start()

	t1.join()
	t2.join()
