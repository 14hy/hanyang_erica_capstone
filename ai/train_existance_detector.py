import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2

sys.path.append(".")

from ExistanceDetector import ExistanceDetector

DATA_DIR = "D:/Users/jylee/Dropbox/Files/datasets/nothing_or_trash"

BATCH_SIZE = 128
IMAGE_SIZE = 128

def next_train_data(data_dir):
	NOTHINC_DATA_DIR = "nothing"
	TRASH_DATA_DIR = "trash"
	
	neg_files = []
	pos_files = []

	neg_data_dir = os.path.join(data_dir, NOTHINC_DATA_DIR, "train").replace("\\", "/")

	for f in pathlib.Path(neg_data_dir).glob("*.jpg"):
		neg_files.append(f)

	pos_data_dir = os.path.join(data_dir, TRASH_DATA_DIR, "train").replace("\\", "/")

	for f in pathlib.Path(pos_data_dir).glob("*.jpg"):
		pos_files.append(f)

	num_batch = int(np.ceil(min(len(neg_files), len(pos_files)) / BATCH_SIZE))

	np.random.shuffle(pos_files)
	np.random.shuffle(neg_files)

	for b in range(num_batch):
		start = b*BATCH_SIZE//2
		end = min((b+1)*BATCH_SIZE//2, len(pos_files))

		X_batch = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
		Y_batch = np.zeros((BATCH_SIZE, 1))
		
		for i in range(start, end):
			pos_img = cv2.resize(plt.imread(pos_files[i]), dsize=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255
			neg_img = cv2.resize(plt.imread(neg_files[i]), dsize=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255

			X_batch[i - start] = pos_img
			X_batch[i - start + BATCH_SIZE//2] = neg_img

			Y_batch[i - start, 0] = 1
			Y_batch[i - start + BATCH_SIZE//2, 0] = 0

		yield X_batch, Y_batch, num_batch


def next_valid_data(data_dir):
	NOTHINC_DATA_DIR = "nothing"
	TRASH_DATA_DIR = "trash"
	
	neg_files = []
	pos_files = []

	neg_data_dir = os.path.join(data_dir, NOTHINC_DATA_DIR, "valid").replace("\\", "/")

	for f in pathlib.Path(neg_data_dir).glob("*.jpg"):
		neg_files.append(f)

	pos_data_dir = os.path.join(data_dir, TRASH_DATA_DIR, "valid").replace("\\", "/")

	for f in pathlib.Path(pos_data_dir).glob("*.jpg"):
		pos_files.append(f)

	num_batch = int(np.ceil(min(len(neg_files), len(pos_files)) / BATCH_SIZE))

	for b in range(num_batch):
		start = b*BATCH_SIZE//2
		end = min((b+1)*BATCH_SIZE//2, len(pos_files))

		X_batch = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
		Y_batch = np.zeros((BATCH_SIZE, 1))
		
		for i in range(start, end):
			pos_img = cv2.resize(plt.imread(pos_files[i]), dsize=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255
			neg_img = cv2.resize(plt.imread(neg_files[i]), dsize=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255

			X_batch[i - start] = pos_img
			X_batch[i - start + BATCH_SIZE//2] = neg_img

			Y_batch[i - start, 0] = 1
			Y_batch[i - start + BATCH_SIZE//2, 0] = 0

		yield X_batch, Y_batch, num_batch


def next_test_data(data_dir):
	NOTHINC_DATA_DIR = "nothing"
	TRASH_DATA_DIR = "trash"
	
	neg_files = []
	pos_files = []

	neg_data_dir = os.path.join(data_dir, NOTHINC_DATA_DIR, "test").replace("\\", "/")

	for f in pathlib.Path(neg_data_dir).glob("*.jpg"):
		neg_files.append(f)

	pos_data_dir = os.path.join(data_dir, TRASH_DATA_DIR, "test").replace("\\", "/")

	for f in pathlib.Path(pos_data_dir).glob("*.jpg"):
		pos_files.append(f)

	num_batch = int(np.ceil(min(len(neg_files), len(pos_files)) / BATCH_SIZE))

	for b in range(num_batch):
		start = b*BATCH_SIZE//2
		end = min((b+1)*BATCH_SIZE//2, len(pos_files))

		X_batch = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
		Y_batch = np.zeros((BATCH_SIZE, 1))
		
		for i in range(start, end):
			pos_img = cv2.resize(plt.imread(pos_files[i]), dsize=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255
			neg_img = cv2.resize(plt.imread(neg_files[i]), dsize=(IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255

			X_batch[i - start] = pos_img
			X_batch[i - start + BATCH_SIZE//2] = neg_img

			Y_batch[i - start, 0] = 1
			Y_batch[i - start + BATCH_SIZE//2, 0] = 0

		yield X_batch, Y_batch, num_batch

def train_detector():
	epochs = 10
	keep_prob = 0.7

	detector = ExistanceDetector("/gpu:0", "D:/ckpts/capstone/detector.ckpt", eta=1e-3)
	detector.build()


	for e in range(epochs):
		train_loss = 0.0
		train_loader = iter(next_train_data(DATA_DIR))

		for X_batch, Y_batch, num_train_batch in train_loader:
			detector.step(X_batch, Y_batch, keep_prob)

			train_loss += detector.compute_loss(X_batch, Y_batch) / num_train_batch

		valid_loader = iter(next_valid_data(DATA_DIR))
		val_loss = 0.0
		val_acc = 0.0

		for X_valid, Y_valid, num_valid_batch in valid_loader:
			loss = detector.compute_loss(X_valid, Y_valid)
			acc = detector.score(X_valid, Y_valid)

			val_loss += loss / num_valid_batch
			val_acc += acc / num_valid_batch

		print("================================")
		print("Epochs {}/{}".format(e+1, epochs))
		print("train_loss: {:.6f}".format(train_loss))
		print("val_loss: {:.6f}".format(val_loss))
		print("val_acc: {:.6f}".format(val_acc))

	detector.save()

def test_detector():
	detector = ExistanceDetector("/gpu:0", "D:/ckpts/capstone/detector.ckpt")
	detector.build()
	detector.load_weights()

	test_loader = iter(next_test_data(DATA_DIR))

	loss = 0.0
	acc = 0.0

	for X_batch, Y_batch, num_test_batch in test_loader:
		l = detector.compute_loss(X_batch, Y_batch)
		score = detector.score(X_batch, Y_batch)

		loss += l/num_test_batch
		acc += score/num_test_batch

	print("Test loss: {:.6f}".format(loss))
	print("Test acc: {:.6f}".format(acc))


def test():
	train_loader = iter(next_train_data(DATA_DIR))
	valid_loader = iter(next_valid_data(DATA_DIR))

	for i in range(10):
		imgs, lbls, _ = next(valid_loader)
		r = np.random.randint(BATCH_SIZE)
		plt.imshow(imgs[r])
		print(lbls[r])
		plt.show()

def test2():
	valid_loader = iter(next_valid_data(DATA_DIR))

	detector = ExistanceDetector("/gpu:0", "D:/ckpts/capstone/detector.ckpt")
	detector.build()
	detector.load_weights()

	imgs, lbls, _ = next(valid_loader)
	print(detector.predict(imgs[0].reshape(1, *imgs[0].shape)))

# test()
test2()
# train_detector()
# test_detector()