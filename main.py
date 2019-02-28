import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2
import sys

sys.path.append("./features")

from StackedEncoder import StackedEncoder


def load_DTD_dataset(samples=5640, label=True):
	data_path = "D:/datasets/dtd/images"
	num_classes = 47

	print("Data loading...")

	images = np.zeros((samples, 128, 128, 3))

	if label:
		labels = np.zeros((samples, num_classes))

	if label:
		label_dict = {}
		label_index = 0
	index = 0

	for d in pathlib.Path(data_path).glob("*"):
		print(d)

		if label:
			label_dict[d] = label_index

		for f in d.glob("*.jpg"):
			img = plt.imread(f).astype(np.float32) / 255
			img = cv2.resize(img, dsize=(128, 128))

			images[index] = img
			if label:
				labels[index, label_dict[d]] = 1

			index += 1

		if label:
			label_index += 1

	# index = 0
	# for f in pathlib.Path(data_path).glob("*/*.jpg"):
	# 	img = plt.imread(f).astype(np.float32) / 255
	# 	img = cv2.resize(img, dsize=(128, 128))

	# 	images[index] = img

	# 	index += 1

	print("{} Data loaded.".format(index))

	if label:
		return images, labels
	return images

def load_trash_dataset(label=False):
	data_path = "D:/datasets/trashdata"
	num_classes = 6

	images = np.zeros((2527, 128, 128, 3))
	if label:
		labels = np.zeros((2527, num_classes))

	if label:
		label_index = 0
	index = 0
	for d in pathlib.Path(data_path).glob("*"):
		if d.name == ".DS_Store":
			continue
		print(d)
		for f in d.glob("*.jpg"):
			img = plt.imread(f).astype(np.float32) / 255
			img = cv2.resize(img, dsize=(128, 128))

			if len(img.shape) == 2:
				images[index, :, :, 0] = img
				images[index, :, :, 1] = img
				images[index, :, :, 2] = img
			else:
				images[index] = img

			if label:
				labels[index, label_index] = 1

			index += 1

		if label:
			label_index += 1

	print("Data {} were loaded.".format(index))

	if label:
		return images, labels
	else:
		return images

def load_trash_dataset2():
	data_path = "D:/Users/jylee/Dropbox/Files/Datasets/trashdata2"
	num_classes = 6

	images = np.zeros((5796, 128, 128, 3))

	index = 0
	for d in pathlib.Path(data_path).glob("*"):
		if d.name == ".DS_Store":
			continue
		print(d)
		for f in d.glob("*.jpg"):
			img = plt.imread(f).astype(np.float32) / 255
			img = cv2.resize(img, dsize=(128, 128))

			if len(img.shape) == 2:
				images[index, :, :, 0] = img
				images[index, :, :, 1] = img
				images[index, :, :, 2] = img
			else:
				images[index] = img

			index += 1


	print("Data {} were loaded.".format(index))

	return images

def load_FMD_dataset(samples=11000, label=True):
	data_path = "D:/datasets/FMD/image"
	num_classes = 10

	print("Data loading...")

	images = np.zeros((samples, 128, 128, 3))

	if label:
		labels = np.zeros((samples, num_classes))
		label_dict = {}
		label_index = 0
	index = 0

	for d in pathlib.Path(data_path).iterdir():
		print(d)

		if label:
			label_dict[d] = label_index

		for f in d.glob("*.jpg"):
			img = plt.imread(f).astype(np.float32) / 255
			img = cv2.resize(img, dsize=(128, 128))

			if len(img.shape) == 2:
				images[index, :, :, 0] = img
				images[index, :, :, 1] = img
				images[index, :, :, 2] = img
			else:
				images[index] = img
			if label:
				labels[index, label_dict[d]] = 1

			index += 1

		for f in d.glob("output/*.jpg"):
			img = plt.imread(f).astype(np.float32) / 255
			img = cv2.resize(img, dsize=(128, 128))

			if len(img.shape) == 2:
				images[index, :, :, 0] = img
				images[index, :, :, 1] = img
				images[index, :, :, 2] = img
			else:
				images[index] = img
			if label:
				labels[index, label_dict[d]] = 1

			index += 1

		if label:
			label_index += 1

	# index = 0
	# for f in pathlib.Path(data_path).glob("*/*.jpg"):
	# 	img = plt.imread(f).astype(np.float32) / 255
	# 	img = cv2.resize(img, dsize=(128, 128))

	# 	images[index] = img

	# 	index += 1

	print("{} Data loaded.".format(index))

	if label:
		return images, labels
	return images

def rnn_data(images, labels, num_samples, num_step, num_classes):
	images_rnn = np.zeros((num_samples, num_step, 128, 128, 3))
	labels_rnn = np.zeros((num_samples, num_classes))

	n = num_samples // num_classes

	for c in range(num_classes):
		n_c = images[np.argmax(labels, axis=1).reshape(-1) == c].shape[0]
		r = np.random.choice(np.arange(n_c), size=(num_step * n,))

		images_rnn[c * n : (c + 1) * n] = images[r].reshape(n, num_step, 128, 128, 3)
		labels_rnn[c * n : (c + 1) * n, c] = 1

	# print(labels_rnn)

	return images_rnn, labels_rnn

def train_FMD_encoder():
	images = load_FMD_dataset(label=False)
	# images = load_trash_dataset()

	from StackedEncoder import StackedEncoder

	# encoder = StackedEncoder("encoder_FMD", "/gpu:0", eta=[1e-5, 1e-5, 1e-5])
	encoder = StackedEncoder("encoder_trash", "/gpu:0", eta=[1e-5, 1e-5, 1e-5])
	encoder.build((128, 128, 3), load_weights=False)
	encoder.fit(images, [0, 1, 2], epochs=[100, 150, 250])
	# encoder.encode(images[0].reshape(1, 128, 128, 3))


def train_FMD_cnn():
	images, labels = load_FMD_dataset(label=True)
	num_classes = labels.shape[1]

	for i in range(10):
		plt.imshow(images[i])
		plt.show()

	# from FeatureCNN import FeatureCNN

	# cnn = FeatureCNN(num_classes, "feature_cnn", device="/gpu:0", eta=1e-3)
	# cnn.build((128, 128, 3), load_weights=False)
	# cnn.fit(images, labels, 0.75, epochs=50)

	# print(cnn.score(images, labels))
	# print(cnn.transform(images))


def train_trash_cnn():
	images, labels = load_trash_dataset(label=True)
	num_classes = labels.shape[1]

	from FeatureCNN import FeatureCNN

	cnn = FeatureCNN(num_classes, "feature_cnn", device="/gpu:0", eta=1e-3)
	cnn.build((128, 128, 3), load_weights=False)
	cnn.fit(images, labels, 0.7, epochs=50)

	print(cnn.score(images, labels))
	# print(cnn.transform(images))

def validate_encoder():
	images = load_trash_dataset2()

	from Encoder import Encoder

	encoder = Encoder("ckpts/encoder_trash1.ckpt", "/gpu:0")
	encoder.build((128, 128, 3), load_weights=True)

	# encoder.encode(images[0].reshape(1, 128, 128, 3))

	for i in np.random.choice(np.arange(images.shape[0]), replace=False, size=(10,)):
		plt.imshow(images[i])
		plt.show()

		decoded = encoder.predict(images[i].reshape(1, 128, 128, 3)).reshape(128, 128, 3)

		plt.imshow(decoded)
		plt.show()

def train_classifier():
	images, labels = load_trash_dataset(label=True)

	num_step = 8
	num_classes = labels.shape[1]

	images_rnn, labels_rnn = rnn_data(images, labels, 4000, num_step, num_classes)

	from Classifier import Classifier

	clf = Classifier(num_step, num_classes, eta=1e-3)
	clf.build(num_gpu=2)
	clf.fit(images_rnn, labels_rnn, epochs=100)

	print(clf.score(images_rnn, labels_rnn))

# train_texture_encoder()
# validate_encoder()
# train_trash_cnn()
# train_FMD_encoder()
train_classifier()
