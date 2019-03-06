import numpy as np
import matplotlib.pyplot as plt
import pathlib
import cv2
import sys

sys.path.append("./features")

from StackedEncoder import StackedEncoder

VM = True

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

def trash_data_generator(batch_size, dataset_type="train"):
	if dataset_type == "train":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/train"
		if VM:
			DATA_DIR = "/home/jylee/datasets/capstonedata/train"
	elif dataset_type == "valid":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/valid"
		if VM:
			DATA_DIR = "/home/jylee/datasets/capstonedata/valid"
	elif dataset_type == "test":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/capstonedata/test"
		if VM:
			DATA_DIR = "/home/jylee/datasets/capstonedata/test"
	else:
		print("Invalid dataset type")
		return

	label_dict = {
		"can": 0,
		"extra": 1,
		"glass": 2,
		"plastic": 3
	}

	data = []

	for d in pathlib.Path(DATA_DIR).glob("*"):
		# print(d)
		if d.name in label_dict.keys():
			label = label_dict[d.name]

			for f in d.glob("*.jpg"):
				data.append((f, label))

	# np.random.shuffle(data)
	num_batch = int(np.ceil(len(data) / batch_size))

	for b in range(num_batch):
		start = b * batch_size
		end = min((b+1) * batch_size, len(data))

		X_batch = np.zeros((end - start, 128, 128, 3))
		Y_batch = np.zeros((end - start, len(label_dict)))

		for i in range(start, end):
			img = cv2.resize(plt.imread(data[i][0]), dsize=(128, 128)).astype(np.float32) / 255
			lbl = int(data[i][1])

			X_batch[i - start] = img
			Y_batch[i - start, lbl] = 1

		yield X_batch, Y_batch, num_batch

def rnn_trash_data_generator(num_sample, num_step, dataset_type="train"):
	BATCH_SIZE = 512

	assert(num_sample % 4 == 0)
	
	loader = iter(trash_data_generator(BATCH_SIZE, dataset_type))

	for X_batch, Y_batch, num_batch in loader:

		for i in range(4):

			X_batch_category = X_batch[(Y_batch == i).squeeze()]
			if len(X_batch_category) < 32:
				continue

			# print(X_batch_category.shape)

			X_batch_rnn = np.zeros((num_sample, num_step, 128, 128, 3))
			Y_batch_rnn = np.zeros((num_sample, 4))

			r = np.random.randint(0, len(X_batch_category), size=(num_step * (num_sample)))

			X_batch_rnn[:] = X_batch_category[r].reshape(num_sample, num_step, 128, 128, 3)
			Y_batch_rnn[:, i] = 1

			yield X_batch_rnn, Y_batch_rnn, num_batch*4

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

def FMD_data_generate_no_label(batch_size):
	if VM:
		DATA_DIR = "/home/jylee/datasets/FMD/image"
	else:
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/FMD/image"

	data = []

	for f in pathlib.Path(DATA_DIR).glob("*/*/*.jpg"):
		data.append(f)

	num_batch = int(np.ceil(len(data) / batch_size))
	np.random.shuffle(data)

	for b in range(num_batch):
		start = b * batch_size
		end = min((b+1) * batch_size, len(data))

		X_batch = np.zeros((end - start, 128, 128, 3))

		for i in range(start, end):
			img = cv2.resize(plt.imread(data[i]), dsize=(128, 128)).astype(np.float32) / 255

			if len(img.shape) == 2:
				X_batch[i - start, :, :, 0] = img
				X_batch[i - start, :, :, 1] = img
				X_batch[i - start, :, :, 2] = img
			else:
				X_batch[i - start] = img

		yield X_batch

def train_FMD_encoder():
	batch_size = 128
	epochs = 100
	
	from StackedEncoder import StackedEncoder

	if VM:
		encoder = StackedEncoder("ckpts/capstone/encoder_FMD", "/gpu:0", eta=[1e-3, 3e-3, 5e-3])
	else:
		encoder = StackedEncoder("D:/ckpts/capstone/encoder_FMD", "/gpu:0", eta=[1e-3, 3e-3, 5e-3])
		
	encoder.build((128, 128, 3), load_weights=False)
	encoder.fit(FMD_data_generate_no_label, index=[0, 1, 2], epochs=[50, 75, 100])
	
def train_trash_encoder():
	batch_size = 128
	epochs = 100
	
	from StackedEncoder import StackedEncoder

	if VM:
		encoder = StackedEncoder("ckpts/capstone/encoder_trash", "/gpu:0", eta=[1e-3, 3e-3, 5e-3])
	else:
		encoder = StackedEncoder("D:/ckpts/capstone/encoder_trash", "/gpu:0", eta=[1e-3, 3e-3, 5e-3])
		
	encoder.build((128, 128, 3), load_weights=False)
	encoder.fit(trash_data_generator, index=[0, 1, 2], epochs=[50, 75, 100])

def FMD_data_generate(batch_size, dataset_type="train"):
	if dataset_type == "train":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/FMD/image/train"
		if VM:
			DATA_DIR = "/home/jylee/datasets/FMD/image/train"
	elif dataset_type == "valid":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/FMD/image/valid"
		if VM:
			DATA_DIR = "/home/jylee/datasets/FMD/image/valid"
	elif dataset_type == "test":
		DATA_DIR = "D:/Users/jylee/Dropbox/Files/Datasets/FMD/image/test"
		if VM:
			DATA_DIR = "/home/jylee/datasets/FMD/image/test"

	label_dict = {
		"fabric": 0,
		"glass": 1,
		"leather": 2,
		"metal": 3,
		"paper": 4,
		"plastic": 5
	}

	data = []

	for d in pathlib.Path(DATA_DIR).glob("*"):
		if d.name in label_dict.keys():
			label = label_dict[d.name]

			for f in d.glob("*.jpg"):
				data.append((f, label))

	num_batch = int(np.ceil(len(data) / batch_size))
	np.random.shuffle(data)

	for b in range(num_batch):
		start = b * batch_size
		end = min((b+1) * batch_size, len(data))

		X_batch = np.zeros((end - start, 128, 128, 3))
		Y_batch = np.zeros((end - start, len(label_dict)))

		for i in range(start, end):
			img = cv2.resize(plt.imread(data[i][0]), dsize=(128, 128)).astype(np.float32) / 255

			if len(img.shape) == 2:
				X_batch[i - start, :, :, 0] = img
				X_batch[i - start, :, :, 1] = img
				X_batch[i - start, :, :, 2] = img
			else:
				X_batch[i - start] = img

			Y_batch[i - start, data[i][1]] = 1

		yield X_batch, Y_batch

def train_FMD_cnn(gpu=0):
	epochs = 150
	batch_size = 128
	num_classes = 6
	if VM:
		ckpt_file = "ckpts/feature_cnn.ckpt"
	else:
		ckpt_file = "D:/ckpts/capstone/feature_cnn.ckpt"
	
	from FeatureCNN import FeatureCNN

	cnn = FeatureCNN(num_classes, ckpt_file, f"/gpu:{gpu}", batch_size=batch_size, eta=1e-2)
	cnn.build((128, 128, 3))

	for e in range(epochs):

		train_loader = iter(FMD_data_generate(batch_size, "train"))
		
		train_loss = 0.0
		cnt = 0

		for X_batch, Y_batch in train_loader:
			cnn.fit(X_batch, Y_batch, 0.8)
			train_loss += cnn.compute_loss(X_batch, Y_batch)

			cnt += 1

		train_loss /= cnt

		valid_loader = iter(FMD_data_generate(batch_size, "valid"))
		valid_acc = 0.0
		valid_loss = 0.0

		cnt = 0

		for X_batch, Y_batch in valid_loader:
			valid_loss += cnn.compute_loss(X_batch, Y_batch)
			valid_acc += cnn.score(X_batch, Y_batch)

			cnt += 1

		valid_loss /= cnt
		valid_acc /= cnt

		print("Epochs {}/{}".format(e+1, epochs))
		print("Train loss: {:.6f}".format(train_loss))
		print("Valid loss: {:.6f}".format(valid_loss))
		print("Valid acc: {:.6f}".format(valid_acc))

	cnn.save()

	# for i in range(10):
	# 	plt.imshow(images[i])
	# 	print(labels[i])
	# 	plt.show()

	# from FeatureCNN import FeatureCNN

	# cnn = FeatureCNN(num_classes, "feature_cnn", device="/gpu:0", eta=1e-3)
	# cnn.build((128, 128, 3), load_weights=False)
	# cnn.fit(images, labels, 0.75, epochs=50)

	# print(cnn.score(images, labels))
	# print(cnn.transform(images))


def train_trash_cnn(gpu=0):
	epochs = 150
	batch_size = 128
	num_classes = 4
	if VM:
		ckpt_file = "ckpts/capstone/feature_cnn.ckpt"
	else:
		ckpt_file = "D:/ckpts/capstone/feature_cnn.ckpt"

	from FeatureCNN import FeatureCNN

	cnn = FeatureCNN(num_classes, ckpt_file, f"/gpu:{gpu}", batch_size=batch_size, eta=1e-2)
	cnn.build((128, 128, 3))

	for e in range(epochs):
		train_loader = iter(trash_data_generator(batch_size, "train"))
		train_loss = 0.0

		cnt = 0
		for X_batch, Y_batch, _ in train_loader:
			cnn.fit(X_batch, Y_batch, 0.8)
			train_loss += cnn.compute_loss(X_batch, Y_batch)
			cnt += 1

		train_loss /= cnt

		val_loader = iter(trash_data_generator(batch_size, "valid"))
		val_loss = 0.0
		val_acc = 0.0

		cnt = 0
		for X_batch, Y_batch, _ in val_loader:
			val_loss += cnn.compute_loss(X_batch, Y_batch)
			val_acc += cnn.score(X_batch, Y_batch)
			cnt += 1

		val_loss /= cnt
		val_acc /= cnt

		print(f"Epoch {e+1}/{epochs}")
		print(f"Train loss: {train_loss:.6f}")
		print(f"Val loss: {val_loss:.6f}")
		print(f"Val acc: {val_acc:.6f}")

	cnn.save()

def validate_encoder():
	loader = iter(FMD_data_generate_no_label(128))

	from Encoder import Encoder

	if VM:
		encoder = Encoder("ckpts/capstone/encoder_FMD1.ckpt", "/gpu:0")
	else:
		encoder = Encoder("ckpts/capstone/encoder_FMD1.ckpt", "/gpu:0")
	encoder.build((128, 128, 3), load_weights=True)

	# encoder.encode(images[0].reshape(1, 128, 128, 3))

	images = next(loader)

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

def train_classifier_with_generator():
	epochs = 100
	num_step = 8
	batch_size = 128

	from Classifier import Classifier

	if VM:
		clf = Classifier(num_step, 4, "ckpts/capstone", eta=1e-4)
	else:
		clf = Classifier(num_step, 4, "D:/ckpts/capstone", eta=1e-4)
	clf.build(num_gpu=1)

	for e in range(epochs):
		train_loader = iter(rnn_trash_data_generator(batch_size, num_step, "train"))
		train_loss = 0.0

		cnt = 0

		for X_batch, Y_batch, num_batch in train_loader:
			clf.fit(X_batch, Y_batch)
			train_loss += clf.compute_loss(X_batch, Y_batch)

			# print(cnt, num_batch*4)
			cnt += 1

		train_loss /= cnt

		valid_loader = iter(rnn_trash_data_generator(batch_size, num_step, "valid"))
		val_loss = 0.0
		val_acc = 0.0

		cnt = 0

		for X_batch, Y_batch, num_batch in valid_loader:
			val_loss += clf.compute_loss(X_batch, Y_batch)
			val_acc += clf.score(X_batch, Y_batch)

			cnt += 1

		val_loss /= cnt
		val_acc /= cnt

		print("=============")
		print("Epochs {}/{}".format(e+1, epochs))
		print("train_loss: {:.6f}".format(train_loss))
		print("val_loss: {:.6f}".format(val_loss))
		print("val_acc: {:.6f}".format(val_acc))

	clf.save()

def test():
	loader = iter(FMD_data_generate_no_label(128))
	images = next(loader)

	for i in range(10):
		image = images[i]
		plt.imshow(image)
		plt.show()


# train_FMD_encoder()
# train_trash_encoder()
# validate_encoder()
# train_trash_cnn()
# train_FMD_cnn()
# train_FMD_encoder()
# train_classifier_with_generator()

# test()
