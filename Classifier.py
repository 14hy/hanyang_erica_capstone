import numpy as np
import tensorflow as tf
import sys
import threading as th

sys.path.append("./features/")

from StackedEncoder import StackedEncoder
from FeatureCNN import FeatureCNN
from ClassifierRNN import ClassifierRNN

class Classifier():

	def __init__(self, num_step, num_classes, eta=1e-3, batch_size=128,  net_type="encoder"):
		self.eta = eta
		self.batch_size = batch_size
		self.num_step = num_step
		self.num_classes = num_classes
		self.net_type = net_type

		# self.encoder = None
		self.encoders = []
		self.cnn = []
		self.rnn = None

		self.encoded_size = 2 * 2 * 16
		self.cnn_size = 8 * 8 * 32
		# self.input_size = self.encoded_size + self.cnn_size
		# self.input_size = self.cnn_size + self.cnn_size

	def build(self, num_gpu=1):
		print("Building classifier...")

		# self.encoder = StackedEncoder("encoder_texture", device="/gpu:0")
		# self.encoder.build((128, 128, 3), load_weights=True)

		if self.net_type == "cnn":
			cnn1 = FeatureCNN(self.num_classes, "feature_cnn1", device="/gpu:0", eta=self.eta)
			cnn1.build((128, 128, 3), load_weights=True)

			self.cnn.append(cnn1)

			cnn2 = FeatureCNN(self.num_classes, "feature_cnn2", device="/gpu:{}".format(num_gpu-1), eta=self.eta)
			cnn2.build((128, 128, 3), load_weights=True)

			self.cnn.append(cnn2)

			self.input_size = self.cnn_size * 2

		elif self.net_type == "encoder":
			encoder1 = StackedEncoder("encoder1", device="/gpu:0", eta=[self.eta] * 3)
			encoder1.build((128, 128, 3))

			self.encoders.append(encoder1)

			encoder2 = StackedEncoder("encoder1", device="/gpu:1", eta=[self.eta] * 3)
			encoder2.build((128, 128, 3))

			self.encoders.append(encoder2)

			self.input_size = self.encoded_size * 2

		self.rnn = ClassifierRNN(self.num_step, self.input_size, self.num_classes, "rnn", device="/gpu:0", eta=self.eta, batch_size=self.batch_size)
		self.rnn.build(load_weights=False)

		print("Classifier building completed.")

	def load_weights(self):
		self.rnn.load_weights()

	def fit(self, X_data, Y_data, epochs=100):
		print("Training classifier...")

		n = X_data.shape[0]
		num_batch = int(np.ceil(n / self.batch_size))

		for e in range(epochs):
			X_shuffled, Y_shuffled = self._shuffle(X_data, Y_data)

			for b in range(num_batch):
				start = b * self.batch_size
				end = min((b + 1) * self.batch_size, X_data.shape[0])
				X_batch = X_shuffled[start : end]
				Y_batch = Y_shuffled[start : end]

				features = self._extract_feature(X_batch)
				self.rnn.fit(features, Y_batch, 0.2, epochs=1, save=False)

			loss = self.rnn._compute_cost()
			print("Loss at epochs {}: {:.6f}".format(e, loss))

		self.rnn.save_model()

		print("Classifier training completed.")

	def predict(self, X_data):
		features = self._extract_feature(X_data)
		preds = self.rnn.predict(features)
		return preds

	def score(self, X_data, Y_data):
		features = self._extract_feature(X_data)
		score = self.rnn.score(features, Y_data)

		return score

	def _shuffle(self, X_data, Y_data):
		n = X_data.shape[0]
		r = np.arange(n)
		np.random.shuffle(r)

		X_shuffled = X_data[r]
		Y_shuffled = Y_data[r]

		return X_shuffled, Y_shuffled

	def _next_batch(self, X_data, Y_data, b):
		start = b * self.batch_size
		end = min((b + 1) * self.batch_size, X_data.shape[0])

		X_batch = X_data[start: end]
		Y_batch = Y_data[start: end]

		return X_batch, Y_batch

	def _extract_feature(self, X_data):
		"""

		:param X_data: shape of (n, num_step, h, w, c)
		:return:
		"""
		n = X_data.shape[0]

		features = np.zeros((n, self.num_step, self.input_size))

		for i in range(self.num_step):
			lst1 = []
			lst2 = []

			if self.net_type == "cnn":
				th1 = th.Thread(target=lambda lst, cnn, x: lst.append(cnn.transform(x)), args=(lst1, self.cnn[0], X_data[:, i, :, :, :]))
				th2 = th.Thread(target=lambda lst, cnn, x: lst.append(cnn.transform(x)), args=(lst2, self.cnn[1], X_data[:, i, :, :, :]))
				size = self.cnn_size
				
			elif self.net_type == "encoder":
				th1 = th.Thread(target=lambda lst, enc, x: lst.append(enc.encode(x)), args=(lst1, self.encoders[0], X_data[:, i, :, :, :]))
				th2 = th.Thread(target=lambda lst, enc, x: lst.append(enc.encode(x)), args=(lst2, self.encoders[1], X_data[:, i, :, :, :]))
				size = self.encoded_size

			th1.start()
			th2.start()

			th1.join()
			th2.join()

			features[:, i, : size] = lst1[0].reshape(n, size)
			features[:, i, size :] = lst2[0].reshape(n, size)

		return features