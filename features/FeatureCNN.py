import tensorflow as tf
import numpy as np


class FeatureCNN():

	def __init__(self, num_classes, ckpt_file, device, eta=1e-3, batch_size=128):
		self.graph = None
		self.sess = None
		self.eta = eta
		self.num_classes = num_classes
		self.device = device
		self.batch_size = batch_size
		self.ckpt_file = "/home/jylee/capstone/ckpts/" + ckpt_file + ".ckpt"
		self.saver = None

	def __del__(self):
		if self.sess:
			self.sess.close()

	def build(self, shape, load_weights=False):
		self.graph = tf.Graph()

		print("Building feature CNN...")

		with tf.device(self.device):
			with self.graph.as_default():
				self.X, self.Y, self.keep_prob = self._init_placeholder(shape)
				self.feature_map, logits = self._build_network(self.X, self.keep_prob)
				self.loss = self._loss_function(logits, self.Y)
				self.accuracy = self._accuracy(logits, self.Y)

				optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)
				self.train_op = optimizer.minimize(self.loss)

				self.sess = tf.Session()
				self.sess.run(tf.global_variables_initializer())

		with self.graph.as_default():
			self.saver = tf.train.Saver()
			if load_weights:
				self.saver.restore(self.sess, self.ckpt_file)

		print("Feature CNN building completed.")

	def fit(self, X_data, Y_data, keep_prob, epochs=100):
		n = X_data.shape[0]
		num_batch = int(np.ceil(n / self.batch_size))

		print("Training feature CNN...")

		with self.graph.as_default():
			for t in range(epochs):
				X_shuffled, Y_shuffled = self._shuffle(X_data, Y_data)

				for b in range(num_batch):
					X_batch, Y_batch = self._next_batch(X_shuffled, Y_shuffled, b)

					self.sess.run(self.train_op, feed_dict={
						self.X: X_batch, self.Y: Y_batch, self.keep_prob: keep_prob
					})

				loss = self._compute_loss(X_data, Y_data)
				print("Loss at epochs {0}: {1}".format(t + 1, loss))

			self.saver.save(self.sess, self.ckpt_file)
			print("Feature CNN was saved.")

		print("Feature CNN training completed.")

	def transform(self, X_data):
		n, h, w, c = X_data.shape
		num_batch = int(np.ceil(n / self.batch_size))

		transformed = np.zeros((n, int(np.ceil(h / 64)), int(np.ceil(w / 64)), 32))

		print("cnnA")

		with self.graph.as_default():
			for b in range(num_batch):
				X_batch = self._next_batch(X_data, None, b)

				transformed[b * self.batch_size: b * self.batch_size + X_batch.shape[0]] = self.sess.run(self.feature_map,
				                                                                                         feed_dict={
					                                                                                         self.X: X_batch
				                                                                                         })

		print("cnnB")

		return transformed

	def score(self, X_data, Y_data):
		n = X_data.shape[0]
		num_batch = int(np.ceil(n / self.batch_size))

		scores = []

		for b in range(num_batch):
			X_batch, Y_batch = self._next_batch(X_data, Y_data, b)

			acc = self.sess.run(self.accuracy, feed_dict={
				self.X: X_batch, self.Y: Y_batch, self.keep_prob: 1.
			})

			scores.append(acc)

		return np.mean(scores)

	def _compute_loss(self, X_data, Y_data):
		n = X_data.shape[0]
		num_batch = int(np.ceil(n / self.batch_size))

		losses = []

		for b in range(num_batch):
			X_batch, Y_batch = self._next_batch(X_data, Y_data, b)

			l = self.sess.run(self.loss, feed_dict={
				self.X: X_batch, self.Y: Y_batch, self.keep_prob: 1.
			})

			losses.append(l)

		return np.mean(losses)

	def _shuffle(self, X_data, Y_data=None):
		n = X_data.shape[0]
		r = np.arange(n)
		np.random.shuffle(r)

		X_shuffled = X_data[r]
		if Y_data is not None:
			Y_shuffled = Y_data[r]
			return X_shuffled, Y_shuffled

		return X_shuffled

	def _next_batch(self, X_data, Y_data=None, b=0):
		start = b * self.batch_size
		end = min((b + 1) * self.batch_size, X_data.shape[0])

		X_batch = X_data[start: end]
		if Y_data is not None:
			Y_batch = Y_data[start: end]
			return X_batch, Y_batch

		return X_batch

	def _build_network(self, X, keep_prob):
		with tf.name_scope("feature_cnn"):
			layer1 = tf.layers.conv2d(X, 8, (3, 3), strides=(1, 1), padding="SAME", activation=None)
			layer1 = tf.layers.batch_normalization(layer1)
			layer1 = tf.nn.relu(layer1)

			layer2 = tf.layers.max_pooling2d(layer1, (2, 2), strides=(2, 2), padding="SAME") # 64

			layer3 = tf.layers.conv2d(layer2, 16, (3, 3), strides=(1, 1), padding="SAME", activation=None)
			layer3 = tf.layers.batch_normalization(layer3)
			layer3 = tf.nn.relu(layer3)

			layer4 = tf.layers.max_pooling2d(layer3, (2, 2), strides=(2, 2), padding="SAME") # 32

			layer5 = tf.layers.conv2d(layer4, 24, (3, 3), strides=(1, 1), padding="SAME", activation=None)
			layer5 = tf.layers.batch_normalization(layer5)
			layer5 = tf.nn.relu(layer5)

			layer6 = tf.layers.max_pooling2d(layer5, (2, 2), strides=(2, 2), padding="SAME") # 16

			layer7 = tf.layers.conv2d(layer6, 32, (3, 3), strides=(1, 1), padding="SAME", activation=None)
			layer7 = tf.layers.batch_normalization(layer7)
			layer7 = tf.nn.sigmoid(layer7)

			layer8 = tf.layers.max_pooling2d(layer7, (2, 2), strides=(2, 2), padding="SAME") # 8

			layer9 = tf.layers.flatten(layer8)

			layer10 = tf.layers.dense(layer9, 256, activation=tf.nn.relu)
			layer10 = tf.layers.dropout(layer10, keep_prob)

			layer11 = tf.layers.dense(layer10, self.num_classes, activation=None)

		return layer8, layer11

	def _loss_function(self, pred, Y):
		with tf.name_scope("loss"):
			crossentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y)
			loss = tf.reduce_mean(crossentropy)

		return loss

	def _accuracy(self, pred, Y):
		with tf.name_scope("accuracy"):
			equality = tf.equal(tf.argmax(pred, axis=1), tf.argmax(Y, axis=1))
			accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

		return accuracy

	def _init_placeholder(self, shape):
		with tf.name_scope("in"):
			X = tf.placeholder(tf.float32, shape=(None, *shape), name="X")
			Y = tf.placeholder(tf.float32, shape=(None, self.num_classes), name="Y")
			keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")

		return X, Y, keep_prob
