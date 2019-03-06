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

	def fit(self, X_batch, Y_batch, keep_prob):

		with self.graph.as_default():

			self.sess.run(self.train_op, feed_dict={
				self.X: X_batch, self.Y: Y_batch, self.keep_prob: keep_prob
			})

	def transform(self, X_data):
		n, h, w, c = X_data.shape
		num_batch = int(np.ceil(n / self.batch_size))

		transformed = np.zeros((n, int(np.ceil(h / 32)), int(np.ceil(w / 32)), 128))

		with self.graph.as_default():
			for b in range(num_batch):
				X_batch = self._next_batch(X_data, None, b)

				transformed[b * self.batch_size: b * self.batch_size + X_batch.shape[0]] = self.sess.run(self.feature_map,
				                                                                                         feed_dict={
					                                                                                         self.X: X_batch
				                                                                                         })


		return transformed

	def save(self):
		with self.graph.as_default():
			self.saver.save(self.sess, self.ckpt_file)

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

	def compute_loss(self, X_batch, Y_batch):
		with self.graph.as_default():
			loss = self.sess.run(self.loss, feed_dict={
				self.X: X_batch, self.Y: Y_batch, self.keep_prob: 1.
			})

		return np.mean(loss)

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
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

		with tf.name_scope("feature_cnn"):
			layer1 = tf.layers.conv2d(X, 32, (3, 3), strides=(1, 1), padding="SAME", activation=None)#, kernel_regularizer=regularizer)
			layer1 = tf.layers.batch_normalization(layer1)
			layer1 = tf.nn.relu(layer1)

			layer2 = tf.layers.max_pooling2d(layer1, (2, 2), strides=(2, 2), padding="SAME") # 64
			
			layer3 = tf.layers.conv2d(layer2, 64, (3, 3), strides=(1, 1), padding="SAME", activation=None)#, kernel_regularizer=regularizer)
			layer3 = tf.layers.batch_normalization(layer3)
			layer3 = tf.nn.relu(layer3)

			layer4 = tf.layers.max_pooling2d(layer3, (2, 2), strides=(2, 2), padding="SAME") # 32

			layer5 = tf.layers.conv2d(layer4, 64, (3, 3), strides=(1, 1), padding="SAME", activation=None)#, kernel_regularizer=regularizer)
			layer5 = tf.layers.batch_normalization(layer5)
			layer5 = tf.nn.relu(layer5)

			layer6 = tf.layers.max_pooling2d(layer5, (2, 2), strides=(2, 2), padding="SAME") # 16
			
			layer7 = tf.layers.conv2d(layer6, 128, (3, 3), strides=(1, 1), padding="SAME", activation=None)#, kernel_regularizer=regularizer)
			layer7 = tf.layers.batch_normalization(layer7)
			layer7 = tf.nn.relu(layer7)

			layer8 = tf.layers.max_pooling2d(layer7, (2, 2), strides=(2, 2), padding="SAME") # 8

			layer9 = tf.layers.conv2d(layer8, 128, (3, 3), strides=(1, 1), padding="SAME", activation=None)#, kernel_regularizer=regularizer)
			layer9 = tf.layers.batch_normalization(layer9)
			layer9 = tf.nn.relu(layer9)

			layer10 = tf.layers.max_pooling2d(layer9, (2, 2), strides=(2, 2), padding="SAME") # 4

			layer11 = tf.layers.flatten(layer10)

			layer12 = tf.layers.dense(layer11, 256, activation=tf.nn.relu)#, kernel_regularizer=regularizer)
			layer12 = tf.layers.dropout(layer12, keep_prob)
			
			layer13 = tf.layers.dense(layer12, 64, activation=tf.nn.relu)#, kernel_regularizer=regularizer)
			layer13 = tf.layers.dropout(layer13, keep_prob)

			layer14 = tf.layers.dense(layer13, self.num_classes, activation=None)#, kernel_regularizer=regularizer)

		return layer10, layer14

	def _loss_function(self, pred, Y):
		with tf.name_scope("loss"):
			crossentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y)
			loss = tf.reduce_mean(crossentropy) + tf.losses.get_regularization_loss()

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
