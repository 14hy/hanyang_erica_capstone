import tensorflow as tf
import numpy as np


class Encoder():

	def __init__(self, ckpt_file, device, rho=0.01, batch_size=128, eta=1e-3):
		self.graph = None
		self.sess = None
		self.saver = None
		self.rho = rho
		self.ckpt_file = ckpt_file
		self.device = device
		self.batch_size = batch_size
		self.eta = eta

	def __del__(self):
		if self.sess:
			self.sess.close()

	def build(self, shape, load_weights=False):
		self.graph = tf.Graph()

		print("Building encoder...")

		with tf.device(self.device):
			with self.graph.as_default() as g:
				self.X = self._init_placeholder(shape)

				self.encoder, self.decoder, logits, constraint = self._build_network(self.X, shape)

				self.loss = self._loss_function(logits, self.X, constraint)

				optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)
				self.train_op = optimizer.minimize(self.loss)

				self.sess = tf.Session()
				self.sess.run(tf.global_variables_initializer())

		with self.graph.as_default():
			self.saver = tf.train.Saver()
			if load_weights:
				self.saver.restore(self.sess, self.ckpt_file)
				print("Weights were loaded")

		print("Encoder building completed.")

	def fit(self, X_batch):

		with self.graph.as_default():

			self.sess.run(self.train_op, feed_dict={
				self.X: X_batch
			})

	def predict(self, X_data):
		n = X_data.shape[0]
		num_batch = int(np.ceil(n / self.batch_size))

		predictions = np.zeros_like(X_data)

		with self.graph.as_default():
			for b in range(num_batch):
				X_batch = self._next_batch(X_data, b)

				predictions[b * self.batch_size: b * self.batch_size + X_batch.shape[0]] = self.sess.run(self.decoder,
				                                                                                         feed_dict={
					                                                                                         self.X: X_batch
				                                                                                         })

		return predictions

	def save(self):
		with self.graph.as_default():
			self.saver.save(self.sess, self.ckpt_file)

	def encode(self, X_data):
		n, h, w, c = X_data.shape
		num_batch = int(np.ceil(n / self.batch_size))

		encoded = np.zeros((n, int(np.ceil(h / 4)), int(np.ceil(w / 4)), 32))

		with self.graph.as_default():
			for b in range(num_batch):
				X_batch = self._next_batch(X_data, b)

				encoded[b * self.batch_size: b * self.batch_size + X_batch.shape[0]] = self.sess.run(self.encoder,
				                                                                                     feed_dict={
					                                                                                     self.X: X_batch
				                                                                                     })

		return encoded

	def load_weights(self):
		with self.graph.as_default():
			self.saver.restore(self.sess, self.ckpt_file)

	def compute_loss(self, X_data):
		n = X_data.shape[0]
		num_batch = int(np.ceil(n / self.batch_size))

		losses = []

		for b in range(num_batch):
			X_batch = self._next_batch(X_data, b)

			l = self.sess.run(self.loss, feed_dict={
				self.X: X_batch
			})
			losses.append(l)

		return np.mean(losses)

	def _shuffle(self, X_data):
		n = X_data.shape[0]
		r = np.arange(n)
		np.random.shuffle(r)

		X_shuffled = X_data[r]

		return X_shuffled

	def _next_batch(self, X_data, b):
		start = b * self.batch_size
		end = min((b + 1) * self.batch_size, X_data.shape[0])

		X_batch = X_data[start: end]

		return X_batch

	def _build_network(self, X, shape):
		regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

		with tf.name_scope("encoder"):
			encoder = tf.layers.conv2d(X, 16, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu)
			encoder = tf.layers.conv2d(encoder, 32, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu)

			encoder = tf.layers.conv2d(encoder, 32, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu)
			encoder = tf.layers.conv2d(encoder, 64, (3, 3), strides=(2, 2), padding="SAME", activation=tf.nn.relu)

			encoder = tf.layers.conv2d(encoder, 64, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu)
			encoder = tf.layers.conv2d(encoder, 128, (3, 3), strides=(2, 2), padding="SAME", activation=tf.nn.sigmoid)

		with tf.name_scope("sparse"):
			rho_hat = tf.reduce_mean(encoder, axis=0)
			constraint = tf.multiply(self.rho, (tf.log(self.rho) - tf.log(rho_hat))) + tf.multiply((1.0 - self.rho), (tf.log(1.0 - self.rho) - tf.log(1.0 - rho_hat)))

		with tf.name_scope("decoder"):
			decoder = tf.layers.conv2d_transpose(encoder, 64, (3, 3), strides=(2, 2), padding="SAME", activation=tf.nn.relu)
			decoder = tf.layers.conv2d_transpose(decoder, 64, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu)

			decoder = tf.layers.conv2d_transpose(decoder, 32, (3, 3), strides=(2, 2), padding="SAME", activation=tf.nn.relu)
			decoder = tf.layers.conv2d_transpose(decoder, 32, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu)

			decoder = tf.layers.conv2d_transpose(decoder, 16, (3, 3), strides=(1, 1), padding="SAME", activation=tf.nn.relu)
			logits = tf.layers.conv2d_transpose(decoder, shape[-1], (3, 3), strides=(1, 1), padding="SAME", activation=None)
			decoder = tf.nn.sigmoid(logits)

		return encoder, decoder, logits, constraint

	def _loss_function(self, logits, Y, constraint):
		with tf.name_scope("loss"):
			# loss = tf.reduce_mean(tf.square(preds - Y))
			crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y)
			loss = 0.6 * tf.reduce_mean(crossentropy) + 0.4 * tf.reduce_mean(tf.square(tf.nn.sigmoid(logits) - Y)) + 3.0 * constraint

		return loss

	def _init_placeholder(self, shape):
		with tf.name_scope("in"):
			X = tf.placeholder(tf.float32, shape=(None, *shape), name="X")

		return X
