import numpy as np
import tensorflow as tf


class ClassifierRNN():

	def __init__(self, num_step, input_size, num_classes, ckpt_file, device, eta=1e-3, batch_size=128):
		self.num_step = num_step
		self.num_classes = num_classes
		self.input_size = input_size
		self.ckpt_file = ckpt_file
		self.eta = eta
		self.batch_size = batch_size
		self.device = device
		self.graph = None
		self.sess = None
		self.saver = None

	def __del__(self):
		if self.sess:
			self.sess.close()

	def build(self, load_weights=False):
		self.graph = tf.Graph()

		print("Building classifier RNN...")

		with tf.device(self.device):
			with self.graph.as_default():
				self.X, self.Y, self.keep_prob = self._init_placeholder(self.num_step, self.input_size, self.num_classes)

				self.logits = self._build_net(self.X, self.keep_prob, self.num_classes)
				self.pred = tf.nn.softmax(self.logits)

				self.loss = self._loss_function(self.logits, self.Y)
				self.accuracy = self._accuracy(self.logits, self.Y)

				optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)
				self.train_op = optimizer.minimize(self.loss)

				self.sess = tf.Session()
				self.sess.run(tf.global_variables_initializer())

		with self.graph.as_default():
			self.saver = tf.train.Saver()
			if load_weights:
				self.saver.restore(self.sess, self.ckpt_file)

		print("Classifier RNN building completed.")

	def load_weights(self):
		with self.graph.as_default():
			self.saver.restore(self.sess, self.ckpt_file)

	def fit(self, X_batch, Y_batch, keep_prob):

		with self.graph.as_default():

			self.sess.run(self.train_op, feed_dict={
				self.X: X_batch, self.Y: Y_batch, self.keep_prob: keep_prob
			})

	def compute_loss(self, X_batch, Y_batch):

		with self.graph.as_default():

			loss = self.sess.run(self.loss, feed_dict={
				self.X: X_batch, self.Y: Y_batch, self.keep_prob: 1.0
			})

		return loss

	def save(self):
		with self.graph.as_default():
			self.saver.save(self.sess, self.ckpt_file)

	def predict(self, X_data):
		n = X_data.shape[0]
		num_batch = int(np.ceil(n / self.batch_size))

		preds = np.zeros((n, self.num_classes))

		with self.graph.as_default():
			for b in range(num_batch):
				start = b * self.batch_size
				end = min((b+1) * self.batch_size, n)

				X_batch = X_data[start : end]

				preds[start : end] = self.sess.run(self.pred, feed_dict={
					self.X: X_batch, self.keep_prob: 1.
				})

		return preds

	def score(self, X_data, Y_data):
		n = X_data.shape[0]
		num_batch = int(np.ceil(n / self.batch_size))

		score = []

		with self.graph.as_default():
			for b in range(num_batch):
				start = b * self.batch_size
				end = min((b+1) * self.batch_size, n)

				X_batch = X_data[start : end]
				Y_batch = Y_data[start : end]

				s = self.sess.run(self.accuracy, feed_dict={
					self.X: X_batch, self.Y: Y_batch, self.keep_prob: 1.
				})

				score.append(s)

		return np.mean(s)

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

	def _build_net(self, X, keep_prob, num_classes):
		with tf.name_scope("rnn_net"):
			cell1 = tf.nn.rnn_cell.LSTMCell(64, activation=tf.nn.tanh, name="LSTM1")
			layer1, state1 = tf.nn.dynamic_rnn(cell1, X, dtype=tf.float32)
			layer1 = tf.layers.dropout(layer1, keep_prob)
			
			cell2 = tf.nn.rnn_cell.LSTMCell(64, activation=tf.nn.tanh, name="LSTM2")
			layer2, state2 = tf.nn.dynamic_rnn(cell2, layer1, dtype=tf.float32)
			layer2 = tf.layers.flatten(layer2)

			layer3 = tf.layers.dense(layer2, 128, activation=tf.nn.relu)
			layer3 = tf.layers.dropout(layer3, keep_prob)
			
			layer4 = tf.layers.dense(layer3, 32, activation=tf.nn.relu)
			layer4 = tf.layers.dropout(layer4, keep_prob)

			layer5 = tf.layers.dense(layer4, num_classes, activation=None)

		return layer5

	def _loss_function(self, logits, labels):
		with tf.name_scope("loss"):
			crossentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
			loss = tf.reduce_mean(crossentropy, name="loss")

		return loss

	def _accuracy(self, logits, labels):
		activations = tf.nn.softmax(logits)
		equality = tf.equal(tf.argmax(activations, axis=1), tf.argmax(labels, axis=1))
		accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

		return accuracy

	def _init_placeholder(self, num_step, input_size, num_classes):
		with tf.name_scope("in"):
			X = tf.placeholder(tf.float32, shape=(None, num_step, input_size), name="X")
			Y = tf.placeholder(tf.float32, shape=(None, num_classes), name="Y")
			keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")

		return X, Y, keep_prob