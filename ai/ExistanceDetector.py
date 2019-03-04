import numpy as np
import tensorflow as tf

IMAGE_SIZE = 128
CHANNEL = 3

class ExistanceDetector():

	def __init__(self, device, ckpt, eta=1e-3):
		self.device = device
		self.ckpt = ckpt
		self.eta = eta

		self.sess = None
		self.saver = None
		self.graph = None

	def __del__(self):
		if self.sess is not None:
			self.sess.close()

	def build(self):
		print("Building existance detector...")

		self.graph = tf.Graph()

		with self.graph.as_default():
			with tf.device(self.device):
				self.X, self.Y, self.keep_prob = self._init_placeholder()

				logits = self._build_net(self.X, self.keep_prob)

				self.loss = self._loss_function(logits, self.Y)
				self.accuracy = self._accuracy(logits, self.Y)
				self.predictions = self._predict(logits)

				optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)
				self.train_op = optimizer.minimize(self.loss)

				self.sess = tf.Session()
				self.sess.run(tf.global_variables_initializer())

			with tf.device("/cpu:0"):
				self.saver = tf.train.Saver()

		print("Existance detector was built.")

	def load_weights(self):
		with self.graph.as_default():
			self.saver.restore(self.sess, self.ckpt)
			print("Existance detector was restored.")

	def save(self):
		with self.graph.as_default():
			self.saver.save(self.sess, self.ckpt)
			print("Existance detector was saved.")

	def step(self, X_batch, Y_batch, keep_prob):
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

	def score(self, X_batch, Y_batch):
		with self.graph.as_default():
			acc = self.sess.run(self.accuracy, feed_dict={
				self.X: X_batch, self.Y: Y_batch, self.keep_prob: 1.0
			})

		return acc

	def predict(self, X_batch):
		with self.graph.as_default():
			predictions = self.sess.run(self.predictions, feed_dict={
				self.X: X_batch, self.keep_prob: 1.0
			})

		return predictions

	def _build_net(self, X, keep_prob):
		with tf.name_scope("net"):
			layer1 = tf.layers.conv2d(X, 8, (3, 3), strides=(1, 1), padding="SAME", activation=None) # 128
			layer1 = tf.layers.batch_normalization(layer1)
			layer1 = tf.nn.relu(layer1)
			
			layer2 = tf.layers.conv2d(layer1, 8, (3, 3), strides=(2, 2), padding="VALID", activation=None) # 64
			layer2 = tf.layers.batch_normalization(layer2)
			layer2 = tf.nn.relu(layer2)

			layer3 = tf.layers.max_pooling2d(layer2, (2, 2), strides=(2, 2), padding="SAME") # 32

			layer4 = tf.layers.conv2d(layer3, 16, (3, 3), strides=(1, 1), padding="SAME", activation=None) # 32
			layer4 = tf.layers.batch_normalization(layer4)
			layer4 = tf.nn.relu(layer4)

			layer5 = tf.layers.conv2d(layer4, 32, (3, 3), strides=(2, 2), padding="VALID", activation=None) # 16
			layer5 = tf.layers.batch_normalization(layer5)
			layer5 = tf.nn.relu(layer5)

			layer6 = tf.layers.max_pooling2d(layer5, (2, 2), strides=(2, 2), padding="SAME") # 8

			layer7 = tf.layers.flatten(layer6) # 8 * 8 * 32

			layer8 = tf.layers.dense(layer7, 128, activation=tf.nn.relu)
			layer8 = tf.layers.dropout(layer8, keep_prob)

			layer9 = tf.layers.dense(layer8, 32, activation=tf.nn.relu)
			layer9 = tf.layers.dropout(layer9, keep_prob)

			layer10 = tf.layers.dense(layer9, 1, activation=None)

			return layer10

	def _loss_function(self, logits, labels):
		with tf.name_scope("loss"):
			crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
			loss = tf.reduce_mean(crossentropy, name="loss")

			with tf.device("/cpu:0"):
				tf.summary.scalar(loss.op.name, loss)

		return loss

	def _accuracy(self, logits, labels):
		with tf.name_scope("accuracy"):
			equality = tf.equal(tf.round(tf.nn.sigmoid(logits)), labels)
			acc = tf.reduce_mean(tf.cast(equality, tf.float32), name="accuracy")

			with tf.device("/cpu:0"):
				tf.summary.scalar(acc.op.name, acc)

		return acc

	def _predict(self, logits):
		with tf.name_scope("predictions"):
			preds = tf.nn.sigmoid(logits)
			predictions = tf.round(preds)

		return predictions

	def _init_placeholder(self):
		with tf.name_scope("in"):
			X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL), name="X")
			Y = tf.placeholder(tf.float32, shape=(None, 1), name="Y")
			keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")

		return X, Y, keep_prob