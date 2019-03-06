import tensorflow as tf
import numpy as np

class FeatureCNN():

	def __init__(self, eta, ckpt, num_classes, device="/cpu:0"):
		self.eta = eta
		self.ckpt = ckpt
		self.num_classes = num_classes
		self.device = device

		self.sess = None
		self.graph = None
		self.saver = None

	def __del__(self):
		if self.sess is not None:
			self.sess.close()

	def build(self):
		self.graph = tf.Graph()

		print("Building Feature CNN...")

		with tf.device(self.device):
			with self.graph.as_default():
				self.X, self.Y, self.keep_prob = self._init_placeholder()

				self.feature_maps, logits = self._build_net(self.X, self.keep_prob)

				self.loss = self._loss_function(logits, self.Y)
				self.accuracy = self._accuracy(logits, self.Y)

				optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)
				self.train_op = optimizer.minimize(self.loss)

				self.sess = tf.Session()
				self.sess.run(tf.global_variables_initializer())

				with tf.device("/cpu:0"):
					self.saver = tf.train.Saver(self.sess, self.ckpt)

		print("Feature CNN was built.")

	def load_weights(self):
		with self.graph.as_default():
			self.saver.restore(self.sess, self.ckpt)

	def save(self):
		with self.graph.as_default():
			self.saver.save(self.sess, self.ckpt)

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

	def accuracy(self, X_batch, Y_batch):
		with self.graph.as_default():
			acc = self.sess.run(self.accuracy, feed_dict={
				self.X: X_batch, self.Y: Y_batch, self.keep_prob: 1.0
			})

		return acc

	def transform(self, X_batch):
		with self.graph.as_default():
			transformed = self.sess.run(self.feature_maps, feed_dict={
				self.X: X_batch, self.keep_prob: 1.0
			})

		return transformed

	def _init_placeholder(self):
		with tf.name_scope("in"):
			X = tf.placeholder(tf.float32, shape=(None, 128, 128, 3), name="X")
			Y = tf.placeholder(tf.float32, shape=(None, self.num_classes), name="Y")
			keep_prob = tf.placeholder(tf.float32, shape=(), name="keep_prob")

		return X, Y, keep_prob

	def _loss_function(self, logits, labels):
		with tf.name_scope("loss"):
			crossentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
			loss = tf.reduce_mean(crossentropy, name="loss")

		return loss

	def _accuracy(self, logits, labels):
		with tf.name_scope("accuracy"):
			ps = tf.nn.softmax(logits)
			equality = tf.equal(tf.argmax(ps, axis=1), tf.argmax(labels, axis=1))
			accuracy = tf.reduce_mean(tf.cast(equality, tf.float32), name="accuracy")

		return accuracy

	def _build_net(self, X, keep_prob):
		with tf.name_scope("net1"):
			layer1_1 = tf.layers.conv2d(X, 32, (3, 3), strides=(1, 1), padding="SAME") # 128
			layer1_1 = tf.layers.batch_normalization(layer1_1)
			layer1_1 = tf.nn.relu(layer1_1)

			layer2_1 = tf.layers.max_pooling2d(layer1_1, (2, 2), strides=(2, 2), padding="SAME") # 64

			layer3_1 = tf.layers.conv2d(layer2_1, 32, (3, 3), strides=(1, 1), padding="SAME")
			layer3_1 = tf.layers.batch_normalization(layer3_1)
			layer3_1 = tf.nn.relu(layer3_1)

			layer4_1 = tf.layers.max_pooling2d(layer3_1, (2, 2), strides=(2, 2), padding="SAME") # 32

			layer5_1 = tf.layers.conv2d(layer4_1, 64, (3, 3), strides=(1, 1), padding="SAME")
			layer5_1 = tf.layers.batch_normalization(layer5_1)
			layer5_1 = tf.nn.relu(layer5_1)

			layer6_1 = tf.layers.max_pooling2d(layer5_1, (2, 2), strides=(2, 2), padding="SAME") # 16
			
			layer7_1 = tf.layers.conv2d(layer6_1, 64, (3, 3), strides=(1, 1), padding="SAME")
			layer7_1 = tf.layers.batch_normalization(layer7_1)
			layer7_1 = tf.nn.relu(layer7_1)

			layer8_1 = tf.layers.max_pooling2d(layer7_1, (2, 2), strides=(2, 2), padding="SAME") # 8
			
		with tf.name_scope("net2"):
			layer1_2 = tf.layers.conv2d(X, 32, (5, 5), strides=(1, 1), padding="SAME")
			layer1_2 = tf.layers.batch_normalization(layer1_2)
			layer1_2 = tf.nn.relu(layer1_2)

			layer2_2 = tf.layers.max_pooling2d(layer1_2, (2, 2), strides=(2, 2), padding="SAME")
			
			layer3_2 = tf.layers.conv2d(layer2_2, 32, (5, 5), strides=(1, 1), padding="SAME")
			layer3_2 = tf.layers.batch_normalization(layer3_2)
			layer3_2 = tf.nn.relu(layer3_2)

			layer4_2 = tf.layers.max_pooling2d(layer3_2, (2, 2), strides=(2, 2), padding="SAME")
			
			layer5_2 = tf.layers.conv2d(layer4_2, 64, (5, 5), strides=(1, 1), padding="SAME")
			layer5_2 = tf.layers.batch_normalization(layer5_2)
			layer5_2 = tf.nn.relu(layer5_2)

			layer6_2 = tf.layers.max_pooling2d(layer5_2, (2, 2), strides=(2, 2), padding="SAME")
			
			layer7_2 = tf.layers.conv2d(layer6_2, 64, (5, 5), strides=(1, 1), padding="SAME")
			layer7_2 = tf.layers.batch_normalization(layer7_2)
			layer7_2 = tf.nn.relu(layer7_2)

			layer8_2 = tf.layers.max_pooling2d(layer7_2, (2, 2), strides=(2, 2), padding="SAME")

		with tf.name_scope("merged"):
			layer9 = tf.concat([layer8_1, layer8_2], axis=0)
			layer9 = tf.layers.dropout(layer9, keep_prob)

			layer10 = tf.layers.dense(layer9, 512, activation=tf.nn.sigmoid)
			layer10 = tf.layers.dropout(layer10, keep_prob)

			layer11 = tf.layers.dense(layer10, 128, activation=tf.nn.relu)
			layer11 = tf.layers.dropout(layer11, keep_prob)

			layer12 = tf.layers.dense(layer11, self.num_classes, activation=None)

			return layer10, layer12