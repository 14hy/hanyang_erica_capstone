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
				params = self._init_params()

				self.feature_maps, logits = self._build_net(self.X, self.keep_prob, params)

				self.loss = self._loss_function(logits, self.Y)
				self.accuracy = self._accuracy(logits, self.Y)

				optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)
				self.train_op = optimizer.minimize(self.loss)

				self.sess = tf.Session()
				self.sess.run(tf.global_variables_initializer())
	
				with tf.device("/cpu:0"):
					self.saver = tf.train.Saver()

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

	def score(self, X_batch, Y_batch):
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

	def _init_params(self):
		params = dict()

		with tf.name_scope("weights"):
			with tf.name_scope("conv1"):
				params["1/W1"] = tf.Variable(tf.random_normal((3, 3, 3, 32)), dtype=tf.float32, name="W1")
				params["1/b1"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32, name="b1")
				params["1/batch_norm1/mean"] = tf.Variable(tf.random_normal((1, 128, 128, 32)), dtype=tf.float32, name="batch_norm_mean1")
				params["1/batch_norm1/var"] = tf.Variable(tf.random_normal((1, 128, 128, 32)), dtype=tf.float32, name="batch_norm_var1")

				params["1/W2"] = tf.Variable(tf.random_normal((3, 3, 32, 32)), dtype=tf.float32, name="W2")
				params["1/b2"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32, name="b2")
				params["1/batch_norm2/mean"] = tf.Variable(tf.random_normal((1, 64, 64, 32)), dtype=tf.float32, name="batch_norm_mean2")
				params["1/batch_norm2/var"] = tf.Variable(tf.random_normal((1, 64, 64, 32)), dtype=tf.float32, name="batch_norm_var2")
				
				params["1/W3"] = tf.Variable(tf.random_normal((3, 3, 32, 64)), dtype=tf.float32, name="W3")
				params["1/b3"] = tf.Variable(tf.random_normal((1, 1, 1, 64)), dtype=tf.float32, name="b3")
				params["1/batch_norm3/mean"] = tf.Variable(tf.random_normal((1, 32, 32, 64)), dtype=tf.float32, name="batch_norm_mean3")
				params["1/batch_norm3/var"] = tf.Variable(tf.random_normal((1, 32, 32, 64)), dtype=tf.float32, name="batch_norm_var3")
				
				params["1/W4"] = tf.Variable(tf.random_normal((3, 3, 64, 64)), dtype=tf.float32, name="W4")
				params["1/b4"] = tf.Variable(tf.random_normal((1, 1, 1, 64)), dtype=tf.float32, name="b4")
				params["1/batch_norm4/mean"] = tf.Variable(tf.random_normal((1, 16, 16, 64)), dtype=tf.float32, name="batch_norm_mean4")
				params["1/batch_norm4/var"] = tf.Variable(tf.random_normal((1, 16, 16, 64)), dtype=tf.float32, name="batch_norm_var4")

			with tf.name_scope("conv2"):
				params["2/W1"] = tf.Variable(tf.random_normal((5, 5, 3, 32)), dtype=tf.float32, name="W1")
				params["2/b1"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32, name="b1")
				params["2/batch_norm1/mean"] = tf.Variable(tf.random_normal((1, 128, 128, 32)), dtype=tf.float32, name="batch_norm_mean1")
				params["2/batch_norm1/var"] = tf.Variable(tf.random_normal((1, 128, 128, 32)), dtype=tf.float32, name="batch_norm_var1")

				params["2/W2"] = tf.Variable(tf.random_normal((5, 5, 32, 32)), dtype=tf.float32, name="W2")
				params["2/b2"] = tf.Variable(tf.random_normal((1, 1, 1, 32)), dtype=tf.float32, name="b2")
				params["2/batch_norm2/mean"] = tf.Variable(tf.random_normal((1, 64, 64, 32)), dtype=tf.float32, name="batch_norm_mean2")
				params["2/batch_norm2/var"] = tf.Variable(tf.random_normal((1, 64, 64, 32)), dtype=tf.float32, name="batch_norm_var2")
				
				params["2/W3"] = tf.Variable(tf.random_normal((5, 5, 32, 64)), dtype=tf.float32, name="W3")
				params["2/b3"] = tf.Variable(tf.random_normal((1, 1, 1, 64)), dtype=tf.float32, name="b3")
				params["2/batch_norm3/mean"] = tf.Variable(tf.random_normal((1, 32, 32, 64)), dtype=tf.float32, name="batch_norm_mean3")
				params["2/batch_norm3/var"] = tf.Variable(tf.random_normal((1, 32, 32, 64)), dtype=tf.float32, name="batch_norm_var3")
				
				params["2/W4"] = tf.Variable(tf.random_normal((5, 5, 64, 64)), dtype=tf.float32, name="W4")
				params["2/b4"] = tf.Variable(tf.random_normal((1, 1, 1, 64)), dtype=tf.float32, name="b4")
				params["2/batch_norm4/mean"] = tf.Variable(tf.random_normal((1, 16, 16, 64)), dtype=tf.float32, name="batch_norm_mean4")
				params["2/batch_norm4/var"] = tf.Variable(tf.random_normal((1, 16, 16, 64)), dtype=tf.float32, name="batch_norm_var4")

			with tf.name_scope("fc"):
				params["W5"] = tf.Variable(tf.random_normal((2*8*8*64, 256)), dtype=tf.float32, name="W5")
				params["b5"] = tf.Variable(tf.random_normal((1, 256)), dtype=tf.float32, name="b5")
				
				params["W6"] = tf.Variable(tf.random_normal((256, 64)), dtype=tf.float32, name="W5")
				params["b6"] = tf.Variable(tf.random_normal((1, 64)), dtype=tf.float32, name="b5")
				
				params["W7"] = tf.Variable(tf.random_normal((64, self.num_classes)), dtype=tf.float32, name="W5")
				params["b7"] = tf.Variable(tf.random_normal((1, self.num_classes)), dtype=tf.float32, name="b5")

		return params

	def _build_net(self, X, keep_prob, params):
		with tf.name_scope("net1"):
			layer1_1 = tf.nn.conv2d(X, params["1/W1"], strides=(1, 1, 1, 1), padding="SAME") + params["1/b1"]
			layer1_1 = tf.nn.batch_normalization(layer1_1, params["1/batch_norm1/mean"], params["1/batch_norm1/var"], None, None, 1e-7)
			layer1_1 = tf.nn.relu(layer1_1)
			layer1_1 = tf.nn.max_pool(layer1_1, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME") # 64

			layer2_1 = tf.nn.conv2d(layer1_1, params["1/W2"], strides=(1, 1, 1, 1), padding="SAME") + params["1/b2"]
			layer2_1 = tf.nn.batch_normalization(layer2_1, params["1/batch_norm2/mean"], params["1/batch_norm2/var"], None, None, 1e-7)
			layer2_1 = tf.nn.relu(layer2_1)
			layer2_1 = tf.nn.max_pool(layer2_1, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME") # 32

			layer3_1 = tf.nn.conv2d(layer2_1, params["1/W3"], strides=(1, 1, 1, 1), padding="SAME") + params["1/b3"]
			layer3_1 = tf.nn.batch_normalization(layer3_1, params["1/batch_norm3/mean"], params["1/batch_norm3/var"], None, None, 1e-7)
			layer3_1 = tf.nn.relu(layer3_1)
			layer3_1 = tf.nn.max_pool(layer3_1, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME") # 16

			layer4_1 = tf.nn.conv2d(layer3_1, params["1/W4"], strides=(1, 1, 1, 1), padding="SAME") + params["1/b4"]
			layer4_1 = tf.nn.batch_normalization(layer4_1, params["1/batch_norm4/mean"], params["1/batch_norm4/var"], None, None, 1e-7)
			layer4_1 = tf.nn.relu(layer4_1)
			layer4_1 = tf.nn.max_pool(layer4_1, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME") # 8
			
		with tf.name_scope("net2"):
			layer1_2 = tf.nn.conv2d(X, params["2/W1"], strides=(1, 1, 1, 1), padding="SAME") + params["2/b1"]
			layer1_2 = tf.nn.batch_normalization(layer1_2, params["2/batch_norm1/mean"], params["2/batch_norm1/var"], None, None, 1e-7)
			layer1_2 = tf.nn.relu(layer1_2)
			layer1_2 = tf.nn.max_pool(layer1_2, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

			layer2_2 = tf.nn.conv2d(layer1_2, params["2/W2"], strides=(1, 1, 1, 1), padding="SAME") + params["2/b2"]
			layer2_2 = tf.nn.batch_normalization(layer2_2, params["2/batch_norm2/mean"], params["2/batch_norm2/var"], None, None, 1e-7)
			layer2_2 = tf.nn.relu(layer2_2)
			layer2_2 = tf.nn.max_pool(layer2_2, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

			layer3_2 = tf.nn.conv2d(layer2_2, params["2/W3"], strides=(1, 1, 1, 1), padding="SAME") + params["2/b3"]
			layer3_2 = tf.nn.batch_normalization(layer3_2, params["2/batch_norm3/mean"], params["2/batch_norm3/var"], None, None, 1e-7)
			layer3_2 = tf.nn.relu(layer3_2)
			layer3_2 = tf.nn.max_pool(layer3_2, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

			layer4_2 = tf.nn.conv2d(layer3_2, params["2/W4"], strides=(1, 1, 1, 1), padding="SAME") + params["2/b4"]
			layer4_2 = tf.nn.batch_normalization(layer4_2, params["2/batch_norm4/mean"], params["2/batch_norm4/var"], None, None, 1e-7)
			layer4_2 = tf.nn.relu(layer4_2)
			layer4_2 = tf.nn.max_pool(layer4_2, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

		with tf.name_scope("merged"):
			layer4_1 = tf.reshape(layer4_1, (-1, 8*8*64))
			layer4_2 = tf.reshape(layer4_2, (-1, 8*8*64))

			layer5 = tf.concat([layer4_1, layer4_2], axis=1)
			layer5 = tf.matmul(layer5, params["W5"]) + params["b5"]
			layer5 = tf.nn.sigmoid(layer5)
			layer5 = tf.nn.dropout(layer5, keep_prob)

			layer6 = tf.matmul(layer5, params["W6"]) + params["b6"]
			layer6 = tf.nn.relu(layer6)
			layer6 = tf.nn.dropout(layer6, keep_prob)

			layer7 = tf.matmul(layer6, params["W7"]) + params["b7"]
			
		return layer5, layer7

