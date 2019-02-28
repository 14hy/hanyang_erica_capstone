import tensorflow as tf
import numpy as np

class Encoder():

    def __init__(self, ckpt_file, eta=1e-2):
        self.graph = None
        self.sess = None
        self.ckpt_file = ckpt_file
        self.eta = eta

    def __del__(self):
        self.sess.close()

    def build(self, image_shape, device):
        print("Building Encoder...")

        self.graph = tf.Graph()

        with self.graph.as_default() as g:
            with tf.device(device):
                self.X = self._init_placeholder(image_shape)

                self.encoder, self.decoder, logits = self._build_encoder(self.X)

                self.loss = self._loss_function(logits, self.X)

                optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)
                self.train_op = optimizer.minimize(self.loss)

                self.sess = tf.Session(graph=g)

                self.sess.run(tf.global_variables_initializer())

            self.saver = tf.train.Saver()
        # tf.summary.FileWriter("./test/", graph=g)

        print("Encoder was built.")

    def load(self):
        with self.graph.as_default() as g:
            try:
                self.saver.restore(self.sess, self.ckpt_file)
            except:
                print("ERROR: failed to load weights of encoder.")
                return

            print("Encoder loaded weights.")

    def fit(self, X_data, epochs=50, batch_size=128, overwrite=True):
        m = X_data.shape[0]
        num_batch = int(np.ceil(m / batch_size))

        print("Training encoder...")

        with self.graph.as_default() as g:
            if overwrite is False:
                self.saver.restore(self.sess, self.ckpt_file)

            for t in range(epochs):
                X_shuffled = self._shuffle(X_data)

                for b in range(num_batch):
                    X_batch = self._next_batch(X_shuffled, batch_size, b)

                    self.sess.run(self.train_op, feed_dict={
                        self.X: X_batch
                    })

                loss = self.sess.run(self.loss, feed_dict={
                    self.X: X_batch
                })

                print("Cost at epoch {0}: {1}".format(t + 1, loss))

            print("Training completed.")

            self.saver.save(self.sess, self.ckpt_file)
            print("Encoder was saved.")

    def predict(self, X_data, batch_size=128):
        m = X_data.shape[0]
        num_batch = int(np.ceil(m / batch_size))

        result = np.zeros_like(X_data)

        with self.graph.as_default() as g:
            for b in range(num_batch):
                X_batch = self._next_batch(X_data, batch_size, b)
                result[b * batch_size: (b + 1) * batch_size] = self.sess.run(self.decoder, feed_dict={
                    self.X: X_batch
                })

        return result

    def encode(self, X_data, batch_size=128):
        m, h, w, c = X_data.shape
        num_batch = int(np.ceil(m / batch_size))

        result = np.zeros((m, h // 2, w // 2, 8))

        with self.graph.as_default() as g:
            for b in range(num_batch):
                X_batch = self._next_batch(X_data, batch_size, b)
                result[b * batch_size: (b + 1) * batch_size] = self.sess.run(self.encoder, feed_dict={
                    self.X: X_batch
                })

        return result

    def get_encoded_shape(self):
        return self.encoder.get_shape().as_list()

    def _shuffle(self, X_data):
        m = X_data.shape[0]
        r = np.arange(m)
        np.random.shuffle(r)

        X_shuffled = X_data[r]

        return X_shuffled

    def _next_batch(self, X_data, batch_size, b):
        start = b * batch_size
        end = min((b + 1) * batch_size, X_data.shape[0])

        X_batch = X_data[start: end]

        return X_batch

    def _build_encoder(self, X):
        n, h, w, c = X.get_shape()

        with tf.name_scope("encoder"):
            encoder = tf.contrib.layers.conv2d(X, 16, (5, 5), stride=1, padding="SAME", activation_fn=None)
            encoder = tf.nn.sigmoid(encoder)

            encoder = tf.contrib.layers.conv2d(encoder, 8, (5, 5), stride=1, padding="SAME", activation_fn=None)
            encoder = tf.nn.sigmoid(encoder)
            encoder = tf.nn.max_pool(encoder, (1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")

        with tf.name_scope("decoder"):
            decoder = tf.contrib.layers.conv2d_transpose(encoder, 16, (5, 5), stride=1, padding="SAME",
                                                         activation_fn=None)
            decoder = tf.nn.sigmoid(decoder)
            decoder = tf.image.resize_nearest_neighbor(decoder, (h, w))

            logits = tf.contrib.layers.conv2d_transpose(decoder, c, (5, 5), stride=1, padding="SAME",
                                                        activation_fn=None)
            decoder = tf.nn.sigmoid(logits)

        return encoder, decoder, logits

    def _loss_function(self, logits, labels):
        with tf.name_scope("loss"):
            # crossentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            # loss = tf.reduce_mean(crossentropy)
            loss = tf.reduce_mean(tf.square(logits - labels))

        return loss

    def _init_placeholder(self, image_shape):
        with tf.name_scope("in"):
            X = tf.placeholder(tf.float32, shape=(None, *image_shape), name="X")

        return X

    def _init_weights(self):
        weights = dict()
        with tf.variable_scope("weights"):
            weights["W1"] = tf.get_variable("W1", shape=(5, 5, 3, 16), dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())
            weights["b1"] = tf.get_variable("b1", shape=(1, 1, 1, 16), dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())

            weights["W2"] = tf.get_variable("W2", shape=(5, 5, 16, 8), dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())
            weights["b2"] = tf.get_variable("b2", shape=(1, 1, 1, 8), dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())

            weights["W3"] = tf.get_variable("W3", shape=(5, 5, 16, 8), dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())
            weights["b3"] = tf.get_variable("b3", shape=(1, 1, 1, 16), dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())

            weights["W4"] = tf.get_variable("W4", shape=(5, 5, 3, 16), dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())
            weights["b4"] = tf.get_variable("b4", shape=(1, 1, 1, 3), dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer())

        return weights