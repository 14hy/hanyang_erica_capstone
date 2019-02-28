import tensorflow as tf
import numpy as np

class ClassifierRNN():

    def __init__(self, ckpt_file, eta=1e-3, batch_size=128):
        self.ckpt_file = ckpt_file
        self.eta = eta
        self.batch_size = batch_size
        self.graph = None
        self.sess = None

    def __del__(self):
        self.sess.close()

    def build(self, num_hidden, num_step, input_size, num_classes):

        print("Building RNN model...")
        self.num_classes = num_classes

        self.graph = tf.Graph()

        with self.graph.as_default() as g:
            self.X, self.Y = self._init_placeholder(num_step, input_size, num_classes)
            self.weights = self._init_weights(num_hidden, num_classes)

            cell = tf.nn.rnn_cell.LSTMCell(num_hidden, activation=tf.nn.relu, dtype=tf.float32)
            output, state = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
            output = tf.reshape(output, shape=(-1, output.get_shape().as_list()[1] * output.get_shape().as_list()[2]))

            dense1 = tf.layers.dense(output, 512, activation=tf.nn.relu)

            self.logits = tf.layers.dense(dense1, num_classes, activation=None)
            self.activations = tf.nn.softmax(self.logits)

            self.loss = self._loss_function(self.logits, self.Y)
            self.accuracy = self._accuracy(self.activations, self.Y)

            optimizer = tf.train.AdamOptimizer(learning_rate=self.eta)
            self.train_op = optimizer.minimize(self.loss)

            self.sess = tf.Session(graph=g)
            self.saver = tf.train.Saver()

            self.sess.run(tf.global_variables_initializer())

        print("RNN Model was built.")

    def fit(self, X_data, Y_data, overwrite=True, epochs=50):

        m, num_step, input_size = X_data.shape
        num_batch = int(np.ceil(m / self.batch_size))

        print("Training RNN...")

        with self.graph.as_default() as g:

            if overwrite is False:
                self.saver.restore(self.sess, self.ckpt_file)

            for t in range(epochs):

                X_shuffled, Y_shuffled = self._shuffle(X_data, Y_data)

                for b in range(num_batch):
                    X_batch, Y_batch = self._next_batch(X_shuffled, Y_shuffled, b)

                    self.sess.run(self.train_op, feed_dict={
                        self.X: X_batch, self.Y: Y_batch
                    })

                loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict={
                    self.X: X_batch, self.Y: Y_batch
                })
                print("Cost at epoch {0}: {1}".format(t + 1, loss))
                print("Accuracy on train batch: {}".format(acc))

            print("RNN was trained.")

            self.saver.save(self.sess, self.ckpt_file)
            print("RNN was saved.")

    def score(self, X_data, Y_data):

        m, num_step, input_size = X_data.shape
        num_batch = int(np.ceil(m / self.batch_size))

        accuracy = np.zeros((num_batch,))

        with self.graph.as_default() as g:
            for b in range(num_batch):
                X_batch, Y_batch = self._next_batch(X_data, Y_data, b)

                accuracy[b] = self.sess.run(self.accuracy, feed_dict={
                    self.X: X_batch, self.Y: Y_batch
                })

        return accuracy.mean()

    def predict(self, X_data):

        m, num_step, input_size = X_data.shape
        num_batch = int(np.ceil(m / self.batch_size))

        predictions = np.zeros((m, self.num_classes))

        with self.graph.as_default() as g:
            for b in range(num_batch):
                X_batch, Y_batch = self._next_batch(X_data, Y_data, b)

                predictions[b * self.batch_size: (b + 1) * self.batch_size] = self.sess.run(self.activations,
                                                                                            feed_dict={
                                                                                                self.X: X_batch,
                                                                                                self.Y: Y_batch
                                                                                            })

        return predictions

    def load(self):
        with self.graph.as_default() as g:
            try:
                self.saver.restore(self.sess, self.ckpt_file)
            except:
                print("ERROR: failed to load weights of RNN.")
                return

            print("RNN loaded weights.")

    def _next_batch(self, X_data, Y_data, b):
        m = X_data.shape[0]

        start = self.batch_size * b
        end = min(self.batch_size * (b + 1), m)

        X_batch = X_data[start: end]
        Y_batch = Y_data[start: end]

        return X_batch, Y_batch

    def _shuffle(self, X_data, Y_data):
        m = X_data.shape[0]

        r = np.arange(m)
        np.random.shuffle(r)

        X_shuffled = X_data[r]
        Y_shuffled = Y_data[r]

        return X_shuffled, Y_shuffled

    def _loss_function(self, logits, labels):
        with tf.name_scope("loss"):
            crossentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            loss = tf.reduce_mean(crossentropy, name="loss")

        tf.summary.scalar(loss.op.name, loss)

        return loss

    def _accuracy(self, activations, labels):
        with tf.name_scope("accuracy"):
            equality = tf.equal(tf.argmax(activations, axis=1), tf.argmax(labels, axis=1))
            accur = tf.reduce_mean(tf.cast(equality, tf.float32))

        tf.summary.scalar(accur.op.name, accur)

        return accur

    def _init_placeholder(self, num_step, input_size, num_classes):
        with tf.name_scope("in"):
            X = tf.placeholder(tf.float32, shape=(None, num_step, input_size), name="X")
            Y = tf.placeholder(tf.float32, shape=(None, num_classes), name="Y")

        return X, Y

    def _init_weights(self, num_hidden, num_classes):
        with tf.name_scope("weights"):
            weights = dict()

            weights["W"] = tf.Variable(tf.random_normal((num_hidden, num_classes)), dtype=tf.float32)
            weights["b"] = tf.Variable(tf.random_normal((1, num_classes)), dtype=tf.float32)

        return weights