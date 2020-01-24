import os

import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense


class QEstimator:
    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None
        self.in_shape = (84, 84, 4)

        with tf.variable_scope(scope):
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        self.input_ph = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name='input')
        self.output_ph = tf.placeholder(shape=[None], dtype=tf.float32, name='output')
        self.actions_ph = tf.placeholder(shape=[None], dtype=tf.int32, name='actions')

        x = tf.cast(self.input_ph, dtype=tf.float32)
        x = tf.divide(x, 255.0)
        batch_size = tf.shape(self.input_ph)[0]

        # CNN
        conv1 = Conv2D(32, 8, 4, activation='relu', name='conv1')(x)
        conv2 = Conv2D(64, 4, 2, activation='relu', name='conv2')(conv1)
        conv3 = Conv2D(64, 3, 1, activation='relu', name='conv3')(conv2)

        # Fully connected layers
        flatten = Flatten(name='flatten')(conv3)
        fc1 = Dense(512, activation='relu', name='fc1')(flatten)
        fc2 = Dense(4, name='fc2')(fc1)
        self.predictions = fc2

        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_ph
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        self.losses = tf.squared_difference(self.output_ph, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        return sess.run(self.predictions, feed_dict={self.input_ph: s})

    def update(self, sess, s, a, y):
        feed_dict = {self.input_ph: s, self.output_ph: y, self.actions_ph: a}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict=feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


class TargetNetUpdate:
    def __init__(self, estimator1, estimator2):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def update(self, sess):
        sess.run(self.update_ops)

