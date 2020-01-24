import tensorflow as tf


class StatePreprocessor:
    def __init__(self):
        self.in_shape = (210, 160, 3)
        self.out_shape = (84, 84)
        self.crop_vals = (20, 0, 176, 160)

        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=list(self.in_shape), dtype=tf.uint8)
            output = tf.image.rgb_to_grayscale(self.input_state)
            off_h, off_w, targ_h, targ_w = self.crop_vals
            output = tf.image.crop_to_bounding_box(output, off_h, off_w, targ_h, targ_w)
            output = tf.image.resize_images(output, list(self.out_shape), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = tf.squeeze(output)
            self.output = output

    def process(self, sess, state):
        return sess.run(self.output, feed_dict={self.input_state: state})
