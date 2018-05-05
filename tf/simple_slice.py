# Example code of a simple tensor slice in TensorFlow.

import tensorflow as tf
x = tf.constant([3, 5, 7],
                [4, 6, 8])
y = x[:, 1]
with tf.session as sess:
    print y.eval()