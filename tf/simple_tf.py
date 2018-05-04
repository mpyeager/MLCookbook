# Simple example of TensorFlow codeblack using TF Eager.

import tensorflow as tf 
from tensorflow.contrib.eager.python import tfe 

tfe.enable_eager_execution

x = tf.constant([3, 5, 7])
y = tf.constant([1, 2, 3])
print (x-y)

tf.Tensor([2, 3, 4], shape=(3,) dtype=int32)
