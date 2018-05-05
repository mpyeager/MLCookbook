

# Getting started with TensorFlow 
# NB: This code is intended for use with a Datalab or Colab notebook running in the europe-west1-b GCP zone.
# In this notebook, you play around with the TensorFlow Python API.


import tensorflow as tf
import numpy as np

print tf.__version__


# Adding two tensors
# First, let's try doing this using numpy, the Python numeric package. numpy code is immediately evaluated.


a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
c = np.add(a, b)
print c


# The equivalent code in TensorFlow consists of two steps:
# Step 1: Build the graph

a = tf.constant([5, 3, 8])
b = tf.constant([3, -1, 2])
c = tf.add(a, b)
print c


# c is an Op ("Add") that returns a tensor of shape (3,) and holds int32. The shape is inferred from the computation graph.
# Try the following in the cell above:
# Change the 5 to 5.0, and similarly the other five numbers. What happens when you run this cell? 
# Add an extra number to a, but leave b at the original (3,) shape. What happens when you run this cell?
# Change the code back to a version that works </li>
# 
# Step 2: Run the graph


with tf.Session() as sess:
  result = sess.run(c)
  print result


# Using a feed_dict
# 
# Same graph, but without hardcoding inputs at build stage


a = tf.placeholder(dtype=tf.int32, shape=(None,))  # batchsize x scalar
b = tf.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)
with tf.Session() as sess:
  result = sess.run(c, feed_dict={
      a: [3, 4, 5],
      b: [-1, 2, 3]
    })
  print result


# Heron's Formula in TensorFlow
# 
# The area of triangle whose three sides are $(a, b, c)$ is $\sqrt{s(s-a)(s-b)(s-c)}$ where $s=\frac{a+b+c}{2}$ 
# 
# Look up the available operations at https://www.tensorflow.org/api_docs/python/tf

def compute_area(sides):
  # slice the input to get the sides
  a = sides[:,0]  # 5.0, 2.3
  b = sides[:,1]  # 3.0, 4.1
  c = sides[:,2]  # 7.1, 4.8
  
  # Heron's formula
  s = (a + b + c) * 0.5   # (a + b) is a short-cut to tf.add(a, b)
  areasq = s * (s - a) * (s - b) * (s - c) # (a * b) is a short-cut to tf.multiply(a, b), not tf.matmul(a, b)
  return tf.sqrt(areasq)

with tf.Session() as sess:
  # pass in two triangles
  area = compute_area(tf.constant([
      [5.0, 3.0, 7.1],
      [2.3, 4.1, 4.8]
    ]))
  result = sess.run(area)
  print result


# Placeholder and feed_dict
# 
# More common is to define the input to a program as a placeholder and then to feed in the inputs. The difference between the code below and the code above is whether the "area" graph is coded up with the input values or whether the "area" graph is coded up with a placeholder through which inputs will be passed in at run-time.

with tf.Session() as sess:
  sides = tf.placeholder(tf.float32, shape=(None, 3))  # batchsize number of triangles, 3 sides
  area = compute_area(sides)
  result = sess.run(area, feed_dict = {
      sides: [
        [5.0, 3.0, 7.1],
        [2.3, 4.1, 4.8]
      ]
    })
  print result

# ## tf.eager
# 
# tf.eager allows you to avoid the build-then-run stages. However, most production code will follow the lazy evaluation paradigm because the lazy evaluation paradigm is what allows for multi-device support and distribution. 
# Develop using tf.eager and then comment out the eager execution and add in the session management code.
# 
# You may need to click on Reset Session to try this out.

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()

def compute_area(sides):
  # slice the input to get the sides
  a = sides[:,0]  # 5.0, 2.3
  b = sides[:,1]  # 3.0, 4.1
  c = sides[:,2]  # 7.1, 4.8
  
  # Heron's formula
  s = (a + b + c) * 0.5   # (a + b) is a short-cut to tf.add(a, b)
  areasq = s * (s - a) * (s - b) * (s - c) # (a * b) is a short-cut to tf.multiply(a, b), not tf.matmul(a, b)
  return tf.sqrt(areasq)

area = compute_area(tf.constant([
      [5.0, 3.0, 7.1],
      [2.3, 4.1, 4.8]
    ]))


print area
