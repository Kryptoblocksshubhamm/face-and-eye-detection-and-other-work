# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:48:34 2017

@author: me
"""

# load the data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# starting tensorflow enviorment

import tensorflow as tf
sess = tf.InteractiveSession()

# Now Building a Softmax Regression Model


#I start building the computation graph by creating nodes for the input images and target output classes.

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#define the weights W and biases b for our model

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

#Predicted Class and Loss Function

y = tf.matmul(x,W) + b

# Now I am applying cross-entropy -

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train the model

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
# Now I am evaluating the model

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# Now building a convolution layer model -

# Weight Initialization

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#Convolution and Pooling

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  
#First Convolutional Layer

Wconv1 = weight_variable([5, 5, 1, 32])
bconv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

hconv1 = tf.nn.relu(conv2d(x_image, Wconv1) + bconv1)
h_pool1 = max_pool_2x2(hconv1)

#Second Convolutional Layer

#In order to build a deep network, we stack several layers of this type.
#The second layer will have 64 features for each 5x5 patch.

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected Layer

W_f1 = weight_variable([7 * 7 * 64, 1024])
b_f1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_f1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_f1) + b_f1)

#Dropout

#To reduce overfitting

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_f1, keep_prob)

#Readout Layer

#Finally, we add a layer, just like for the one layer softmax regression above.

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
                  
#Train and Evaluate the Model

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
 
#accuracy of the model

using softmax regression = 90.9%
using CNN = 99.1%


  