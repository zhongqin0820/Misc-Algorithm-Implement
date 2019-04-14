from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
# import numpy as np

print(tf.__version__)



data = input_data.read_data_sets(r'./data/MNIST',one_hot=True)

batch_size = 100

n_batch = data.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None,784], name = 'x')
y = tf.placeholder(tf.float32, [None,10], name = 'y')

w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
pred = tf.matmul(x,w) + b

loss = tf.reduce_mean(tf.square(y-pred))
train = tf.train.AdagradOptimizer(0.05).minimize(loss)

init = tf.global_variables_initializer()

correct = tf.equal(tf.arg_max(y,1),tf.arg_max(pred,1))
acc = tf.reduce_mean(tf.cast(correct,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for idx in range(200):
        for batch in range(n_batch):
            b_xs, b_ys = data.train.next_batch(batch_size)
            sess.run(train,feed_dict={x:b_xs,y:b_ys})
        accp = sess.run(acc, feed_dict={x:data.test.images,y:data.test.labels})
        print(idx, accp)