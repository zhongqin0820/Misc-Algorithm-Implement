import tensorflow as tf
from tensorflow.contrib import rnn
# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)
# define constants
# unrolled through 28 times steps
time_steps = 28
# hidden LSTM units
num_units = 128
# rows of 28 pixels
n_input = 28
# learning rate for adam
learning_rate = 0.001
# mnist is meant to be classified in 10 classes(0-9)
n_classes = 10
# size of batch
batch_size = 128

# define weigths and biases of appropriate shape to accomplish above task
out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))
# define placeholders
# input image placeholder
x = tf.placeholder("float", [None, time_steps, n_input])
# input label placeholder
y = tf.placeholder("float", [None, n_classes])

# processing the input tensor from [batch_size, time_steps, n_input] to "time_steps" of [batch_size,n_input] tensors
input = tf.unstack(x, time_steps, 1)

# define the network
lstm_layer = rnn.BasicLSTMCell(n_hidden, forget_bias=1)
outputs,_ = rnn.static_rnn(lstm_layer, input, dtype="float32")

# converting last output of dimension 
predication = tf.matmul(outputs[-1], out_weights) + out_bias


# loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
# optimization
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# model eval
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init
init = tf.global_variables_initializer()
# training
with tf.Session() as sess:
    sess.run(init)
    iter = 1
    while iter<800:
        batch_x,batch_y = mnist.train.next_batch(batch_size=batch_size)
        batch_x = batch_x.reshape((batch_size,time_steps,n_input))
        sess.run(opt, feed_dict={x:batch_x, y:batch_y})
        if iter%10==0:
            acc_ = sess.run(acc, feed_dict={x:batch_x, y:batch_y})
            loss_ = sess.run(loss, feed_dict={x:batch_x, y:batch_y})
            print('iter:',iter)
            print('acc:',acc_)
            print('loss:',loss_)
            print('-------------')
        iter = iter + 1

# test
test_x = mnist.test.images[:128].reshape((-1,time_steps,n_input))
test_y = mnist.test.labels[:128]
acc_test = sess.run(acc, feed_dict={x:batch_x, y:batch_y})
print('test acc:', acc_test)

