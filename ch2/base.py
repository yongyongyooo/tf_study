import tensorflow as tf

import numpy as np
import utils
from utils import get_batch, get_test, c_size, data_size

batch_size = 100
hidden_size = 512

X = tf.placeholder(tf.float32, [None, c_size * c_size], name='x-input')
Y = tf.placeholder(tf.float32, [None, 2], name='y-input')

W1 = tf.Variable(tf.random_normal([c_size*c_size, hidden_size]))
b1 = tf.Variable(tf.random_normal([hidden_size]))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([hidden_size, 2]))
b2 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(20):
    total_cost = 0
    goal_page = data_size // batch_size

    for page in range(goal_page):
        batch_x, batch_y = get_batch(page, batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={X:batch_x, Y: batch_y})
        total_cost += c
        print(page, '/', goal_page)

    print('epoch :', epoch, 'cost : ', total_cost / goal_page)


correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_data, test_label = get_test()

print('Accuracy:', sess.run(accuracy, feed_dict={ X: test_data, Y: test_label}))
