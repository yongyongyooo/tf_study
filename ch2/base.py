import tensorflow as tf

import numpy as np
import random
from PIL import Image
from ch2.utils import get_batch, c_size, data_size, test_dir, test_path
import matplotlib.pyplot as plt

batch_size = 10
hidden_size = 512

X = tf.placeholder(tf.float32, [None, c_size * c_size * 3], name='x-input')
Y = tf.placeholder(tf.float32, [None, 2], name='y-input')

W1 = tf.Variable(tf.random_normal([c_size * c_size * 3, hidden_size]))
b1 = tf.Variable(tf.random_normal([hidden_size]))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([hidden_size, 2]))
b2 = tf.Variable(tf.random_normal([2]))
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(5):
    total_cost = 0
    goal_page = data_size // batch_size

    for page in range(goal_page):
        batch_x, batch_y = get_batch(page, batch_size)
        batch_x = batch_x.reshape((batch_size, c_size * c_size * 3))
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y})
        total_cost += c

        if page % 10 == 0:
            print(page, '/', goal_page)

    print('epoch :', epoch, 'cost : ', total_cost / goal_page)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
value = 0
for test_x in test_dir:
    test_img = Image.open(test_path + test_x)
    temp_img = np.array(test_img).reshape((1, c_size * c_size * 3))

    test_label = test_x.split('.')[0]
    class_map = [[0, 0]]

    if test_label == 'cat':
        class_map[0][0] = 1
    else:
        class_map[0][1] = 1

    result = sess.run(correct_prediction, feed_dict={X: temp_img, Y: np.array(class_map)})[0]

    if result:
        value += 1

print('accuracy : ', value / len(test_dir))

while True:
    r = random.randint(0, len(test_dir) - 1)
    test_x = test_dir[r]

    print('label : ', test_x.split('.')[0])
    test_img = Image.open(test_path + test_x)
    temp_img = np.array(test_img).reshape((1, c_size * c_size * 3))
    # print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    # print("Prediction: ", sess.run(
    #     tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
    result = sess.run(hypothesis, feed_dict={X: temp_img})

    if result[0][0] > result[0][1]:
        print('예측 : cat')
    else:
        print('예측 : dog')

    plt.imshow(test_img, cmap='Greys', interpolation='nearest')
    plt.show()
