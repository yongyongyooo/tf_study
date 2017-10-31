import numpy as np
import tensorflow as tf
from utils import get_batch, get_test, c_size, data_size
import matplotlib.pyplot as plt


batch_size = 10
learning_rate = 0.05
X = tf.placeholder(tf.float32, [None, c_size, c_size, 3], name='x-input')
Y = tf.placeholder(tf.float32, [None, 2], name='y-input')

conv1 = tf.layers.conv2d(inputs=X,
                             filters=20,
                             kernel_size=[3, 3],
                             padding='SAME',
                             activation=tf.nn.relu)

pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                pool_size=[2, 2],
                                padding='SAME',
                                strides=2)

conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=20,
                             kernel_size=[3, 3],
                             padding='SAME',
                             activation=tf.nn.relu)

pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                pool_size=[2, 2],
                                padding='SAME',
                                strides=2)

flat_size = 32 * 32 * 20
layer2 = tf.reshape(pool2, [-1, flat_size])

W1 = tf.get_variable('W1',
                         shape=[flat_size, 40],
                         initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([40]))
hypo1 = tf.matmul(layer2, W1) + b1

dense1 = tf.layers.dense(inputs=hypo1, units=20, activation=tf.nn.relu)
dense2 = tf.layers.dense(inputs=dense1, units=2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense2, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(5):
    total_cost = 0
    goal_page = data_size // batch_size

    for page in range(goal_page):
        batch_x, batch_y = get_batch(page, batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y})
        total_cost += c

        if page % 10 == 0:
            print(page, '/', goal_page)

    print('epoch :', epoch, 'cost : ', total_cost / goal_page)

correct_prediction = tf.equal(tf.argmax(dense2, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# test_data, test_label = get_test()

# print('Accuracy:', sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

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
