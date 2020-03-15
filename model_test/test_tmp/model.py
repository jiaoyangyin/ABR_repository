import tensorflow as tf
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BATCH_SIZE = 40
REGULARIZATION_RATE = 0.005
LEARNING_RATE_BASE = 0.0001
TRAINING_STEPS = 100000000
DATA_SIZE = 8560
DATA_SIZE_VALID = 3200

Fc1_size = 100
OUTPUT_NODE = 4

filter_width = 2
in_width = 40
in_channels = 1
out_channels = 5

x_train = np.loadtxt(open("./dataset/train_data.csv", "rb"),
                     delimiter=",", skiprows=0)
y_train = np.loadtxt(open("./dataset/train_label.csv", "rb"),
                     delimiter=",", skiprows=0)
x_valid = np.loadtxt(open("./dataset/validation_data.csv", "rb"),
                     delimiter=",", skiprows=0)
y_valid = np.loadtxt(open("./dataset/validation_label.csv", "rb"),
                     delimiter=",", skiprows=0)

shuffle_id = np.arange(x_train.shape[0])
np.random.shuffle(shuffle_id)
train = x_train[shuffle_id]
y_train = y_train[shuffle_id]

valid_data = np.hstack((x_valid, y_valid))
np.random.shuffle(valid_data)
np.random.shuffle(valid_data)
np.random.shuffle(valid_data)
valid = valid_data[:, 0:40]
y_valid = valid_data[:, 40:44]

for i in range(40):
    x_trainmax, x_trainmin = train[:, i].max() + 0.1, train[:, i].min() - 0.1
    if i == 0:
        x_train = (train[:, i] - x_trainmin) / (x_trainmax - x_trainmin)
    elif i > 0:
        train_normal = (train[:, i] - x_trainmin) / (x_trainmax - x_trainmin)
        x_train = np.c_[x_train, train_normal]

for i in range(40):
    x_validmax, x_validmin = valid[:, i].max() + 0.1, valid[:, i].min() - 0.1
    if i == 0:
        x_valid = (valid[:, i] - x_validmin) / (x_validmax - x_validmin)
    elif i > 0:
        valid_normal = (valid[:, i] - x_validmin) / (x_validmax - x_validmin)
        x_valid = np.c_[x_valid, valid_normal]

shuffle_id = np.arange(x_train.shape[0])
np.random.shuffle(shuffle_id)
x_train = x_train[shuffle_id]
y_train = y_train[shuffle_id]

x_valid = x_train[0:800]
y_valid = y_train[0:800]

x_train = x_train[800:11760]
y_train = y_train[800:11760]

shuffle_id = np.arange(x_train.shape[0])
np.random.shuffle(shuffle_id)
x_train = x_train[shuffle_id]
y_train = y_train[shuffle_id]

x_valid = np.r_[x_valid, x_train[0:800]]
y_valid = np.r_[y_valid, y_train[0:800]]

x_train = x_train[800:10960]
y_train = y_train[800:10960]

shuffle_id = np.arange(x_train.shape[0])
np.random.shuffle(shuffle_id)
x_train = x_train[shuffle_id]
y_train = y_train[shuffle_id]

x_valid = np.r_[x_valid, x_train[0:800]]
y_valid = np.r_[y_valid, y_train[0:800]]

x_train = x_train[800:10160]
y_train = y_train[800:10160]

shuffle_id = np.arange(x_train.shape[0])
np.random.shuffle(shuffle_id)
x_train = x_train[shuffle_id]
y_train = y_train[shuffle_id]

x_valid = np.r_[x_valid, x_train[0:800]]
y_valid = np.r_[y_valid, y_train[0:800]]

x_train = x_train[800:9360]
y_train = y_train[800:9360]

def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    global moving_mean, moving_variance, mean, variance, count
    with tf.variable_scope(name, default_name='BatchNorm'):
        shape = x.get_shape().as_list()
        shape1 = shape
        shape = shape[-1:]

        moving_mean = tf.get_variable('mean', shape=shape, initializer=tf.zeros_initializer, trainable=False)
        moving_variance = tf.get_variable('variance', shape=shape, initializer=tf.ones_initializer, trainable=False)

        def mean_var_with_update():
            axis = list(range(len(shape1) - 1))
            mean, variance = tf.nn.moments(x, axis, name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)

        xx, yy = tf.cond(train, lambda: (3, 4), lambda: (4, 3))

        mean, variance = tf.cond(xx < yy, mean_var_with_update, lambda: (moving_mean, moving_variance))

        if affine:
            beta = tf.get_variable('beta', shape, initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', shape, initializer=tf.ones_initializer)

            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)

        return x


def inference(input_tensor, is_train):
    tf.summary.histogram("input_tensor", input_tensor)
    with tf.variable_scope('conv1'):
        conv1_weights = tf.get_variable("weight", [filter_width, in_channels, out_channels],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [out_channels],
                                       initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv1d(input_tensor, conv1_weights, stride=1, padding='SAME')
        #conv1 = batch_norm(conv1, is_train)
        conv1 = tf.layers.batch_normalization(conv1, training=is_train)
        output1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
        #tf.summary.image("conv1_output", output1[:, :, :, 0:1], 10)

    with tf.name_scope('pool1'):
        pool1 = tf.layers.max_pooling1d(output1, pool_size=2, strides=2, padding='SAME')
        #tf.summary.image("pool1_output", pool1[:, :, :, 0:1], 10)

    with tf.variable_scope('conv2'):
        conv2_weights = tf.get_variable("weight", [filter_width, out_channels, out_channels],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [out_channels],
                                       initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv1d(pool1, conv2_weights, stride=1, padding='SAME')
        #conv2 = batch_norm(conv2, is_train)
        conv2 = tf.layers.batch_normalization(conv2, training=is_train)
        output2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
        #tf.summary.image("conv2_output", output2[:, :, :, 0:1], 10)

    with tf.name_scope('pool2'):
        pool2 = tf.layers.max_pooling1d(output2, pool_size=2, strides=2, padding='SAME')


    with tf.variable_scope('conv3'):
        conv3_weights = tf.get_variable("weight", [filter_width, out_channels, out_channels],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [out_channels],
                                       initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv1d(pool2, conv3_weights, stride=1, padding='SAME')
        #conv2 = batch_norm(conv2, is_train)
        conv3 = tf.layers.batch_normalization(conv3, training=is_train)
        output3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope('pool3'):
        pool3 = tf.layers.max_pooling1d(output3, pool_size=2, strides=2, padding='SAME')


    with tf.variable_scope('conv4'):
        conv4_weights = tf.get_variable("weight", [filter_width, out_channels, out_channels],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [out_channels],
                                       initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv1d(pool3, conv4_weights, stride=1, padding='SAME')
        #conv2 = batch_norm(conv2, is_train)
        conv4 = tf.layers.batch_normalization(conv4, training=is_train)
        output4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope('pool4'):
        pool4 = tf.layers.max_pooling1d(output4, pool_size=2, strides=2, padding='SAME')


    #with tf.name_scope('pool2'):
        #pool2 = tf.nn.max_pool(output2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #tf.summary.image("pool2_output", pool2[:, :, :, 0:1], 10)

    shape = pool4.get_shape().as_list()
    #print(shape)
    input_node = shape[1] * shape[2]
    reshape = tf.reshape(pool4, [shape[0], input_node])

    with tf.variable_scope('fc1'):
        #fc1_weights = tf.get_variable("weight", [input_node, Fc1_size],
        #                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        #fc1_biases = tf.get_variable("bias", [Fc1_size], initializer=tf.constant_initializer(0.1))
        #fc1 = tf.nn.relu(batch_norm(tf.matmul(reshape, fc1_weights) + fc1_biases, train))
        #fc1 = tf.matmul(reshape, fc1_weights) + fc1_biases
        #fc1 = batch_norm(fc1, is_train)
        #fc1 = tf.layers.batch_normalization(fc1, training=is_train)
        #fc1 = tf.nn.relu(fc1)
        fc1 = tf.layers.dense(inputs=reshape, units=Fc1_size, activation=None,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE),
                              bias_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE),
                              activity_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE))
        fc1 = tf.layers.batch_normalization(fc1, training=is_train)
        fc1 = tf.nn.relu(fc1)
        #fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('fc2'):
        #fc2_weights = tf.get_variable("weight", [Fc1_size, OUTPUT_NODE],
        #                              initializer=tf.truncated_normal_initializer(stddev=0.1))
        #fc2_biases = tf.get_variable("bias", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        #fc2 = tf.nn.relu(batch_norm(tf.matmul(fc1, fc2_weights) + fc2_biases, train))
        #fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases
        #fc2 = batch_norm(fc2, is_train)
        #fc2 = tf.layers.batch_normalization(fc2, training=is_train)
        #fc2 = tf.nn.relu(fc2)
        fc2 = tf.layers.dense(inputs=fc1, units=Fc1_size, activation=None,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE),
                              bias_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE),
                              activity_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE))
        fc2 = tf.layers.batch_normalization(fc2, training=is_train)
    
    with tf.variable_scope('fc3'):
        fc3 = tf.layers.dense(inputs=fc1, units=OUTPUT_NODE, activation=None,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE),
                              bias_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE),
                              activity_regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE))

    return fc3


def train():
    x = tf.placeholder(tf.float32, shape=(BATCH_SIZE, in_width, in_channels), name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    is_train = tf.placeholder(tf.bool, name='is_train')
    y = tf.nn.softmax(inference(x, is_train))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        global training_data, train_data, x_train, y_train, x_valid, y_valid, count

        saver.restore(sess, "./model.ckpt")

        x_train1 = np.reshape(x_train, (-1, in_width, in_channels))
        x_valid1 = np.reshape(x_valid, (-1, in_width, in_channels))

        valid_acc = 0
 
        for j in range(int(np.shape(x_valid1)[0] / BATCH_SIZE)):
            start1 = j * BATCH_SIZE % int(np.shape(x_valid1)[0])
            end1 = min(start1 + BATCH_SIZE, int(np.shape(x_valid1)[0]))
            tmp_valid = sess.run(accuracy, feed_dict={x: np.reshape(x_valid[start1:end1], (BATCH_SIZE, in_width, in_channels)),
                                                      y_: y_valid[start1:end1], is_train: False})
            valid_acc = valid_acc + tmp_valid

        valid_acc = valid_acc / (np.shape(x_valid1)[0] / BATCH_SIZE)

        print("valid accuracy ""is %g." % valid_acc)  

def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()

