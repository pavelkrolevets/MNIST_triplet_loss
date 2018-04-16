from __future__ import print_function
import tensorflow as tf
from keras.datasets import mnist
import numpy as np
from keras import backend as K

#MODEL
# Architecture
def inference(x):
    phase_train = tf.constant(True)
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.layers.conv2d(inputs=x, filters=32,  kernel_size=[7, 7], padding="same", activation=tf.nn.relu)
    norm1 = tf.layers.batch_normalization(conv1)
    pool1 = tf.layers.max_pooling2d(inputs=norm1, pool_size=[2, 2], strides=2)

    conv2a = tf.layers.conv2d(inputs=pool1, filters=32, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv2a, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    norm2 = tf.layers.batch_normalization(conv2)
    pool2 = tf.layers.max_pooling2d(inputs=norm2, pool_size=[2, 2], strides=2)

    conv3a = tf.layers.conv2d(inputs=pool2, filters=32, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv3a, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    conv4a = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[1, 1], padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv4a, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    flat = tf.reshape(conv4, [-1, 7 * 7 * 64])

    fc_1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu)

    embed = tf.layers.dense(inputs=fc_1, units=128)

    output = tf.nn.l2_normalize(embed, dim=1, epsilon=1e-12, name=None)

    return output

def loss(output, labels):
    triplet = tf.contrib.losses.metric_learning.triplet_semihard_loss(labels, output, margin=margin)
    loss = tf.reduce_mean(triplet, name='triplet')
    return loss

def training(cost, global_step):
    tf.summary.scalar("cost", cost)
    learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op



if __name__ == '__main__':
    # Parameters
    training_epochs = 1
    display_step = 1
    batch_size = 128
    margin = 1.0
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    x_train_real = []
    for i in range(int(60000)):
        img = x_train[i].reshape((28, 28))
        x_train_real.append(img.reshape((28, 28, 1)))

    x_train_real = np.array(x_train_real)

    x = tf.placeholder("float", [None, 28, 28, 1], name='placehold_x')
    y = tf.placeholder('float', [None], name='placehold_y')

    output = inference(x)
    tf.identity(output, name="inference")

    cost = loss(output, y)
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = training(cost, global_step)
    saver = tf.train.Saver()

    """Making iterator"""
    features_placeholder = tf.placeholder(x_train_real.dtype, x_train_real.shape)
    labels_placeholder = tf.placeholder(y_train.dtype, y_train.shape)

    training_dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
    batched_dataset = training_dataset.batch(batch_size)

    training_init_op = batched_dataset.make_initializable_iterator()
    next_element = training_init_op.get_next()

    """Training"""
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    with tf.device('/cpu:0'):
        with sess.as_default():

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch_train = int(x_train.shape[0] / batch_size)
                sess.run(training_init_op.initializer, feed_dict={features_placeholder: x_train_real,
                                                                  labels_placeholder: y_train})
                # Loop over all batches
                for i in range(total_batch_train):
                    # Fit training using batch data
                    batch_x, batch_y = sess.run(next_element)

                    sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
                    # Compute average loss
                    avg_cost = sess.run(cost, feed_dict={x: batch_x, y: batch_y})

                    if not i % 10:
                        print('Epoch #: ', epoch, 'global step', sess.run(global_step), '  Batch #: ', i, 'loss: ',
                              avg_cost)

                saver.save(sess, "model_logs/model-checkpoint", global_step=global_step, write_meta_graph=True)

            print("Optimization Finished!")