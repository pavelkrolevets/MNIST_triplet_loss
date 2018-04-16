from __future__ import print_function
import tensorflow as tf
import keras
from keras.datasets import mnist
import numpy as np
from keras import backend as K



if __name__ == '__main__':
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
        img = np.uint8(img * 255)
        img = img / 255
        x_train_real.append(img.reshape((28, 28, 1)))

    x_train_real = np.array(x_train_real)

    checkpoint_directory = './model_logs/'
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_directory)

    graph=tf.Graph()
    with tf.Session(graph=graph) as sess:

        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        # We can verify that we can access the list of operations in the graph
        for op in graph.get_operations():
            print(op.name)
        #keep_prob = graph.get_tensor_by_name('keep_prob:0')
        x = graph.get_tensor_by_name('placehold_x:0')

        # keep_prob = graph.get_tensor_by_name('keep_prob:0')  # dropout probability
        oper_restore = graph.get_tensor_by_name('inference:0')

        embed_train = []
        embed_test = []

        for i in range(x_test.shape[0]):
            number = x_test[i,:]
            #print(number)
            number = np.reshape(number, (1,28,28,1))
            prediction = sess.run(oper_restore, feed_dict={x: number})
            prediction = np.reshape(prediction,(128))
            embed_test.append(prediction)

        embed_test=np.asanyarray(embed_test)

        for i in range(x_train_real.shape[0]):
            number = x_train_real[i,:,:,:]
            #print(number)
            number = np.reshape(number, (1,28,28,1))
            prediction = sess.run(oper_restore, feed_dict={x: number})
            prediction = np.reshape(prediction,(128))
            embed_train.append(prediction)

        embed_train=np.asanyarray(embed_train)



    np.save('./np_embeddings/embeddings_train.npy', embed_train)
    np.save('./np_embeddings/labels_train.npy', y_train)
    np.save('./np_embeddings/embeddings_test.npy', embed_test)
    np.save('./np_embeddings/labels_test.npy', y_test)

    print("Saved")



