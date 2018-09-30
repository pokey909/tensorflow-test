import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import chess
# from  tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.tf.keras.layers import Conv2D, MaxPooling2D

# def read_and_decode(filename_queue):
#   reader = tf.TFRecordReader()
#   _, serialized_example = reader.read(filename_queue)
#   features = tf.parse_single_example(
#       serialized_example,
#       # Defaults are not specified since both keys are required.
#       features={
#           'image_raw': tf.FixedLenFeature([], tf.string),
#           'label': tf.FixedLenFeature([], tf.int64),
#           'height': tf.FixedLenFeature([], tf.int64),
#           'width': tf.FixedLenFeature([], tf.int64),
#           'depth': tf.FixedLenFeature([], tf.int64)
#       })
#   image = tf.decode_raw(features['image_raw'], tf.uint8)
#   label = tf.cast(features['label'], tf.int32)
#   height = tf.cast(features['height'], tf.int32)
#   width = tf.cast(features['width'], tf.int32)
#   depth = tf.cast(features['depth'], tf.int32)
#   return image, label, height, width, depth


# with tf.Session() as sess:
#   filename_queue = tf.train.string_input_producer(["../data/svhn/svhn_train.tfrecords"])
#   image, label, height, width, depth = read_and_decode(filename_queue)
#   image = tf.reshape(image, tf.pack([height, width, 3]))
#   image.set_shape([32,32,3])
#   init_op = tf.initialize_all_variables()
#   sess.run(init_op)
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)
#   for i in range(1000):
#     example, l = sess.run([image, label])
#     print (example,l)
#   coord.request_stop()
#   coord.join(threads)

def load_data(session, fq):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(fq)
    features={
        'X': tf.FixedLenFeature([], tf.string),
        'Y': tf.FixedLenFeature([], tf.string),
        'width': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'planes': tf.FixedLenFeature([], tf.int64),
        'fen': tf.FixedLenFeature([], tf.string)
    }              
    parsed = tf.parse_single_example(serialized_example, features=features)
    X = tf.cast(tf.decode_raw(parsed['X'], tf.int64), tf.float32)
    Y = tf.cast(tf.decode_raw(parsed['Y'], tf.int64), tf.float32)
    fen = parsed['fen']
    # w, h, planes = sess.run([parsed['width'], parsed['height'], parsed['planes']])

    X = tf.reshape(X, [-1, 8, 8, 12])
    Y = tf.reshape(Y, [-1, 8, 8, 6])
    Y = tf.reshape(Y, [-1, 6])
    # X, Y, FEN = tf.train.shuffle_batch([X, Y, fen], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    # return X, Y, FEN
    return X,Y,fen


def makeModel():
    input_shape = (8, 8, 12)
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    # Add another:
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    # Add a softmax layer with 10 output units:
    model.add(keras.layers.Dense(6, activation='relu'))
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                loss='mse',
                metrics=['accuracy'])
    return model

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  keras.callbacks.EarlyStopping(patience=20000, monitor='acc'),
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./logs')
]    
with tf.Session(config=config)  as sess:

    tf.logging.set_verbosity(tf.logging.DEBUG)
    filename = "pawn.tfrecord"
    filename_queue = tf.train.string_input_producer([filename], num_epochs=40)
    x,y,fen = load_data(sess, filename_queue)
   # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    X, Y, FEN = sess.run([x, y, fen])
    # dataset = tf.data.Dataset.from_tensor_slices((X,Y))
    # dataset = dataset.batch(32)
    # dataset = dataset.repeat()
    model = makeModel()
    model.fit(X,Y, epochs=30000, batch_size=32, callbacks=callbacks, verbose=1)#, steps_per_epoch=30)
    model.evaluate(X,Y, steps=30000)#, steps_per_epoch=30)
    # Create a coordinator and run all QueueRunner objects
    # for ep in range(6):
    #     for batch_index in range(10):
    #         X, Y, FEN = sess.run([x, y, fen])
    #         Y = Y.astype(np.uint8)
    #         for j in range(18):
    #             idx = (j % 6) + 1
    #             plt.subplot(6, 3, j+1)
    #             plt.subplots_adjust(hspace=0.5)
    #             piece = chess.PIECE_NAMES[idx]
    #             if j < 6:
    #                 plt.imshow(Y[batch_index, :,:,j].T)
    #                 # plt.imshow(Y[:,:,j].T)
    #             else:
    #                 plt.imshow(X[batch_index, :,:,j - 6].T, cmap='summer')
    #                 # plt.imshow(X[:,:,j - 6].T, cmap='summer')
    #             plt.xticks(np.arange(8), ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'))
    #             plt.yticks(np.arange(8), np.arange(1, 9))
    #             plt.title(piece)
    #             plt.gca().invert_yaxis()
    #             plt.grid(True)
    #         plt.draw()
    #         print(chess.Board(fen=FEN[batch_index].decode('utf-8')).unicode(invert_color=True, borders=False))
    #         print("------------")
    #         plt.waitforbuttonpress(0)
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()