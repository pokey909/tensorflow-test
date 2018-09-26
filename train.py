import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import chess

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
    X = tf.decode_raw(parsed['X'], tf.int64)
    Y = tf.decode_raw(parsed['Y'], tf.int64)
    fen = parsed['fen']
    # w, h, planes = sess.run([parsed['width'], parsed['height'], parsed['planes']])

    X = tf.reshape(X, [8, 8, 12])
    Y = tf.reshape(Y, [8, 8, 6])

    X, Y, FEN = tf.train.shuffle_batch([X, Y, fen], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    return X, Y, FEN
    # return X,Y,fen

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
with tf.Session(config=config)  as sess:

    filename = "bishop.tfrecord"
    filename_queue = tf.train.string_input_producer([filename], num_epochs=40)
    x,y,fen = load_data(sess, filename_queue)

   # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for ep in range(6):
        for batch_index in range(10):
            X, Y, FEN = sess.run([x, y, fen])
            Y = Y.astype(np.uint8)
            for j in range(18):
                idx = (j % 6) + 1
                plt.subplot(6, 3, j+1)
                plt.subplots_adjust(hspace=0.5)
                piece = chess.PIECE_NAMES[idx]
                if j < 6:
                    plt.imshow(Y[batch_index, :,:,j].T)
                    # plt.imshow(Y[:,:,j].T)
                else:
                    plt.imshow(X[batch_index, :,:,j - 6].T, cmap='summer')
                    # plt.imshow(X[:,:,j - 6].T, cmap='summer')
                plt.xticks(np.arange(8), ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'))
                plt.yticks(np.arange(8), np.arange(1, 9))
                plt.title(piece)
                plt.gca().invert_yaxis()
                plt.grid(True)
            plt.draw()
            print(chess.Board(fen=FEN[batch_index].decode('utf-8')).unicode(invert_color=True, borders=False))
            print("------------")
            plt.waitforbuttonpress(0)
    # Stop the threads
    coord.request_stop()
    
    # Wait for threads to stop
    coord.join(threads)
    sess.close()