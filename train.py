import tensorflow as tf
import numpy as np

def load_data():
    # Transforms a scalar string `example_proto` into a pair of a scalar string and
    # a scalar integer, representing an image and its label, respectively.
    def _parse_function(example_proto):
        #   features = {"train/label": tf.FixedLenFeature((), tf.string, default_value=""),
        #               "train/feature": tf.FixedLenFeature((), tf.int32, default_value=0)}
        features={
            'X': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'x_shape': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'Y': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
            'y_shape': tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64))
        }              
        # example = tf.train.Example()
        # example.ParseFromString(example_proto)
        parsed = tf.parse_single_example(example_proto, features)
                
        X = tf.reshape(parsed["X"], (8,8,12))
        Y = tf.reshape(parsed["Y"], (8,8,12))
        return {"positions": X}, {"moves": Y}

    # Creates a dataset that reads all of the examples from two files.
    filenames = ["king.tfrecord"]
    dat = tf.data.TFRecordDataset(filenames).map(_parse_function).shuffle(buffer_size=10000).repeat(10).batch(32)

    iterator = dat.make_one_shot_iterator()

    # # `features` is a dictionary in which each value is a batch of values for
    # # that feature; `labels` is a batch of labels.
    # features, labels = iterator.get_next()
    return iterator



for serialized_example in tf.python_io.tf_record_iterator("king.tfrecord"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    x_1 = np.array(example.features.feature["X"].float_list.value)
    s_x = np.array(example.features.feature["x_shape"].int64_list.value)
    y_1 = np.array(example.features.feature["Y"].float_list.value)
    s_y = np.array(example.features.feature["y_shape"].int64_list.value)
    
    x_1 = np.reshape(x_1, s_x)
    print("First restored example:\n", x_1)
    print("shape of X:", s_x)

    y_1 = np.reshape(y_1, s_y)
    print("First restored label:\n", y_1)
    print("shape of Y", s_y)

iterator = load_data()

with tf.Session()  as sess:
    # Compute for 100 epochs.
    for _ in range(100):
        # sess.run(iterator.initializer)
        while True:
            try:
                sess.run(iterator.get_next())
            except tf.errors.OutOfRangeError:
                break
