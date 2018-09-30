import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import chess

def load_data():
    Y = np.load('label_8x8_pawn.npy')
    X = np.load('train_8x8x12.npy')
    Y = tf.reshape(Y, [Y.shape[0], -1])
    # X, Y, FEN = tf.train.shuffle_batch([X, Y, fen], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    # return X, Y, FEN
    Xval = X[0:4999, :, :, :]
    Yval = Y[0:4999, :]
    X = X[5000:, :, :, :]
    Y = Y[5000:, :]
    return X,Y,Xval, Yval


def makeModel(X,Y):
    input_shape = (X.shape[1], X.shape[2], X.shape[3])
    print(input_shape)
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    # Add another:
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
    # model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1))
    # model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dropout(0.5))

    # Add a softmax layer with 10 output units:
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.compile(optimizer=keras.optimizers.SGD(momentum=0.1, decay=0.1, nesterov=True),
                loss='mse',
                metrics=['accuracy'])
    return model


config = tf.ConfigProto(device_count = {'GPU': 0})
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  keras.callbacks.EarlyStopping(patience=200, monitor='acc'),
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./logs')
]  
  
with tf.Session(config=config)  as sess:
   # Initialize all global and local variables
    tf.logging.set_verbosity(tf.logging.DEBUG)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    x,y,xval, yval = load_data()

    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(32).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((xval, yval))
    val_dataset = val_dataset.batch(32).repeat()

    model = makeModel(x,y)
    model.fit(dataset, 
        epochs=3000, 
        steps_per_epoch=300,
        callbacks=callbacks, 
        verbose=1,
        validation_data=val_dataset,
        validation_steps=3
        )#, steps_per_epoch=30)
    model.evaluate(dataset, steps=30000)#, steps_per_epoch=30)

    sess.close()