import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import chess
import keras.backend.tensorflow_backend as tfb

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

def load_data():
    Yy = np.load('label_8x8_pawn.npy').astype(np.int16)
    Y = Yy.reshape(Yy.shape[0], -1)
    X = np.load('train_8x8x12.npy').astype(np.float32)
    print(X.shape)
    print(Yy.shape)

    # Y = np.apply_along_axis(lambda x: keras.utils.to_categorical(x, num_classes=64), 1, Yy.reshape(Yy.shape[0], -1))
    # Y = tf.reshape(Y, [Y.shape[0], -1])
    # X, Y, FEN = tf.train.shuffle_batch([X, Y, fen], batch_size=10, capacity=30, num_threads=1, min_after_dequeue=10)
    # return X, Y, FEN
    Xval = X[0:4999, :, :, :].astype(np.float32)
    Yval = Y[0:4999, :]
    X = X[5000:, :, :, :].astype(np.float32)
    Y = Y[5000:, :]
    print("Y: ", Y.shape)
    return X,Y,Xval, Yval


def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                      logits=output
                                                    )
    return tf.reduce_mean(loss, axis=-1)

def makeModel(X,Y):
    input_shape = (X.shape[1], X.shape[2], X.shape[3])
    print(input_shape)
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    # Add another:
    # model.add(keras.layers.Conv2D(768, kernel_size=(3, 3),
    #              activation='relu',
    #              input_shape=input_shape))
    # # model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1))
    # model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    # model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=1))
    # # model.add(keras.layers.Dropout(0.25))
    # model.add(keras.layers.Flatten())
    # # model.add(keras.layers.Dropout(0.5))

    # # Add a softmax layer with 10 output units:
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dense(128, activation='relu'))

    # model.add(keras.layers.Dense(8*12, activation='relu'))
    # model.add(keras.layers.Dense(8*12, activation='relu'))
    # model.add(keras.layers.Dense(8*12, activation='relu'))
    # model.add(keras.layers.Flatten())
    
    # model.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    # model.add(keras.layers.MaxPooling2D((1, 1)))
    # model.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    # model.add(keras.layers.MaxPooling2D((1, 1)))
    # model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(keras.layers.MaxPooling2D((1, 1)))
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(8*8*12, activation='relu'))
    model.add(keras.layers.Dense(600, activation='relu'))
    model.add(keras.layers.Dense(8*8*12, activation='relu'))
    model.add(keras.layers.Dense(600, activation='relu'))
    model.add(keras.layers.Dense(8*8*12, activation='relu'))
    model.add(keras.layers.Dense(64, activation='softmax'))
    print(model.output_shape)
    model.compile(optimizer=keras.optimizers.RMSprop(),
                loss='binary_crossentropy', #$categorical_crossentropy',
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

    model = makeModel(x,y)

    # dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # dataset = dataset.batch(32).repeat()
    # val_dataset = tf.data.Dataset.from_tensor_slices((xval, yval))
    # val_dataset = val_dataset.batch(32).repeat()

    # model.fit(dataset, 
    #     epochs=3000, 
    #     steps_per_epoch=1000,
    #     callbacks=callbacks, 
    #     verbose=1,
    #     validation_data=val_dataset,
    #     validation_steps=3
    #     )#, steps_per_epoch=30)

    model.fit(x,y, epochs=3000, shuffle=False, batch_size=500, validation_data=(xval,yval), callbacks=callbacks)
    model.evaluate(dataset, steps=30000)#, steps_per_epoch=30)

    sess.close()