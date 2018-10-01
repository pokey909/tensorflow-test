import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import chess

def load_data():
    Yy = np.load('label_8x8_pawn.npy').astype(np.float32)
    Y = Yy.reshape(Yy.shape[0], -1)
    X = np.load('train_8x8x12.npy').astype(np.float32)
    Xval = X[0:4999, :, :, :].astype(np.float32)
    Yval = Y[0:4999, :]
    X = X[5000:, :, :, :].astype(np.float32)
    Y = Y[5000:, :]
    return X,Y,Xval, Yval

def makeModel(X,Y):
    input_shape = (X.shape[1], X.shape[2], X.shape[3])
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=input_shape))
    model.add(keras.layers.Dense(8*8*12, activation='relu'))
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(8*8, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.01, amsgrad=True, decay = 0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model


config = tf.ConfigProto(device_count = {'GPU': 1})
callbacks = [
  # Interrupt training if `acc` stops improving for over 50 epochs
  keras.callbacks.EarlyStopping(patience=50, monitor='acc', mode='max', min_delta=0.0001),
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./logs/21')
]  
  
with tf.Session(config=config)  as sess:
   # Initialize all global and local variables
    tf.logging.set_verbosity(tf.logging.DEBUG)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    x,y,xval, yval = load_data()

    model = makeModel(x,y)

    model.fit(x,y, epochs=3000, shuffle=True, batch_size=64, validation_data=(xval,yval), callbacks=callbacks)
    print(model.evaluate(x,y, steps=300))#, steps_per_epoch=30)
    model.predict_classes(xval, batch_size=32, verbose=1)
    sess.close()