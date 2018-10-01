import chess
import os
import sys

import tensorflow as tf
import features as fe
import random
import data
import numpy as np
import Const
import chess.pgn

FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def saveExample(features):
#     X_flat = np.reshape(X, [X.shape[0], np.prod(X.shape[1:])])

def setupBoards(fen):
    boards = []
    pgn = open('Ashley.pgn')
    
    initialBoards=[]
    while True:
        g = chess.pgn.read_game(pgn)
        if g == None:
            break
        b = chess.Board()
        initialBoards.append(chess.Board())
        for move in g.main_line():
            b.push(move)
            initialBoards.append(chess.Board(b.fen()))
    return initialBoards

def npy_to_tfrecords(x, y, boards, output_file):
    # write records to a tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(x.shape[0]):
        # Feature contains a map of string to feature proto objects
        X = np.squeeze(x[i,:,:,:])
        Y = np.squeeze(y[i,:,:,:])

        feature = {}
        feature['X'] = _bytes_feature(tf.compat.as_bytes(X.tostring()))
        feature['Y'] = _bytes_feature(tf.compat.as_bytes(Y.tostring()))
        feature['width'] = _int64_feature(X.shape[0])
        feature['height'] = _int64_feature(X.shape[1])
        feature['planes'] = _int64_feature(X.shape[2])
        feature['fen'] = _bytes_feature(tf.compat.as_bytes(boards[i].fen()))
        # Construct the Example proto object
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize the example to a string
        serialized = example.SerializeToString()
        # write the serialized objec to the disk
        writer.write(serialized)
    writer.close()

def genTrainingData():
    boards = setupBoards(None)
    features = None
    shapeKnown = False
    index = 0
    print("Extracting features...")
    for b in boards:
        f = fe.extract_features(b).astype(np.int16)
        if not shapeKnown:
            features = np.zeros(shape=(len(boards),) + f.shape, dtype=f.dtype)
            shapeKnown = True
        features[index,:,:,:] = f
        index = index + 1

    positions = features[:, :, :, 0:(Const.X_black_king + 1)]
    moves = features[:,:,:,Const.X_white_pawns_moves:(Const.X_white_king_moves + 1)]
    return positions, moves

X,Y = genTrainingData()
np.save('train_8x8x12.npy', X)
for pt in chess.PIECE_TYPES:
    piece = pt - 1
    labelForPieceTensor = np.squeeze(Y[:,:,:,piece])
    filename = 'label_8x8_' + chess.PIECE_NAMES[pt] + ".npy"
    np.save(filename, labelForPieceTensor)
    print("Written {:d} labels for {:s}".format(labelForPieceTensor.shape[0], chess.PIECE_NAMES[pt]))
