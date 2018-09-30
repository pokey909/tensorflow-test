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

def setupBoards(fen, pieceToMove):
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
            # b.turn = pieceToMove.color
            initialBoards.append(chess.Board(b.fen()))

    # for game in initialBoards:
    #     if game == None:
    #         initialBoard = chess.Board(fen=fen, chess960=False)
    #     else:
    #         initialBoard = game
    #     if len(pieceSquares) == 0:        
    #         print(chess.PIECE_NAMES[pieceToMove.piece_type] + " not on the board. skipping position...")
    #         # free = np.random.choice(list(~chess.SquareSet(initialBoard.occupied)))
    #         # initialBoard.set_piece_at(free, pieceToMove)
    #         # pieceSquares.add(free)

    #     # print(pieceToMove)
    #     # print(pieceSquares)
    #     for pieceSquare in pieceSquares:
    #         board = chess.Board(initialBoard.fen())
    #         freeSquares = ~chess.SquareSet(board.occupied)
    #         for i in freeSquares:
    #             board = initialBoard
    #             board.remove_piece_at(pieceSquare)
    #             board.set_piece_at(i, pieceToMove)
    #             board.turn = pieceToMove.color
    #             boards.append(board)
    # return boards
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

def allMovesForPiece(pieceToMove):
    
    boards = setupBoards(None, pieceToMove)
    features = None
    shapeKnown = False
    index = 0
    print("Extracting features...")
    for b in boards:
        f = fe.extract_features(b)
        if not shapeKnown:
            features = np.zeros(shape=(len(boards),) + f.shape, dtype=f.dtype)
            shapeKnown = True
        features[index,:,:,:] = f
        index = index + 1

    positions = features[:, :, :, 0:(Const.X_black_king + 1)]
    moves = features[:,:,:,Const.X_white_pawns_moves:(Const.X_white_king_moves + 1)]
    print("Written {:d} examples".format(features.shape[0]))
    npy_to_tfrecords(positions, moves, boards, chess.PIECE_NAMES[pieceToMove.piece_type] + ".tfrecord")

for pt in chess.PIECE_TYPES:
    allMovesForPiece(chess.Piece(pt, chess.WHITE))