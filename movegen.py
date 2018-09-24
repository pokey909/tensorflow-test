import chess
import os
import sys

import tensorflow as tf
import features as fe
import random
import data
import numpy as np
import Const

FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# [Squares[1..64]]
def boardToVector(initialPosition, legalMovesBoard):
    features = fe.extract_features(initialPosition)
    feature = {
        'train/label': _int64_feature(1),
        'train/feature': _int64_feature(builder)
    }

    return builder

# Squares[1..64], side_to_move, player_color
def boardToString(board):
    return ",".join(map(str, boardToVector(board)))

def setupBoards(fen, pieceToMove):
    boards = []
    
    initialBoard = chess.Board(fen=fen, chess960=False)
    pieceSquares = initialBoard.pieces(pieceToMove.piece_type, pieceToMove.color)
    if len(pieceSquares) == 0:
        print("PieceToMove not on the board. Adding it...")
        initialBoard.set_piece_at(chess.A1, pieceToMove)
        pieceSquares.add(chess.A1)

    for pieceSquare in pieceSquares:
        board = initialBoard
        board.remove_piece_at(pieceSquare)
        freeSquares = ~chess.SquareSet(board.occupied)
        for i in freeSquares:
            board = chess.Board(fen=fen, chess960=False)
            board.set_piece_at(i, pieceToMove)
            board.turn = pieceToMove.color
            boards.append(board)
    return boards

def allMovesForPiece(features, pieceToMove):
    boards = setupBoards(None, pieceToMove)
    features = None
    shapeKnown = False
    index = 0
    print("Extracting features...")
    for b in boards:
        f = fe.extract_features(b)
        if not shapeKnown:
            features = np.zeros(shape=(len(boards),) + f.shape)
            shapeKnown = True
        features[index] = f
        index = index + 1

    positions = features[:, :, :, 0:(Const.X_black_king + 1)]
    moves = features[:,:,:,Const.X_white_pawns_moves:(Const.X_white_king_moves + 1)]
    data.ndarray_to_tfrecords(positions, moves, chess.PIECE_NAMES[pieceToMove.piece_type] + ".tfrecord")

for pt in chess.PIECE_TYPES:
    allMovesForPiece(None, chess.Piece(pt, chess.WHITE))