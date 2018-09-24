#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Jens E. Köhler"

import numpy as np
import tensorflow as tf


def ndarray_to_tfrecords(X, Y, file_path, verbose=True):
	""" Converts a Numpy array (or two Numpy arrays) into a TFRecords format file.
	Description:
		Convert input data provided as numpy.ndarray into TFRecords format file.
	Args:
		X : (numpy.ndarray) of rank N
			Numpy array of M training examples. Its dtype should be float32, float64 or int64.
			X gets reshaped into rank 2, where the first dimension denotes to m (the number of
			examples) and the rest to the dimensions of one example. The shape of one example
			is stored to feature 'x_shape'.
		Y : (numpy.ndarray) of rank N or None
			Numpy array of M labels. Its dtype should be float32, float64, or int64.
			None if there is no label array. Y gets also reshaped into rank 2, similiar to X.
			The shape of one label is stored to feature 'y_shape'
		file_path: (str) path and name of the resulting tfrecord file to be generated
		verbose : (bool) if true, progress is reported.
	Raises:
		ValueError: if input type is not float64, float32 or int64.
	"""
	
	
	def _dtype_feature(ndarray):
		"""match appropriate tf.train.Feature class with dtype of ndarray"""
		assert isinstance(ndarray, np.ndarray)
		dtype_ = ndarray.dtype
		if dtype_ == np.float64 or dtype_ == np.float32:
			return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
		elif dtype_ == np.int64:
			return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
		elif dtype_ == np.uint8:
			return lambda array: tf.train.Feature(bytes_list=tf.train.BytesList(value=array))
		else:
			raise ValueError("The input should be numpy ndarray. Instead got {}".format(ndarray.dtype))
	
	assert isinstance(X, np.ndarray)
	X_flat = np.reshape(X, [X.shape[0], np.prod(X.shape[1:])])
	dtype_X = _dtype_feature(X_flat)
	
	assert isinstance(Y, np.ndarray) or Y is None
	if Y is not None:
		assert X.shape[0] == Y.shape[0]
		Y_flat = np.reshape(Y, [Y.shape[0], np.prod(Y.shape[1:])])
		dtype_Y = _dtype_feature(Y_flat)
	
	# Generate tfrecord writer
	with tf.python_io.TFRecordWriter(file_path) as writer:
		
		if verbose:
			print("Serializing {:d} examples into {}".format(X.shape[0], file_path))
		
		# iterate over each sample and serialize it as ProtoBuf.
		for idx in range(X_flat.shape[0]):
			if verbose:
				print("- write {0:d} of {1:d}".format(idx, X_flat.shape[0]), end="\r")
			
			x = X_flat[idx]
			x_sh = np.asarray(X.shape[1:])
			dtype_xsh = _dtype_feature(x_sh)
			
			if Y is not None:
				y = Y_flat[idx]
				y_sh = np.asarray(Y.shape[1:])
				dtype_ysh = _dtype_feature(y_sh)
			
			d_feature = {}
			d_feature["X"] = dtype_X(x)
			d_feature["x_shape"] = dtype_xsh(x_sh)
			if Y is not None:
				d_feature["Y"] = dtype_Y(y)
				d_feature["y_shape"] = dtype_ysh(y_sh)
			
			features = tf.train.Features(feature=d_feature)
			example = tf.train.Example(features=features)
			serialized = example.SerializeToString()
			writer.write(serialized)
	
	if verbose:
		print("Writing {} done!".format(file_path))


TMP_PATH = "/tmp/testing_ndarray_to_tfrecords.tfrecords"


def _test():
	print("-----\nBEGIN TESTING:\n-----")
	# create some example data
	m = 20
	d1 = 5
	d2 = 4
	
	xx = np.random.randn(m, d1, d2)
	yy = np.random.randn(m, 1, 2)
	
	# save ndarray as TFRecords format file
	ndarray_to_tfrecords(xx, yy, TMP_PATH, verbose=True)
	
	# check if data is stored correctly
	for serialized_example in tf.python_io.tf_record_iterator(TMP_PATH):
		example = tf.train.Example()
		example.ParseFromString(serialized_example)
		x_1 = np.array(example.features.feature["X"].float_list.value)
		s_x = np.array(example.features.feature["x_shape"].int64_list.value)
		y_1 = np.array(example.features.feature["Y"].float_list.value)
		s_y = np.array(example.features.feature["y_shape"].int64_list.value)
		
		print("First original example:\n", xx[0])
		x_1 = np.reshape(x_1, s_x)
		print("First restored example:\n", x_1)
		print("shape of X:", s_x)
		print("\nFirst original label:\n", yy[0])
		y_1 = np.reshape(y_1, s_y)
		print("First restored label:\n", y_1)
		print("shape of Y", s_y)
		
		break
	print("-----\nEND TESTING\n-----")

if __name__ == "__main__":
	# testing and show use case
	_test()