# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

import tensorflow as tf

path_list = ['raw\\samp_train.csv',
			'raw\\samp_cst_feat.csv',
			'raw\\mrc_info.csv']

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from preprocess import preprocess

def predict_cst():
	user_mrc, user_mrc_name = preprocess.get_data(path_list[0])
	user_feat, user_feat_name = preprocess.get_data(path_list[1])
	label_mrc, label_mrc_name = preprocess.get_data(path_list[2], encoding='ansi')
	print(user_mrc_name)
	print(user_feat_name)
	print(label_mrc_name)

	train_datas = []
	for train_idx in range(0, 9000):
		for key in list(user_feat[train_idx].keys()):
			train_datas.append(user_feat[train_idx][key])
	train_feats_tensor = tf.convert_to_tensor(np.array(train_datas, dtype=np.float32), dtype=tf.float32, name="customer_features")
	train_feats_tensor = tf.reshape(train_feats_tensor, [int(train_feats_tensor.shape[0]), int(train_feats_tensor.shape[1]), 1])
	print(train_feats_tensor) # (9000, 226, 1)

	train_labels = []
	for label_idx in range(0, 9000):
		for key in list(user_mrc[label_idx].keys()):
			train_labels.extend(user_mrc[label_idx][key])
	train_labels_tensor = tf.convert_to_tensor(np.array(train_labels, dtype=np.int8), dtype=tf.int8, name="market_labels")
	print(train_labels_tensor) # (9000, )


	test_datas = []
	for test_idx in range(9000, len(user_feat)):
		for key in list(user_feat[test_idx].keys()):
			test_datas.append(user_feat[test_idx][key])
	test_feats_tensor = tf.convert_to_tensor(np.array(test_datas, dtype=np.float32), dtype=tf.float32, name="test_customer_features")
	test_feats_tensor = tf.reshape(test_feats_tensor, [int(test_feats_tensor.shape[0]), int(test_feats_tensor.shape[1]), 1])
	print(test_feats_tensor) # (1124, 226, 1)

	test_labels = []
	for test_label_idx in range(9000, len(user_mrc)):
		for key in list(user_mrc[test_label_idx].keys()):
			test_labels.extend(user_mrc[test_label_idx][key])
	test_labels_tensor = tf.convert_to_tensor(np.array(test_labels, dtype=np.int8), dtype=tf.int8, name="test_market_labels")
	print(test_labels_tensor) # (1124,)


	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(226, 1)),
		tf.keras.layers.MaxPooling1D(pool_size=2),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(50, activation='relu'),
		tf.keras.layers.Dense(1)
		])
	
	model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

	model.fit(train_feats_tensor, train_labels_tensor, epochs=1000)

	model.evaluate(test_feats_tensor, test_labels_tensor, verbose=0)

if __name__ == '__main__':
	predict_cst()