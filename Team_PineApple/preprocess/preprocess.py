# -*- coding: utf-8 -*-

import os
import sys

import csv

path_list = ['raw\\samp_train.csv',
			'raw\\samp_cst_feat.csv',
			'raw\\mrc_info.csv']

def get_data(filename, encoding='utf-8'):
	f = open(filename, 'r', encoding=encoding)
	rdr = csv.reader(f)
	data_list = []
	for line in rdr:
		data_list.append({line[0]: line[1:]})
	data_name = data_list.pop(0)	
	return data_list, data_name


user_mrc, user_mrc_name = get_data(path_list[0])
user_feat, user_feat_name = get_data(path_list[1])
label_mrc, label_mrc_name = get_data(path_list[2], encoding='ansi')

'''
for i in range(2):
	cnt = -1
	print(path_list[i])
	f = open(path_list[i], 'r', encoding='utf-8')
	rdr = csv.reader(f)
	for line in rdr:
		cnt += 1
		for l in line:
			print(l, end=' ')
		print("")
	print("objects length: " + str(cnt) + "\n")
	f.close()

cnt = -1
f = open(path_list[2], 'r')
rdr = csv.reader(f)
for line in rdr:
	cnt += 1
	for l in line:
		print(l, end=' ')
	print("")
print("objects length: " + str(cnt) + "\n")
f.close()
'''
