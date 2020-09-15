# -*- coding: utf-8 -*-

import os
import sys

import csv

path_list = ['raw\\samp_train.csv',
			'raw\\samp_cst_feat.csv',
			'raw\\mrc_info.csv']

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
