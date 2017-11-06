#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-11-06 14:44:00
'''
import numpy as np

def calculateEntropy(label_list):
	'''
	calculate entropy for a given label list	
	'''
	total_cnt = len(label_list)

	if total_cnt == 0:
		return 0

	label2cnt = {}
	for cand_label in label_list:
		label2cnt.setdefault(cand_label,0)
		label2cnt[cand_label] += 1
	
	entropy = 0.0
	for cand_label in label2cnt:
		prob = float(label2cnt[cand_label]) / total_cnt

		entropy -= prob * np.log2(prob)

	return entropy


def getMatchedDataSet(X,y,selected_feature_idx,selected_feature_value):
	'''
	get all x without selected_feature whose selected_feature value is equal to selected_feature_value	
	X: np.ndarray
	y: np.ndarray
	'''
	feature_cnt = X.shape[1]

	new_feature_idx_list = [idx for idx in range(feature_cnt) if idx != selected_feature_idx]

	X_without_selected_feature = X[:,new_feature_idx_list]

	matched_data_idx = []
	for idx in range(X.shape[0]):
		if X[idx][selected_feature_idx] == selected_feature_value:
			matched_data_idx.append(idx)

	match_X = X_without_selected_feature[matched_data_idx]
	match_y = y[matched_data_idx]

	return match_X,match_y

def getMajorlabel(label_list):
	label2cnt ={}
	for cand_label in label_list:
		label2cnt.setdefault(cand_label,0)
		label2cnt[cand_label] += 1

	max_cnt = 0
	major_label = None
	for cand_label in label2cnt:
		cand_cnt = label2cnt[cand_label]
		if max_cnt < cand_cnt:
			max_cnt = cand_cnt
			major_label = cand_label

	return major_label

	

