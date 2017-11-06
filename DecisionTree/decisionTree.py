#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-11-06 15:04:52
'''
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import numpy as np
import pickle

from baseUtil import calculateEntropy
from baseUtil import getMatchedDataSet
from baseUtil import getMajorlabel


class NotFitError(Exception):
	'''
	'''
	pass

class DecisionTree(object):
	def __init__(self,mode = 'ID3'):
		self._tree = None

		if mode == 'ID3' or mode == 'C4.5':
			self._mode = mode
		else:
			raise Exception('model should be ID3 or C4.5. given value is :' + str(mode))
	
	def getBestSplitFeatureWithID3(self,X,y):
		'''
		get best split feature using info gain	
		'''
		feature_cnt = X.shape[1]

		entropy_before_split = calculateEntropy(y)

		best_info_gain = 0
		best_feature_idx = -1

		for idx in range(feature_cnt):
			uniq_feature_set = set(X[:,idx])

			new_entropy = 0.0

			for cand_value in uniq_feature_set:
				sub_X,sub_y = getMatchedDataSet(X,y,idx,cand_value)
				
				cand_prob = float(len(sub_y))/len(y)
				
				new_entropy += calculateEntropy(sub_y) * cand_prob


			cand_info_gain = new_entropy - entropy_before_split

			if cand_info_gain > best_info_gain:
				best_info_gain = cand_info_gain
				best_feature_idx = idx

		return best_feature_idx

	
	def getBestSplitFeatureWithC45(self,X,y):
		'''
		get best split feature using info gain rate	
		'''
		feature_cnt = X.shape[1]

		entropy_before_split = calculateEntropy(y)

		best_info_gain_rate = 0.0
		best_feature_idx = -1
		
		total_sample_cnt = X.shape[0]
		for idx in range(feature_cnt):
			uniq_feature_set = set(X[:,idx])

			new_entropy = 0.0
			split_infomation = 0.0

			for cand_value in uniq_feature_set:
				sub_X,sub_y = getMatchedDataSet(X,y,idx,cand_value)
				
				cand_prob = float(len(sub_y))/len(y)
				
				new_entropy += calculateEntropy(sub_y) * cand_prob
				split_infomation -= cand_prob * np.log2(cand_prob)

			
			if split_infomation != 0.0:
				cand_info_gain_rate = (new_entropy - entropy_before_split) / split_infomation

				if cand_info_gain_rate > best_info_gain_rate:
					best_info_gain_rate = cand_info_gain_rate
					best_feature_idx = idx
		
		return best_feature_idx

	def createTree(self,X,y,cand_feature_list):
		'''
		cand_feature_list: cand split feature str tuple,each element is a feature string	
		'''
		label_list = list(y)

		if len(label_list) == 1:
			return label_list[0]
		
		#无分裂特征可选,选择最多的标签
		if len(cand_feature_list) == 0:
			return getMajorlabel(label_list)

		if self._mode == 'ID3':
			best_feature_idx = self.getBestSplitFeatureWithID3(X,y)
		else:
			best_feature_idx = self.getBestSplitFeatureWithC45(X,y)

		feature_list = list(cand_feature_list)
		best_feature_str = feature_list[best_feature_idx]

		feature_list.remove(best_feature_str)
		feature_list = tuple(feature_list)
		
		cand_tree = {best_feature_str : dict()}
		feature_val_set = set(X[:,best_feature_idx])

		for cand_value in feature_val_set:
			sub_X,sub_y = getMatchedDataSet(X,y,best_feature_idx,cand_value)
			
			cand_tree[best_feature_str][cand_value] = self.createTree(sub_X,sub_y,feature_list)

		return cand_tree
	
	def fit(self,X,y):
		#type detect
		if not isinstance(X,np.ndarray):
			try:
				X = np.array(X)
			except Exception,e:
				print 'X type error. type=' + str(X)
				sys.exit(1)

		if not isinstance(y,np.ndarray):
			try:
				y = np.array(y)
			except Exception,e:
				print 'Y type error. type=' + str(y)
				sys.exit(1)
		
		feature_list = tuple(['x' + str(idx) for idx in range(X.shape[1])])
		self._tree = self.createTree(X,y,feature_list)

		return self
	
	def predict(self,X):
		if not isinstance(X,np.ndarray):
			try:
				X = np.array(X)
			except Exception,e:
				print 'X type error. type=' + str(X)
				sys.exit(1)

		if not self._tree:
			raise NotFitError('Not fit model. call fit or load model firstly')

		def _classify(tree,sample):
			'''
			classify one sample using tree
			'''
			feature_str = tree.keys()[0]

			sub_tree_of_feature = tree[feature_str]

			feature_idx = int(feature_str[1:])
			feature_val = sample[feature_idx]
			
			#当前特征值在训练数据中没有出现，选择训练数据中最接近的值对应的分支
			if feature_val not in sub_tree_of_feature:
				most_likely_value = None
				min_diff = sys.maxint

				for cand_value in sub_tree_of_feature:
					cand_diff = abs(cand_value - feature_val)
					if cand_diff < min_diff:
						min_diff = cand_diff
						most_likely_value = cand_value

				if None != most_likely_value:
					sub_tree_of_value = sub_tree_of_feature[most_likely_value]
				else:
					raise Exception('No value in subtree')
			
			else:
				sub_tree_of_value = sub_tree_of_feature[feature_val]

			if isinstance(sub_tree_of_value,dict):
				label = _classify(sub_tree_of_value,sample)
			else:
				label = sub_tree_of_value
			
			return label
		
		#only one sample, one diem array
		if len(X.shape) == 1:
			return _classify(self._tree,X)
		else:
			results = []
			for idx in range(X.shape[0]):
				cand_sample = X[idx]
				results.append(_classify(self._tree,cand_sample))

			return np.array(results)


	def saveModel(self,model_name):
		if not self._tree:
			raise NotFitError('tree is None. call fit firstly')
		
		try:
			pickle.dump(self._tree,file(model_name,'wb'),True)
		except Exception,e:
			print 'Save model fail. err = ' + str(e)
			sys.exit(1)

	def loadModel(self,model_name):
		try:
			self._tree = pickle.load(file(model_name,'rb'))
		
		except Exception,e:
			print 'Load model fail. err = ' + str(e)
			sys.exit(1)
