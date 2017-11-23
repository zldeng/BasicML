#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng hitdzl@gmail.com
  create@2017-11-22 18:56:17
'''
import numpy as np
import sys
import pickle
import json

class LogisticRegression(object):
	def __init__(self,step_size = 0.1,max_iter = 100,l2_reg_lambd = 0.5,debug = True):
		self.step_size = step_size
		self.max_iter = max_iter
		self.l2_reg_lambd = l2_reg_lambd
		self.coef = None
		self.pred_cutoff = 0.5
		self.debug = debug

		

	def _getDerivations(self,labels,feature_matrix,pred_prob):
		'''
		get derivations of negative log likelyhood with L2 regulation	
		'''
		sample_cnt = feature_matrix.shape[0]
		pred_diff = pred_prob - labels

		derivations = 1.0 / sample_cnt * np.dot(np.transpose(feature_matrix),pred_diff) \
					  + self.l2_reg_lambd * 2 * self.coef

		return derivations

	def calNegativeLogLikelihood(self,labels,feature_matrix):
		scores = np.dot(feature_matrix,self.coef)

		log_exp = np.log(1 + np.exp(-scores))

		mask = np.isinf(log_exp)
		log_exp[mask] = -scores[mask]
		
		likelyhood = np.sum((1 - labels) * scores + log_exp)
		
		return likelyhood

	def predProbability(self,feature):
		'''
		get probality of label 1
		'''
		product = np.dot(feature,self.coef)

		prob = 1.0 / (1.0 + np.exp(-product))

		return prob

	def fit(self,feature_matrix,labels):
		feature_matrix = np.column_stack((np.ones(feature_matrix.shape[0]),feature_matrix))

		self.coef = np.zeros(feature_matrix.shape[1])

		for itr in range(self.max_iter):
			predictions = self.predProbability(feature_matrix)

			derivations = self._getDerivations(labels,feature_matrix,predictions)

			self.coef -= self.step_size * derivations

			if self.debug and itr % 5 == 0:
				cand_likelyhood = self.calNegativeLogLikelihood(labels,feature_matrix)
				print 'Iter %d : likelyhood: %.8f' % (itr,cand_likelyhood)


	def predict(self,feature_matrix):
		feature_matrix = np.column_stack((np.ones(feature_matrix.shape[0]),feature_matrix))

		predictions = self.predProbability(feature_matrix)
		
		return (predictions > self.pred_cutoff).astype(int)


