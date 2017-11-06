#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-11-06 19:26:15
'''
from decisionTree import DecisionTree

def test():
	X = [[1, 2, 0, 1, 0],
		 [0, 1, 1, 0, 1],
		 [1, 0, 0, 0, 1],
		 [2, 1, 1, 0, 1],
		 [1, 1, 0, 1, 1]]
	y = ['yes','yes','no','no','no']

	decision_tree = DecisionTree(mode = 'C4.5')

	decision_tree.fit(X,y)

	res = decision_tree.predict(X)
	print res
	
	model_name = 'test.dt'
	decision_tree.saveModel(model_name)
	
	new_tree = DecisionTree()
	new_tree.loadModel(model_name)
	print new_tree.predict(X)


if __name__ == '__main__':
	test()
