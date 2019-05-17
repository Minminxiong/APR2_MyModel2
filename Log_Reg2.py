# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:14:07 2019
通过网格搜索，查找最优的参数，获取最优的结果
@author: XMM
"""
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn.model_selection import ParameterGrid

def parse_args():

	parser = argparse.ArgumentParser(description="Run APR.")

	parser.add_argument('--input_trainFile', nargs='?', default='features/train_features_k25_cas50.csv',
	                    help='Input train cascadaes features file path')

	parser.add_argument('--input_testFile', nargs='?', default='features/validation1_features_k25_cas10.csv',
	                    help='Input test cascadaes features file path')

	parser.add_argument('--output', nargs='?', default='../dataset/features/result.csv',
	                    help='Features path')
	parser.add_argument('--k', type=int, default=25,
	                    help='Number of early adopters. Default is 25.')
	parser.add_argument('--myModel', type=int, default=0,
	                    help='1 denotes user MyModel featuers. 0 denotes APR features.')
	return parser.parse_args()

def read_features_csv(file, My_Model):
    '''读取文件中的数据
       My_Model: 根据args.myModel的值来确定 
    '''
    #需要注意第一行数据未读取
    df = pd.read_csv(file)
    columns_size = df.columns.size
    if My_Model == 0:
        X = df.iloc[:,1: columns_size - args.k - 2]
    else:
        X = pd.concat([df.iloc[:,1: columns_size - 2* args.k -2] ,df.iloc[:, columns_size - args.k -2: -2]], axis=1)
    Y = df.iloc[:,-1]
    return X, Y
    
def get_data(train_file, test_file, myModel):
    '''
    获取训练集，测试集数据
    并将其中的特征数据标准化
    '''
    train_x, train_y = read_features_csv(train_file, myModel)
    test_x, test_y = read_features_csv(test_file, myModel)
    ss = StandardScaler()
    train_x = ss.fit_transform(train_x)
    test_x = ss.fit_transform(test_x)
    return train_x, train_y, test_x, test_y    

def get_result(y, y_pre):
    accuracy = accuracy_score(y, y_pre)
    precision = precision_score(y,y_pre)
    recall = recall_score(y, y_pre)
    F1 = f1_score(y, y_pre)
    print("accuracy, precision, recall, F1:", accuracy, precision, recall, F1)
        
def Grid_Logistic():
    param_grid = ParameterGrid({'C':[.1, 1, 10], 'class_weight': ['balanced']})    
    train_x, train_y, test_x, test_y = get_data(args.input_trainFile, args.input_testFile, args.myModel)
    for params in param_grid:
        log_reg = LogisticRegression(**params)
        print(log_reg)
        log_reg.fit(train_x, train_y)
        y_pre = log_reg.predict(test_x)
        get_result(test_y, y_pre)


def main(args):
    Grid_Logistic()
    
if __name__ == "__main__":
	args = parse_args()
	y_p_pre = main(args)
    
    

# =============================================================================
# class Log_reg():
#     def __init__(self):
#         self.log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, \
#                                           intercept_scaling=1, class_weight='balanced', random_state=0, solver='liblinear', \
#                                           max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
#         self.ss = StandardScaler()
#         
#     def read_features_csv(self, file, My_Model):
#         #需要注意第一行数据未读取
#         df = pd.read_csv(file)
#         columns_size = df.columns.size
#         if My_Model == 0:
#             X = df.iloc[:,1: columns_size - args.k - 2]
#         else:
#             X = pd.concat([df.iloc[:,1: columns_size - 2* args.k -2] ,df.iloc[:, columns_size - args.k -2: -2]], axis=1)
#         Y = df.iloc[:,-1]
#         return X, Y
# 
#     
#     def fit_logistic_regression(self, train_file):
#         x, y = self.read_features_csv(train_file, args.myModel)   
#     #    ss = StandardScaler()
#         
#         x = self.ss.fit_transform(x)
#         
# #        print(x,y)
#     #    log_reg = LogisticRegression()
#         self.log_reg.fit(x, y)
#     #    y_pre = log_reg.predict_proba(x)
# #        return log_reg
#     
#     def predict_test(self, test_file):
#         x, y = self.read_features_csv(test_file, args.myModel)
#         x = self.ss.fit_transform(x)
#         y_p_pre = self.log_reg.predict_proba(x)
#         y_pre = self.log_reg.predict(x)
#         accuracy = accuracy_score(y, y_pre)
#         precision = precision_score(y,y_pre)
#         recall = recall_score(y, y_pre)
#         F1 = f1_score(y, y_pre)
#         print("accuracy, precision, recall, F1:", accuracy, precision, recall, F1)
#         return y_p_pre, y
# 
# def main(args):
#     log_reg = Log_reg()
#     log_reg.fit_logistic_regression(args.input_trainFile)
#     y_p_pre, y = log_reg.predict_test(args.input_testFile)
#     return [y_p_pre, y]
# =============================================================================