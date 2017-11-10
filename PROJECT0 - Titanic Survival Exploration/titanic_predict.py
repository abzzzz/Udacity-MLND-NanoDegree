#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 15:35:22 2017

@author: Haoran
"""

import numpy as np
import pandas as pd

# 数据可视化代码
from titanic_visualizations import survival_stats
from IPython.display import display
# 加载数据集
in_file = 'titanic_data.csv'
full_data = pd.read_csv(in_file)

# 显示数据列表中的前几项乘客数据
display(full_data.head())

# 从数据集中移除 'Survived' 这个特征，并将它存储在一个新的变量中。
outcomes = full_data['Survived']
data = full_data.drop('Survived', axis = 1)  #可以使用drop移除read_csv中的某一项

# 显示已移除 'Survived' 特征的数据集
display(data.head())

def accuracy_score(truth, pred):
    """ 返回 pred 相对于 truth 的准确率 """
    
    # 确保预测的数量与结果的数量一致
    if len(truth) == len(pred): 
        
        # 计算预测准确率（百分比）
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    
    else:
        return "Number of predictions does not match number of outcomes!"
    
# 测试 'accuracy_score' 函数
predictions = pd.Series(np.ones(5, dtype = int)) #五个预测全部为1，既存活
print accuracy_score(outcomes[:5], predictions)

def predictions_0(data):
    """ 不考虑任何特征，预测所有人都无法生还 """

    predictions = []
    for _, passenger in data.iterrows():
        
        # 预测 'passenger' 的生还率
        predictions.append(0)
    
    # 返回预测结果
    return pd.Series(predictions)

# 进行预测
predictions = predictions_0(data)
print(accuracy_score(outcomes[:], predictions))

print accuracy_score(outcomes, predictions)

survival_stats(data, outcomes, 'Sex')

def predictions_1(data):
    """ 只考虑一个特征，如果是女性则生还 """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female':
            predictions.append(1)
        
        else:
            predictions.append(0)
    
    # 返回预测结果
    return pd.Series(predictions)

# 进行预测
predictions = predictions_1(data)
print(accuracy_score(outcomes[:], predictions))

survival_stats(data, outcomes, 'Age', ["Sex == 'male'"])

def predictions_2(data):
    """ 考虑两个特征: 
            - 如果是女性则生还
            - 如果是男性并且小于10岁则生还 """
    
    predictions = []
    for _, passenger in data.iterrows():
        
        if passenger['Sex'] == 'female':
            predictions.append(1)
        else:
            if passenger['Age'] < 10:
                predictions.append(1)
            else:
                predictions.append(0)
    
    # 返回预测结果
    return pd.Series(predictions)

# 进行预测
predictions = predictions_2(data)
print (accuracy_score(outcomes[:], predictions))

#survival_stats(data, outcomes, 'Fare', ["Sex == 'male'", "Pclass == 2", "Age > 17","SibSp == 0","Parch == 0", "Embarked == 'S'"])
#survival_stats(data, outcomes, 'Parch', ["Sex == 'male'", "Pclass == 1", "Age <= 20","SibSp == 1"])
#survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Pclass == 1", "Age < 10"])
#survival_stats(data, outcomes, 'Age', ["Sex == 'male'", "Pclass == 2", "Age < 10"])
#survival_stats(data, outcomes, 'Parch', ["Sex == 'male'", "Pclass == 3", "Age < 10","SibSp == 4"])
#survival_stats(data, outcomes, 'Pclass', ["Sex == 'female'"])
#survival_stats(data, outcomes, 'SibSp', ["Sex == 'female'"])
#survival_stats(data, outcomes, 'Parch', ["Sex == 'female'"])
#女性的存活和fare没有关系
#survival_stats(data, outcomes, 'Parch', ["Sex == 'female'","Pclass == 1","SibSp == 1"])
#survival_stats(data, outcomes, 'Parch', ["Sex == 'female'","Pclass == 2","SibSp == 2"])
#survival_stats(data, outcomes, 'Parch', ["Sex == 'female'","Pclass == 3","SibSp == 4"])

def predictions_3(data):
    """ 考虑多个特征，准确率至少达到80% """
    
    predictions = []
    for _, passenger in data.iterrows():
        #女性：从年龄开始 一层层的 pclass sibsp parch 将不明确归类的情况一层层深入
        if passenger['Sex'] == 'female':
            if passenger['Pclass'] == 3:
                if passenger['SibSp'] == 0:
                    if passenger['Parch'] <= 3:
                        predictions.append(1)
                    else:
                        predictions.append(0)
                    
                elif passenger['SibSp'] == 1:
                    if passenger['Parch'] == 0 or passenger['Parch'] ==1 or passenger['Parch'] ==5:
                        predictions.append(1)
                    else:
                        predictions.append(0)
                
                elif passenger['SibSp'] == 2:
                    if passenger['Parch'] <= 1:
                        predictions.append(1)
                    else:
                        predictions.append(0)
                
                elif passenger['SibSp'] == 3:
                    if passenger['Parch'] == 0:
                        predictions.append(1)
                    else:
                        predictions.append(0)
                else:
                    predictions.append(0)
            else:
                predictions.append(1)
        
        #男性：从年龄开始 一层层的 pclass sibsp parch 将不明确归类的情况一层层深入
        else:
            if passenger['Age'] < 10:
                if passenger['Pclass'] == 3:
                    if passenger['SibSp'] == 0 or passenger['SibSp'] == 1:
                        predictions.append(1)
                    
                    elif passenger['SibSp'] == 4:
                        if passenger['Parch'] == 2:
                            predictions.append(1)
                        else:
                            predictions.append(0)
                    
                    else:
                        predictions.append(0)
                
                else:
                    predictions.append(1)
            
            else:
                if passenger['Pclass'] == 1:
                    if passenger['Age'] <= 20:
                        if passenger['SibSp'] == 0:
                            predictions.append(1)
                        
                        elif passenger['SibSp'] == 1:
                            if passenger['Parch'] == 2:
                                predictions.append(1)
                            else:
                                predictions.append(0)
                    
                        else:
                            predictions.append(0)
                    else:
                        if passenger['Age'] <= 40:
                            predictions.append(1)
                        else:
                            predictions.append(0)
                
                elif passenger['Pclass'] == 2:
                    if passenger['Age'] <= 17:
                        if passenger['SibSp'] == 0:
                            if passenger['Parch'] == 2:
                                predictions.append(1)
                            else:
                                predictions.append(0)
                        elif passenger['SibSp'] == 1 or passenger['SibSp'] == 2:
                            predictions.append(1)
                        else:
                            if passenger['SibSp'] == 0 or passenger['SibSp'] == 1:
                                if passenger['Parch'] == 0:
                                    if passenger['Embarked'] == 'S':
                                        if passenger['Fare'] < 20:
                                            predictions.append(1)
                                        else:
                                            predictions.append(0)
                                        
                                    else:
                                        predictions.append(0)
                                else:
                                    predictions.append(0)
                            else:
                                predictions.append(0)
                    else:
                        predictions.append(0)
                
                else:
                    predictions.append(0)
                    
    # 返回预测结果
    return pd.Series(predictions)

# 进行预测
predictions = predictions_3(data)
print (accuracy_score(outcomes[:], predictions))