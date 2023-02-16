# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 19:59:46 2022

@author: a2452
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import exp


data = pd.read_csv("student.csv")

print(data.head())

#NAN?
print(data.isnull().any())

#descriptive statistics
describe = data.describe()
print(describe)


corr = data.corr()
print(corr)


data_corr = data.corr()['G3']


# One-Hot Encoding of Categorical Variables
student = pd.get_dummies(data)

student_allcorr = student.corr()
student_corr = student.corr()['G3']

plt.figure(figsize=(50, 40))
hm = sns.heatmap(student_allcorr, annot=True)
plt.show()

#选择特征值
# Find correlations with the Grade
stmost_corr = student.corr().abs()['G3'].sort_values(ascending=False)

# Maintain the top 8 most correlation features with Grade
most_corr = stmost_corr[1:13]
print(most_corr)



#创建特征值的dataframe
student_pre = student.loc[:, most_corr.index]
print(student_pre.head())

# 删除 higher_no , romantic_no 
student_pre = student_pre.drop('higher_no',axis=1)
student_pre = student_pre.drop('romantic_no',axis=1)
print(student_pre.head())


#创建XY 和 训练测试集
X = student_pre

y = data['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=459323)

# -------------------------------

# Standard ML Models for comparison
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# Building original regression model
def evaluate_origin(X_train, X_test, y_train, y_test):

    model_name_list = ['Linear Regression', 'Random Forest', 'SVM',
                       'K-Nearest Neighbors','ElasticNet Regression',
                      'Extra Trees', 'Gradient Boosted', 'Baseline']
                        

    # Instantiate the models
    model1 = LinearRegression()
    
    model2 = RandomForestRegressor()    
    
    model3 = SVR()   
    
    model4 = KNeighborsRegressor()
    
    model5 = ElasticNet()
    
    model6 = ExtraTreesRegressor()
    
    model7 = GradientBoostingRegressor()
    
    # Dataframe for results
    results = pd.DataFrame(columns=['MAE', 'RMSE'], index = model_name_list)
    
    # Train and predict with each model
    for i, model in enumerate([model1, model2, model3, model4, model5, model6, model7]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Metrics
        MAE = np.mean(abs(predictions - y_test))
        RMSE = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [MAE, RMSE]
    
    # Median Value Baseline Metrics
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    
    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]
    
    return results

results_origin = evaluate_origin(X_train, X_test, y_train, y_test)

print()
print("Using origin models:")
print(results_origin)

# -------------------------------

# setting parameters

def evaluate_adjust(X_train, X_test, y_train, y_test):

    model_name_list = ['Linear Regression', 'Random Forest', 'SVM',
                       'K-Nearest Neighbors','ElasticNet Regression',
                      'Extra Trees', 'Gradient Boosted', 'Baseline']
                        

    # Instantiate the models
    model1 = LinearRegression()
    
    model2 = RandomForestRegressor(n_estimators=500,
                                  max_depth=5,
                                  min_samples_leaf=2,
                                  max_features="auto",
                                  criterion = 'squared_error',
                                  random_state=1)    
    
    model3 = SVR(kernel='rbf', 
                 degree=3, 
                 C=100, 
                 gamma=0.001)   
    
    model4 = KNeighborsRegressor(n_neighbors=13,
                                   algorithm ="auto", 
                                   leaf_size = 15,
                                   metric = "euclidean")
    
    model5 = ElasticNet(alpha=1.2, 
                        l1_ratio=1, 
                        max_iter = 100)
    
    model6 = ExtraTreesRegressor(n_estimators=450,
                                  max_depth=7,
                                  min_samples_leaf=1,
                                  min_samples_split=2,
                                  max_features="auto",
                                  criterion = 'squared_error',
                                  random_state=1)
    
    model7 = GradientBoostingRegressor(n_estimators=300, 
                               learning_rate= 0.05,
                               max_depth = 6)
    
    # Dataframe for results
    results = pd.DataFrame(columns=['MAE', 'RMSE'], index = model_name_list)
    
    # Train and predict with each model
    for i, model in enumerate([model1, model2, model3, model4, model5, model6, model7]):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Metrics
        MAE = np.mean(abs(predictions - y_test))
        RMSE = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Insert results into the dataframe
        model_name = model_name_list[i]
        results.loc[model_name, :] = [MAE, RMSE]
    
    # Median Value Baseline Metrics
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    
    results.loc['Baseline', :] = [baseline_mae, baseline_rmse]
    
    return results



# 预测结果
results_adjust = evaluate_adjust(X_train, X_test, y_train, y_test)

print()
print("After Adjust :")
print(results_adjust)

# -------------------------------

# #模型比对
plt.figure(figsize=(20, 12))

# Root mean squared error
ax1 =  plt.subplot(1, 2, 1)
results_adjust.sort_values('MAE', ascending = True).plot.bar(y = 'MAE', color = 'm', ax = ax1)
plt.title('Mean Absolute Error Adjusted Model') 


# Median absolute percentage error
ax2 = plt.subplot(1, 2, 2)
results_adjust.sort_values('RMSE', ascending = True).plot.bar(y = 'RMSE', color = 'c', ax = ax2)
plt.title('Root Mean Squared Error Adjusted Model') 

plt.show()

# -------------------------------

#比对图表
x = np.arange(len(results_origin.MAE))
width = 0.35


model_name_list = ('Linear Regression', 'Random Forest', 'SVM',
                   'K-Nearest Neighbors','ElasticNet Regression',
                  'Extra Trees', 'Gradient Boosted', 'Baseline')
                    


# #模型比对
plt.figure(figsize=(20, 12))

# Root mean squared error
ax1 =  plt.subplot(1, 2, 1)
results_origin.sort_values('MAE', ascending = True).plot.bar(y = 'MAE', color = 'm', ax = ax1)
plt.title('Mean Absolute Error Original Model') 


# Median absolute percentage error
ax2 = plt.subplot(1, 2, 2)
results_origin.sort_values('RMSE', ascending = True).plot.bar(y = 'RMSE', color = 'c', ax = ax2)
plt.title('Root Mean Squared Error Original Model') 


plt.show()


#原始模型跟挑选特征对比
fig,(ax1, ax2) = plt.subplots(ncols=2)
rects1 = ax1.bar( x-width/2 , results_origin.MAE, width, label="Original Model")
rects2 = ax1.bar( x+width/2 , results_adjust.MAE, width, label="Adjusted Model")
ax1.set_xticks(x, model_name_list,rotation=90)
ax1.set_title('MAE Comparison') 
ax1.legend()

rects1 = ax2.bar( x-width/2 , results_origin.RMSE, width, label="Original Model")
rects2 = ax2.bar( x+width/2 , results_adjust.RMSE, width, label="Adjusted Model")
ax2.set_xticks(x, model_name_list,rotation=90)
ax2.set_title('RMSE Comparison') 
ax2.legend()

plt.show()



# -------------------------------

#使用全部特征
student_train = student.drop('G3',axis=1)

X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(student_train , y, test_size=0.2 ,random_state=459323)

results_allfeatures = evaluate_adjust(X_all_train, X_all_test, y_all_train, y_all_test)

print()
print("Using all features :")
print(results_allfeatures)

#结果图
plt.figure(figsize=(20, 12))

# Root mean squared error
ax1 =  plt.subplot(1, 2, 1)
results_allfeatures.sort_values('MAE', ascending = True).plot.bar(y = 'MAE', color = 'm', ax = ax1)
plt.title('Model Mean Absolute Error Which use all features') 


# Median absolute percentage error
ax2 = plt.subplot(1, 2, 2)
results_allfeatures.sort_values('RMSE', ascending = True).plot.bar(y = 'RMSE', color = 'c', ax = ax2)
plt.title('Model Root Mean Squared Error Which use all features') 

plt.show()



# -------------------------------

#删除G2

X_dropG2 = student_train.drop('G2',axis=1)

X_G2_train, X_G2_test, y_G2_train, y_G2_test = train_test_split(X_dropG2, y, test_size=0.2 ,random_state=459323)

results_dropG2 = evaluate_adjust(X_G2_train, X_G2_test, y_G2_train, y_G2_test)


plt.figure(figsize=(20, 12))

# Root mean squared error
ax1 =  plt.subplot(1, 2, 1)
results_dropG2.sort_values('MAE', ascending = True).plot.bar(y = 'MAE', color = 'm', ax = ax1)
plt.title('Model Mean Absolute Error Which drop G2') 


# Median absolute percentage error
ax2 = plt.subplot(1, 2, 2)
results_dropG2.sort_values('RMSE', ascending = True).plot.bar(y = 'RMSE', color = 'c', ax = ax2)
plt.title('Model Root Mean Squared Error Which drop G2') 

plt.show()

print()
print("Drop G2 :")
print(results_dropG2)

# -------------------------------

#删除G1

X_dropG1 = X_dropG2.drop('G1',axis=1)


X_G1_train, X_G1_test, y_G1_train, y_G1_test = train_test_split(X_dropG1, y, test_size=0.2 ,random_state=459323)

results_dropG1 = evaluate_adjust(X_G1_train, X_G1_test, y_G1_train, y_G1_test)


plt.figure(figsize=(20, 12))

# Root mean squared error
ax1 =  plt.subplot(1, 2, 1)
results_dropG1.sort_values('MAE', ascending = True).plot.bar(y = 'MAE', color = 'm', ax = ax1)
plt.title('Model Mean Absolute Error Which drop G1') 


# Median absolute percentage error
ax2 = plt.subplot(1, 2, 2)
results_dropG1.sort_values('RMSE', ascending = True).plot.bar(y = 'RMSE', color = 'c', ax = ax2)
plt.title('Model Root Mean Squared Error Which drop G1') 

plt.show()

print()
print("Drop G1 :")
print(results_dropG1)

# -------------------------------

#全部特征跟挑选特征对比
fig,(ax1, ax2) = plt.subplots(ncols=2)
rects1 = ax1.bar( x-width/2 , results_allfeatures.MAE, width, label="Using all features")
rects2 = ax1.bar( x+width/2 , results_adjust.MAE, width, label="Chosen features")
ax1.set_xticks(x, model_name_list,rotation=90)
ax1.set_title('MAE Comparison') 
ax1.legend()

rects1 = ax2.bar( x-width/2 , results_allfeatures.RMSE, width, label="Using all features")
rects2 = ax2.bar( x+width/2 , results_adjust.RMSE, width, label="Chosen features")
ax2.set_xticks(x, model_name_list,rotation=90)
ax2.set_title('RMSE Comparison') 
ax2.legend()

plt.show()


# -------------------------------

#删掉G2G1三个对比
width = 0.3

fig,(ax1, ax2) = plt.subplots(ncols=2)

rects1 = ax1.bar( x-width , results_adjust.MAE, width, label="With G2,G1")
rects2 = ax1.bar( x , results_dropG2.MAE, width, label="Drop G2")
rects3 = ax1.bar( x+width , results_dropG1.MAE, width, label="Drop G1")
ax1.set_xticks(x, model_name_list,rotation=90)
ax1.set_title('MAE Comparison') 

rects1 = ax2.bar( x-width , results_adjust.RMSE, width, label="With G2,G1")
rects2 = ax2.bar( x , results_dropG2.RMSE, width, label="Drop G2")
rects3 = ax2.bar( x+width , results_dropG1.RMSE, width, label="Drop G1")
ax2.set_xticks(x, model_name_list,rotation=90)
ax2.set_title('RMSE Comparison') 
ax2.legend(loc='center right', bbox_to_anchor=(0.5, 1.2),ncol=3)




