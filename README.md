# Regression models comparision by student performance on math subject

Pattern Recognition course project using Python

## Data set

https://archive.ics.uci.edu/ml/datasets/student+performance

This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). **This study used only Mathematics data sets.**

This dataset contains student achievement of 395 student. Each student profile has 33 features.

Attribute Information:

1. school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)

2. sex - student's sex (binary: 'F' - female or 'M' - male)

3. age - student's age (numeric: from 15 to 22)

4. address - student's home address type (binary: 'U' - urban or 'R' - rural)

5. famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)

6. Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)

7. Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)

8. Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 â€“ 5th to 9th grade, 3 â€“ secondary education or 4 â€“ higher education)

9. Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')

10. Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')

11. reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')

12. guardian - student's guardian (nominal: 'mother', 'father' or 'other')

13. traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)

14. studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)

15. failures - number of past class failures (numeric: n if 1<=n<3, else 4)

16. schoolsup - extra educational support (binary: yes or no)

17. famsup - family educational support (binary: yes or no)

18. paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)

19. activities - extra-curricular activities (binary: yes or no)

20. nursery - attended nursery school (binary: yes or no)

21. higher - wants to take higher education (binary: yes or no)

22. internet - Internet access at home (binary: yes or no)

23. romantic - with a romantic relationship (binary: yes or no)

24. famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)

25. freetime - free time after school (numeric: from 1 - very low to 5 - very high)

26. goout - going out with friends (numeric: from 1 - very low to 5 - very high)

27. Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)

28. Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)

29. health - current health status (numeric: from 1 - very bad to 5 - very good)

30. absences - number of school absences (numeric: from 0 to 93)

These grades are related with the course subject, Math:

31. G1 - first period grade (numeric: from 0 to 20)

32. G2 - second period grade (numeric: from 0 to 20)

33. G3 - final grade (numeric: from 0 to 20, output target)




## Package used

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from math import exp
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

```

## Exploratory Data Analysis

Input the dataset as dataframe and check the data status.

Output the first five records of data set.

Check whether data sets have missing values, and no missing values are found.

Output descriptive statistics of datasets.


## Preprocessing



查看特征的相关系数

查看特征与目标值"G3"的相关系数

将所有的类别特征使用 One hot encoding 重新编码

重新编码后的特征数有59个(包括"G3")

再次查看所有特征与目标值"G3"的相关系数

挑出相关系数前十的特征

并重新制作dataframe




## Regression Model

制作两个function 并将所有的模型都建进去，包括

**LinearRegression()  
RandomForestRegressor()    
SVR()   
KNeighborsRegressor()  
ElasticNet()  
ExtraTreesRegressor()  
GradientBoostingRegressor()**


建立一个回圈并建立预测

再回圈中制作MAE,RMSE的标准跟dataframe

和制作储存模型的结果的dataframe

制作Baseline的标准

最后回传结果


第一个function维持所有回归模型默认的数值  
第二个function调整参数值  
调整完的参数值如下  

### Linear Regression

因想维持原本的模型，所以线性回归并没有进行参数调整

### Random Forest Regressor

设定参数数值  
n_estimators = 500,  
max_depth = 5,  
min_samples_leaf = 2,  
max_features = "auto",  
criterion = 'squared_error',  
random_state = 1  

### Support Vector Machine


设定参数数值  
kernel = 'rbf',   
degree = 3,  
C = 100,  
gamma = 0.001 

### KNeighbors Regressor

设定参数数值  
n_neighbors = 13,  
algorithm = "auto",  
leaf_size = 15,  
metric = "euclidean"

### Elastic Net

设定参数数值  
alpha = 1.2,  
l1_ratio = 1,  
max_iter = 100  

### ExtraTrees Regressor

设定参数数值  
n_estimators = 450,  
max_depth = 7,  
min_samples_leaf = 1,  
min_samples_split = 2,  
max_features = "auto",  
criterion = 'squared_error',  
random_state=1
                                  
### Gradient Boosting Regressor

设定参数数值  
n_estimators = 300,  
learning_rate = 0.05,  
max_depth = 6


## Create Plots 

制作结果的 bar plot 所有的图表都包括MSE,RMSE

制作模型调整参数后的 bar plot

制作原始模型参数的barplot

制作调整过参数的模型与原始模型参数的比对图表

训练使用全部特征值的回归模型

制作使用全部特征的 barplot

训练删掉G2后的回归模型

制作删掉G2后的 barplot

训练删掉G2G1后的回归模型

制作删掉G2G1后的barplot

制作使用全部特征的回归模型与挑选过特征的回归模型的比对图表

制作删掉G2的模型、删掉G2G1的模型，与没有删除的模型的比对图表
