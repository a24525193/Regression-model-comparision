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


Viewing the correlation coefficient of a feature

View the correlation coefficient between the characteristic and the target value "G3"

Recode all category features with One hot encoding

The number of recoded features is 59 (including "G3")

Review the correlation coefficient between all characteristics and the target value "G3" again

Pick out the top ten characteristics of correlation coefficient

And remake the dataframe



## Regression Model

Make two functions and build all the models into them, including

**LinearRegression()  
RandomForestRegressor()    
SVR()   
KNeighborsRegressor()  
ElasticNet()  
ExtraTreesRegressor()  
GradientBoostingRegressor()**

Build a loop and build a forecast

Make MAE, RMSE standard and dataframe in the circle

and make a dataframe storing the results of the model

Standards for making Baseline

Finally return the result


The first function maintains the default values for all regression models
The second function adjusts the parameter value
The adjusted parameter values are as follows

### Linear Regression

Because I wanted to maintain the original model, the linear regression did not adjust the parameters.

### Random Forest Regressor

Set parameter value:  
n_estimators = 500,  
max_depth = 5,  
min_samples_leaf = 2,  
max_features = "auto",  
criterion = 'squared_error',  
random_state = 1  

### Support Vector Machine


Set parameter value:  
kernel = 'rbf',   
degree = 3,  
C = 100,  
gamma = 0.001 

### KNeighbors Regressor

Set parameter value:  
n_neighbors = 13,  
algorithm = "auto",  
leaf_size = 15,  
metric = "euclidean"

### Elastic Net

Set parameter value:  
alpha = 1.2,  
l1_ratio = 1,  
max_iter = 100  

### ExtraTrees Regressor

Set parameter value:    
n_estimators = 450,  
max_depth = 7,  
min_samples_leaf = 1,  
min_samples_split = 2,  
max_features = "auto",  
criterion = 'squared_error',  
random_state=1
                                  
### Gradient Boosting Regressor

Set parameter value:    
n_estimators = 300,  
learning_rate = 0.05,  
max_depth = 6


## Create Plots 

Make a bar plot of the results All charts include MSE, RMSE.

Make a bar plot after adjusting the parameters of the model.

Make a bar plot of the original model parameters.

Make a comparison chart of the parameters of the adjusted model and the original model parameters.

Train a regression model using all eigenvalues.

Make a bar plot that uses all features.

Train the regression model after deleting G2.

Make a barplot after deleting G2.

Train the regression model after deleting G2, G1.

Make a bar plot after deleting G2, G1.

Make a comparison chart of the regression model using all features and the regression model with selected features.

Make a comparison chart of the model with G2 deleted, the model with G2G1 deleted, and the model without deletion.

