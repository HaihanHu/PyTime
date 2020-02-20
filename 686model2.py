#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 20:31:23 2019

@author: shirleyhu
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Lasso,LassoCV

##########  Data  ##########

df= pd.read_csv('/Users/shirleyhu/Documents/686/model2.csv')
#model1:count of publications > 1, model2:whole, model3: sum1-5 above average


#df.fillna(value=0,inplace=True)
##
#corr = df.corr()
#ax = sns.heatmap(
#    corr, 
#    vmin=-1, vmax=1, center=0,
#    cmap=sns.diverging_palette(20, 220, n=200),
#    square=True
#)
#ax.set_xticklabels(
#    ax.get_xticklabels(),
#    rotation=45,
#    horizontalalignment='right'
#);
#
#corr.sort_values(["sum6"], ascending = False, inplace = True)
#
#
#a = pd.DataFrame(corr.sum6)
#a = a.sort_values('sum6')


y = df['sum6']
#y = np.log(df['sum6']+1)
x = df.drop(['author','sum6'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#
#plt.figure(figsize = (10,5))#print("skew: ",y.skew())
#sns.distplot(y)


##plot
#fig, ax = plt.subplots(figsize=(10,8))
#ax=sns.countplot(x = "first_year", data = df,\
#             palette='Blues_d',alpha=0.7)
#sns.despine()
#ax.spines['bottom'].set_visible(False)
#plt.show()
#
#fig, ax = plt.subplots(figsize=(10,8))
#ax=sns.countplot(y = "first_year", data = df,\
#             order = df['first_year'].value_counts().index, palette='Blues_d',alpha=0.7)
#sns.despine()
#ax.spines['bottom'].set_visible(False)
##for p in ax.patches:
##    ax.annotate('{:}'.format(p.get_width()), (p.get_width()+20, p.get_y()),va="top")
##plt.title("Product Distribution")
##plt.savefig("product.jpg", bbox_inches="tight")
#plt.show()


########## LR model ##########


lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_lr = lr.predict(X_train)
y_test_lr = lr.predict(X_test)
lr.score(X_test,y_test)
print ('R^2 is: \n', lr.score(X_test,y_test))


# Plot predictions
#plt.scatter(y_train_lr, y_train, s=10,c = "blue", marker = "o", label = "Training data")
#plt.scatter(y_test_lr, y_test,s=10, c = "lightgreen", marker = "o", label = "Validation data")
#plt.title("Linear regression")
#plt.xlabel("Predicted values")
#plt.ylabel("Real values")
#plt.legend(loc = "upper left")
#plt.plot([0, 30], [0, 30], c = "red")
#plt.show()
#
#
#
########### RF model ##########

rf = RandomForestRegressor(n_estimators=20, random_state=0)  
rf.fit(X_train, y_train)  
y_train_rf = rf.predict(X_train)
y_test_rf = rf.predict(X_test)

### Plot residuals
##plt.scatter(y_train_rf, y_train_rf - y_train, c = "steelblue", marker = "s", label = "Training data")
##plt.scatter(y_test_rf, y_test_rf - y_test, c = "orangered", marker = "s", label = "Validation data")
##plt.title("Random Forest Regression")
##plt.xlabel("Predicted values")
##plt.ylabel("Residuals")
##plt.legend(loc = "upper left")
##plt.show()

# #Plot predictions
#plt.scatter(y_train, y_train_rf,s=10, c = "steelblue", marker = "o", label = "Training data")
#plt.scatter(y_test, y_test_rf,s=10, c = "orangered", marker = "o", label = "Validation data")
#plt.title("Random Forest Regression")
#plt.xlabel("Real values")
#plt.ylabel("Predicted values")
#plt.legend(loc = "upper left")
#plt.plot([0, 30], [0, 30], c = 'orange')
#plt.show()
##rf.score(X_test,y_test)
#print ('R^2 is: \n', rf.score(X_test,y_test))
#
########### LASSO model ##########

lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)
print("Try again for more precision with alphas centered around " + str(alpha))
lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 10)
lasso.fit(X_train, y_train)
alpha = lasso.alpha_
print("Best alpha :", alpha)


lasso = Lasso(alpha = [0.003]).fit(X_train, y_train)
#
#coef = pd.Series(lasso.coef_, index = X_train.columns)
##print('Lasso picked'+str(sum(coef != 0 )))
#
#coef.sort_values().plot(kind = "barh")
#plt.title("Coefficients in the Lasso Model")
  
y_train_lasso = lasso.predict(X_train)
y_test_lasso = lasso.predict(X_test)
#
#plt.scatter(y_train_lasso, y_train,s=10, c = "steelblue", marker = "o", label = "Training data")
#plt.scatter(y_test_lasso, y_test,s=10, c = "orangered", marker = "o", label = "Validation data")
#plt.title("Lasso Regression")
#plt.xlabel("Predicted values")
#plt.ylabel("Real values")
#plt.legend(loc = "upper left")
#plt.plot([0, 30], [0, 30], c = 'orange')
#plt.show()

lasso.score(X_test,y_test)


########### RMSE Score ##########

scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))
    return(rmse)

def rmse_cv_test(model):
    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))
    return(rmse)

#print("RMSE on Training set :", rmse_cv_train(rf).mean())
#print("RMSE on Test set :", rmse_cv_test(rf).mean())



######### RF Features Importance ##########


#feature_importances = pd.DataFrame(rf.feature_importances_,
#                                   index = X_train.columns,
#                                    columns=['importance']).sort_values('importance',                                                                 
#                                                                        ascending=True)
#plt.rcParams["figure.figsize"] = (10,8)
#feature_importances.plot(kind = "barh")
#
#plt.title("Features Importance in the Random Forest Model")
#
#plt.show()




errordf=pd.DataFrame(columns=['train','test'])

errordf = errordf.append(pd.DataFrame([[rmse_cv_train(lr).mean(),rmse_cv_test(lr).mean()]],index=['lr'],columns=errordf.columns))
errordf = errordf.append(pd.DataFrame([[rmse_cv_train(lasso).mean(),rmse_cv_test(lasso).mean()]],index=['lasso'],columns=errordf.columns))
errordf = errordf.append(pd.DataFrame([[rmse_cv_train(rf).mean(),rmse_cv_test(rf).mean()]],index=['rf'],columns=errordf.columns))
#plt.ylim(8,14)
#plt.title("RMSE")
errordf.plot.bar(rot=0);
plt.ylim(0,2)
plt.title("RMSE")
