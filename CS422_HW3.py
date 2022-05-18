
#Lucas Allen, 5004607031, HW #3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import seed, shuffle, randrange
from sklearn import preprocessing 

#data input and formatting
df = pd.read_csv('auto-mpg.data.csv')
df[['origin', 'name']] = df.carname.str.split(pat = '\t', expand = True)
df = df.drop('carname', axis = 1)
df = df.drop('name', axis = 1)
# print(df.dtypes)
# print(df.tail())
int_origins = np.array(df['origin'])
df['origin'] = int_origins.astype(float)
# print(df.dtypes)
# print(df.describe())
# print(df.head())

#credit Dr.Kang, find b_opt for X, y
def SolverLinearRegression(X, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), y)

# Used to perform K_fold cross validation, defaults to 3 folds unless other param. passed in
def KFold_CV(dataset, folds = 3): 
    from random import randrange
    my_split = list()
    my_copy = list(dataset)
    fold_size = int(len(dataset)/folds)
    for i in range(folds): 
        fold = list()
        while len(fold) < fold_size: 
            index = randrange(len(my_copy))
            fold.append(my_copy.pop(index))
        my_split.append(fold)
    return my_split

# Performs linear regression solver on split datasets to find coefficients. Also finds RMSE for each fold and 
# indpendent variable
def find_coefficients_after_KFold():
    seed (1)
    # to find 10 folds of coefficient for cylinder
    X = preprocessing.scale(df.iloc[:,1])
    y = df['mpg']
    y = preprocessing.scale(y)
    
    dataset = X 
    X_folds = np.array(KFold_CV(dataset, folds = 10))
    y_folds = np.array(KFold_CV(y, folds = 10))
    # print(folds)
    for i in range(10): 
        X = pd.DataFrame(X_folds[i])
        y = pd.DataFrame(y_folds[i])  
        # print(y)
        b_opt = SolverLinearRegression(X,y)
        rmse = (np.sum((np.dot(X, b_opt) - y)**2)/len(X))**0.5
        print(f'cylin.coeff fold', i, b_opt, 'rmse', rmse)
    print('\n')

    # to find 10 folds of coefficient for displacement
    X = preprocessing.scale(df.iloc[:,2])
    y = df['mpg']
    y = preprocessing.scale(y)
    dataset = X 
    X_folds = np.array(KFold_CV(dataset, folds = 10))
    y_folds = np.array(KFold_CV(y, folds = 10))
    # print(folds)
    for i in range(10): 
        X = pd.DataFrame(X_folds[i])
        y = pd.DataFrame(y_folds[i])    
        b_opt = SolverLinearRegression(X,y)
        rmse = (np.sum((np.dot(X, b_opt) - y)**2)/len(X))**0.5
        print(f'displ. coeff fold', i, b_opt, 'rmse', rmse)
    print('\n')

    # to find 10 folds of coefficient for horsepower
    X = preprocessing.scale(df.iloc[:,3])
    y = df['mpg']
    y = preprocessing.scale(y)
    dataset = X 
    X_folds = np.array(KFold_CV(dataset, folds = 10))
    y_folds = np.array(KFold_CV(y, folds = 10))
    # print(folds)
    for i in range(10): 
        X = pd.DataFrame(X_folds[i])
        y = pd.DataFrame(y_folds[i])    
        b_opt = SolverLinearRegression(X,y)
        rmse = (np.sum((np.dot(X, b_opt) - y)**2)/len(X))**0.5
        print(f'horsePow. coeff fold', i, b_opt, 'rmse', rmse)
    print('\n')

    # to find 10 folds of coefficient for weight
    X = preprocessing.scale(df.iloc[:,4])
    y = df['mpg']
    y = preprocessing.scale(y)
    dataset = X 
    X_folds = np.array(KFold_CV(dataset, folds = 10))
    y_folds = np.array(KFold_CV(y, folds = 10))
    # print(folds)
    for i in range(10): 
        X = pd.DataFrame(X_folds[i])
        y = pd.DataFrame(y_folds[i])   
        b_opt = SolverLinearRegression(X,y)
        rmse = (np.sum((np.dot(X, b_opt) - y)**2)/len(X))**0.5
        print(f'weight coeff fold', i, b_opt, 'rmse', rmse)
    print('\n')

    # to find 10 folds of coefficient for acceleration
    X = preprocessing.scale(df.iloc[:,5])
    y = df['mpg']
    y = preprocessing.scale(y)
    dataset = X 
    X_folds = np.array(KFold_CV(dataset, folds = 10))
    y_folds = np.array(KFold_CV(y, folds = 10))
    # print(folds)
    for i in range(10): 
        X = pd.DataFrame(X_folds[i])
        y = pd.DataFrame(y_folds[i])    
        b_opt = SolverLinearRegression(X,y)
        rmse = (np.sum((np.dot(X, b_opt) - y)**2)/len(X))**0.5
        print(f'acceler. coeff fold', i, b_opt, 'rmse', rmse)
    print('\n')

    # to find 10 folds of coefficient for model_year
    X = preprocessing.scale(df.iloc[:,6])
    y = df['mpg']
    y = preprocessing.scale(y)
    dataset = X 
    X_folds = np.array(KFold_CV(dataset, folds = 10))
    y_folds = np.array(KFold_CV(y, folds = 10))
    # print(folds)
    for i in range(10): 
        X = pd.DataFrame(X_folds[i])   
        y = pd.DataFrame(y_folds[i]) 
        b_opt = SolverLinearRegression(X,y)
        rmse = (np.sum((np.dot(X, b_opt) - y)**2)/len(X))**0.5
        print(f'Model_year coeff fold', i, b_opt, 'rmse', rmse)
    print('\n')

    # to find 10 folds of coefficient for origin
    X = preprocessing.scale(df.iloc[:,7])
    y = df['mpg']
    y = preprocessing.scale(y)
    dataset = X 
    X_folds = np.array(KFold_CV(dataset, folds = 10))
    y_folds = np.array(KFold_CV(y, folds = 10))
    # print(folds)
    for i in range(10): 
        X = pd.DataFrame(X_folds[i])
        y = pd.DataFrame(y_folds[i])    
        b_opt = SolverLinearRegression(X,y)
        rmse = (np.sum((np.dot(X, b_opt) - y)**2)/len(X))**0.5
        print(f'orgin coeff fold', i, b_opt, 'rmse', rmse)
    print('\n')

find_coefficients_after_KFold()