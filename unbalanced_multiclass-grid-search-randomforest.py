#Written By: Muhammad Rezaul Karim
# Used API Links:
#1. http://scikit-learn.org/stable/modules/classes.html#
#2. http://contrib.scikit-learn.org/imbalanced-learn/stable/api.html

#To run experiments in python:
#1. Download and install Anaconda: https://www.anaconda.com/download/
#It Install the necessary python libraries (e.g. scikit-learn, numpy, matplotlib etc.)
#2.Install imbalanced-learn python library seperately. Run the following command in the Anaconda command prompt:
# conda install -c glemaitre imbalanced-learn
#3. Open Anaconda Navigator. Launch 'jupyter notbook' to run python code


# This file is open for public re-use.
# With this file, you can perform machine learning based classification experiments with a non-stationary (not time-sensitive) 
# unbalanced data set. The last column in the data set must contain the dependent (class) variable, while all the other columns
# are independent (predictor) variables. 

#General Steps for Building Classification Models with a given data set:
#Step 1: Clean your data set (Impute missing values, remove samples with missing values etc.). Cleaning actions
# are problem specific, not performed in this file. Use an already cleaned data set for experimentation
#Step 2: Balance your data set. This code is written to work with unbalanced data sets. Use one of the balancing methods
#listed below. These techniques have their own parameters which can be tuned. By default one is selected
#Step 3: Transform your features (normalize, standardize etc.) to different range of values (e.g. [0,1])
#Step 4: Perform feature selection (reduce the number of features, keep important features etc.)
# Use one of the feature selection methods listed below. By default one is selected
#Step 5: Optimize hyper-parameters (Tune parameters) with K-fold stratified cross-validation
#Step 6: Build the actual models and perfrom evaluations with K-fold stratified cross-validation

######Important#############
# For handling missing Values: use 'sklearn.preprocessing.Imputer'
# Use 'sklearn.preprocessing.OneHotEncoder', (OneHotEncoder encoding) to feed categorical predictors to linear models 
# and SVMs with the standard kernels.

from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids

import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
import statistics

from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
    

#print(len(data[0,:]))
print("Process Started...")  

NUM_RUNS = 1 # The number of trials
NUM_JOBS=8  # The number of threads for parallelization. Each CPU core can handle 2 threads
seed=12345 # seed used by the random number generator. Used for re-producability

########################### Load Data Set From File################################################################
#Load data set with python panda read_csv method
dataSetFileName="C:/Users/r_kar/OneDrive/Pictures/Documents/training-dataset0.csv"
dataset1=pd.read_csv(dataSetFileName)
numOfrows, numOfColumns=dataset1.shape
print(dataset1.shape)  # number of rows and columns in the data set
 
dataset_data=dataset1.iloc[ : ,0:numOfColumns-1] #all predictor variable
dataset_target=dataset1.iloc[ : ,numOfColumns-1] # dependent variable. Assumption is that the last column contains 
#the dependent variable

# LabelEncoder will convert string class names into numeric class names (e.g. 0,1,2 etc.)
labelEncoder = preprocessing.LabelEncoder()
convertedIntoClasses=labelEncoder.fit_transform(list(dataset_target))
encodedClasses=np.unique(np.array(convertedIntoClasses)) # unique labels in the converted labels
print("New names for classes:",encodedClasses)
print("Actual names for classes:",labelEncoder.inverse_transform(encodedClasses))
print()

# Use the newly encoded class names
dataset_target=convertedIntoClasses

print("Count of Samples Per Class in unbalanced state:")  
print(sorted(Counter(dataset_target).items())) #count of different classes

###########################Resolve Class Imbalance (only one technique will be used)################################################################
#Class Imbalance Handling 1 (An Under-sampling Technique):  
#ClusterCentroids makes use of K-means to reduce the number of samples. Therefore, 
#each class will be synthesized with the centroids of the K-means method instead of the original samples:)
#imbalanceHandeler = ClusterCentroids(random_state=seed)
#X_resampled, y_resampled = imbalanceHandeler.fit_sample(dataset_data, dataset_target)
#print("Count of Samples Per Class in balanced state:")   
#print(sorted(Counter(y_resampled).items()))
#dataset_data = X_resampled
#dataset_target = y_resampled

##Class Imbalance Handling 2 (An Under-sampling Technique): RandomUnderSampler is a fast and easy way to balance the data 
#by randomly selecting a subset of data for the targeted classes:
imbalanceHandeler  = RandomUnderSampler(random_state=seed)
X_resampled, y_resampled = imbalanceHandeler.fit_sample(dataset_data, dataset_target)
print("Count of Samples Per Class in balanced state:")  
print(sorted(Counter(y_resampled).items()))
print()
dataset_data = X_resampled
dataset_target = y_resampled

# Class Imbalance Handling 3 (An Under-sampling Technique): NearMiss method implements 3 different types of heuristic 
# which can be selected with the parameter version (version=1, version=2, version=3):
#imbalanceHandeler =  NearMiss(random_state=seed, version=1,n_jobs=NUM_JOBS)
#X_resampled, y_resampled = imbalanceHandeler.fit_sample(dataset_data, dataset_target)
#print("Count of Samples Per Class in balanced state:") 
#print(sorted(Counter(y_resampled).items()))
#dataset_data = X_resampled
#dataset_target = y_resampled

#Class Imbalance Handling 4 (An Over-sampling Technique) by SMOTE (Synthetic Minority-Over Sampling Technique)
# k_neighbors: number of nearest neighbours to used to construct synthetic samples
# n_jobs: The number of threads to open if possible
#imbalanceHandeler =  SMOTE(random_state=seed, ratio='auto', kind='regular', k_neighbors=5, n_jobs=NUM_JOBS)
#X_resampled, y_resampled = imbalanceHandeler.fit_sample(dataset_data, dataset_target)
#print("Count of Samples Per Class in balanced state:") 
#print(sorted(Counter(y_resampled).items()))
#dataset_data = X_resampled
#dataset_target = y_resampled

#Class Imbalance Handling 5 (Combine over- and under-sampling using SMOTE and Tomek links). Perform over-sampling using SMOTE 
#and cleaning using Tomek links. Tomek method performs under-sampling by removing Tomek’s links.
# A Tomek’s link exist if the two samples are the nearest neighbors of each other
#smoteObject=SMOTE(random_state=seed, ratio='auto', kind='regular', k_neighbors=5, n_jobs=NUM_JOBS)
#tomekObject=TomekLinks(random_state=seed,ratio='auto', n_jobs=NUM_JOBS)
#imbalanceHandeler =  SMOTETomek(random_state=seed,ratio='auto', smote=smoteObject,tomek=tomekObject)
#X_resampled, y_resampled = imbalanceHandeler.fit_sample(dataset_data, dataset_target)
#print("Count of Samples Per Class in balanced state:") 
#print(sorted(Counter(y_resampled).items()))
#dataset_data = X_resampled
#dataset_target = y_resampled


# Please do not comment out this lines.
# This part is common for the above balancing methods. Executed after the balancing process.
X_dataset,Y_dataset = dataset_data,dataset_target
X_data, y_data = X_dataset[:, 0:len(X_dataset[0])], Y_dataset
print("Before feature selection. Note the number of predictors (second value)") 
print(X_data.shape)  #reduced data set number of rows and columns


#####################Feature transformation (only one of the following methods need to be used)############################################
#print("Feature values after transformation") 
#Feature Transformation 1: Standardize features to 0 mean and unit variance
#scaler = preprocessing.StandardScaler().fit(X_data)
#X_data_transformed = scaler.transform(X_data)
#X_data=X_data_transformed


#Feature Transformation 2: transforms features by scaling each feature to a [0,1] range.
scaler = preprocessing.MinMaxScaler().fit(X_data)
X_data_transformed = scaler.transform(X_data)
X_data=X_data_transformed


#Feature Transformation 3: Normalize samples individually to unit norm.
#scaler = preprocessing.Normalizer().fit(X_data)
#X_data_transformed = scaler.transform(X_data)
#X_data=X_data_transformed



################# Perform feature selection (only one of the following methods need to be used)########################################################
print("After feature selection. Note the number of predictors (second value)") 
# Feature Selection Method 1: Random Forest Based feature selection
clf = RandomForestClassifier(random_state=seed)
clf = clf.fit(X_data,y_data)
#print(clf.feature_importances_ ) 
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X_data)
print(X_new.shape)  #reduced data set number of rows and columns
X_data=X_new
#print(X_data)  # if you need to see the reduced data set

#Feature Selection Method 2: chi^2 test based top k feature selection
#X_new = SelectKBest(chi2, k=2).fit_transform(X_data, y_data)
#X_data=X_new
#print(X_new.shape)

#Feature Selection Method 3: Recursive feature elimination with cross-validation
#estimator = SVR(kernel="linear")
#selector = RFECV(estimator, step=1, cv=5)
#selector = selector.fit(X_data, y_data)
#X_new=selector.transform(X_data)
#X_data=X_new
#print(X_new.shape)


#################Tune parameters, build the actual models and perfrom evaluations #################


#This is for multi class classification
classifier = OneVsRestClassifier(RandomForestClassifier(random_state=seed))

#Max depth and max_features need to be less than the selected features. So check
#the number of selected features
parameters_grid = {
    "estimator__n_estimators": [10,20,50,100], # The number of trees in the forest.
    "estimator__max_depth": [2,4,8,12],    # maximum depth of each tree
    "estimator__max_features": [2,4,8, 12], # max features per random selection
   # "estimator__max_leaf_nodes":[2,3,5,7,10],  # max leaf nodes per tree. minimum 1
   # "estimator__min_samples_leaf":[2,5,7,10], # min # samples per leaf
}


#estimator: estimator object for GridSearchCV
    

# Arrays to store scores
grid_search_best_scores = np.zeros(NUM_RUNS)  # numpy arrays
final_evaluation_scores = np.zeros(NUM_RUNS)  #numpy arrays
#estimator: estimator object for GridSearchCV
    

# Loop for each trial
for i in range(NUM_RUNS):

    folds_for_grid_search = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
    folds_for_evaluation = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)

    # Parameter tuning with grid search and cross validation
    tuned_model = GridSearchCV(estimator=classifier, param_grid=parameters_grid, cv= folds_for_grid_search, scoring='f1_macro',
                               n_jobs=NUM_JOBS)     
    #n_jobs=Number of jobs to run in parallel
    tuned_model.fit(X_data, y_data)
    grid_search_best_scores[i] = tuned_model.best_score_
    
   # print (tuned_model.best_score_)
    print()
    print("Best Selected Parameters:")
    print(tuned_model.best_params_)
   

    # final cross validation for evaluation
    #temp_scores = cross_val_score(tuned_model, X=X_data, y=y_data, cv= folds_for_evaluation, scoring='f1_macro')
   # final_evaluation_scores[i] = temp_scores.mean()
    #print(temp_scores)
    #y_pred = cross_val_predict(tuned_model, X_data, y_data, cv=10)
    y_pred = cross_val_predict(tuned_model.best_estimator_, X_data, y_data, cv=folds_for_evaluation)
    # generate classification report
    print()
    print("Classification Results")
    print(classification_report(y_data, y_pred))
    

#print("Average difference of {0:6f} with std. dev. of {1:6f}."
 #     .format( final_evaluation_scores.mean(), final_evaluation_scores.std()))



