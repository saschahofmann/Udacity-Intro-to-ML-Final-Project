#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from matplotlib import pyplot as plt
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# New imports
import numpy as np
from timeit import default_timer as timer
from sklearn.preprocessing import MinMaxScaler
from tester import test_classifier    
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
## Task 2: Remove outliers   
print 'Outlier \'TOTAL\' has been removed.\n'
data_dict.pop('TOTAL',0)

### Task 3: Create new feature(s)
# create new features: fraction of messages to and from pois
print 'Create new features: comm_with_poi.  \n'

for key in data_dict.keys():
    if data_dict[key]['from_poi_to_this_person'] != 'NaN':
        fraction_from_poi = float(data_dict[key]['from_poi_to_this_person'])/data_dict[key]['from_messages']
        fraction_to_poi = float(data_dict[key]['from_this_person_to_poi'])/data_dict[key]['to_messages']
        data_dict[key]['comm_with_poi'] = fraction_from_poi+fraction_to_poi

    else:
        data_dict[key]['comm_with_poi'] = 0
        
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'expenses', 'exercised_stock_options', 
                 'deferral_payments',  'long_term_incentive',
                 'other', 'restricted_stock', 'restricted_stock_deferred',
                 'deferred_income','comm_with_poi','shared_receipt_with_poi','director_fees',]   
features_list_used = ['poi', 'salary', 'bonus',  'exercised_stock_options', 
 'long_term_incentive', 'comm_with_poi', 'shared_receipt_with_poi', 
  'expenses','director_fees'] #  'deferred_income',

print ('Start feature selection. , \'to_messages\', \'from_messages\', \'from_this_person_to_poi\', \'from_poi_to_this_person\' '
       + ' and \'total_stock_vallue\' have been excluded from selection because of a strong correlation with other features,'
       + ' \'loan_advances\' and \'total_payments\' because they isolate only one or few data points. \n')

k = 5
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN = True)
labels, features = targetFeatureSplit(data)


print 'Features are scaled with the MinMaxScaler for PCA analysis.'


scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)


pca = PCA()
features_pca = pca.fit_transform(scaled_features)
idx = np.argsort(abs(pca.components_[0]))
new = [features_list[i+1] for i in idx]
new.reverse()
new = new[:k]

print '%d strongest features in first PCA component:' % (k)
print  '\t{:30s}{:s}'.format('feature', 'contribution to first PCA component')
print '\t'+'-'*66
for i in xrange(len(new[:k])):
    print '\t{:30s}{:0.3f}'.format(new[i], np.sort(abs(pca.components_[0]))[::-1][i])
print'\n'
#labels_arr = np.asarray(labels)
#plt.scatter(features_pca[:,0][labels_arr == 1.0], features_pca[:,1][labels_arr == 1.0])
#plt.scatter(features_pca[:,0][labels_arr == 0.0], features_pca[:,1][labels_arr == 0.0])

print 'Select %d best features using sklearns SelectKBest:' % (k) 
print '\t{:27s}{:s}'.format('feature', 'score')
print '\t'+'-'*33

selector = SelectKBest(score_func = f_classif, k = k)
selector.fit(scaled_features, labels)
# New Features List

df = pd.Series(data = selector.scores_,  index = features_list[1:] )
df.sort_values(ascending = False).to_csv( 'feature_scores.csv', index = False)
print '\t{}\n'.format(df.sort_values(ascending = False)[:k].to_string().replace("\n", "\n\t"))

# Write new feature list
tmp_list = []
for i in xrange(len(features_list[1:])):
    if selector.get_support()[i]:
        tmp_list.append(features_list[i+1])
for feature in new:
    if feature not in tmp_list:
        tmp_list.append(feature)
        
       
features_list[1:] = tmp_list
features_list.append('comm_with_poi')
features_list.remove('deferred_income')
features_list.remove('director_fees')
print 'The new features_list contains the %d highest scoring  features from SelectKBest \
and the first PCA component. The newly created feature \'comm_with_poi\' is \
added and \'deferred_income\' and \'director_fees\' removed.' %(k)
print 'The new features_list = ', features_list

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN = True)
labels, features = targetFeatureSplit(data)
palette = {0 : 'blue', 1 : 'red'}
labels_c = map(lambda x: palette[int(x)], labels)
data_frame = pd.DataFrame(features, columns=features_list[1:])
grr = pd.scatter_matrix(data_frame, alpha=0.8,c = labels_c)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Splitting in training and testing
print 'Trying different classifiers from Sklearn:'


print '\nTry Support Vector Machine:'
clf = SVC()
test_classifier(clf, data_dict, features_list)

print '\nTry Decision Tree classifier:'
clf = DecisionTreeClassifier()#(criterion= 'gini', max_depth = 10, min_samples_split = 5,)
test_classifier(clf, data_dict, features_list)

print '\nTry AdaBoost classifier:'
clf = AdaBoostClassifier(DecisionTreeClassifier())#, learning_rate = 1.0, n_estimators = 2)
test_classifier(clf, data_dict, features_list)

print '\nTry RandomForest Classifier:'
clf = RandomForestClassifier()
test_classifier(clf, data_dict, features_list)

print '\nThe DecisionTree and AdaBoost classifiers already meet the asked \
precision and recall. The DecisionTree seems to have a slightly better performance.'





features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# Provided to give you a starting point. Try a variety of classifiers.

clf_list = [] # [ 'Decision Tree', 'RandomForest', ]  #AdaBoost
print '\nStart Algorithm tuning for %s and %s:'  %(clf_list[0], clf_list[1])
for algorithm in clf_list:
    if algorithm == 'SVM':
        clf = SVC()
        parameters = {'kernel':('linear', 'rbf'),}# 'gamma':   ['auto', 1.0, 100]}
    elif algorithm == 'Decision Tree':
        clf = DecisionTreeClassifier()
        parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                      'max_depth': [None, 5, 10, 50], 'min_samples_split': [5, 10, 15, 20, 30]}
    elif algorithm =='RandomForest':
        clf = RandomForestClassifier()
        parameters = {'n_estimators': [2,5,10,50], 'min_samples_split': [2,5,10,50],
        'criterion': ['entropy', 'gini'], 'max_depth': [5, 10,50]}
    elif algorithm =='Adaboost':
        clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
        parameters = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
               'base_estimator__max_depth': [None, 5, 10,50] ,
               'base_estimator__min_samples_split': [2, 5, 10, 15, 20,50],
              "n_estimators": [1, 2, 5,15, 50],  'learning_rate': [ .5, 1.0, 2.0]}
        
    print '\tTesting %s:' % (algorithm)
    clf = GridSearchCV(clf, parameters, scoring = 'recall', cv = 3)
    clf.fit(features_train, labels_train)
    clf_best = clf.best_estimator_
    clf_best.fit(features_train, labels_train)
    
    labels_pred = clf_best.predict(features_test)
    accuracy = accuracy_score(labels_test, labels_pred)
    precision = precision_score(labels_test, labels_pred)
    recall = recall_score(labels_test, labels_pred)
    print '\t\t',algorithm, accuracy, recall, precision
    print '\t\t', clf.best_params_, '\n' 
    



acc, prec, rec, fi = [], [], [], []
for i in xrange(10):
    clf = DecisionTreeClassifier(criterion = 'entropy')
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    clf.fit(features_train, labels_train)  
    labels_pred = clf.predict(features_test)
    acc.append(accuracy_score(labels_test, labels_pred))
    prec.append(precision_score(labels_test, labels_pred))
    rec.append(recall_score(labels_test, labels_pred))
    fi.append(clf.feature_importances_)
print '\nThe Sklearn Decision Tree with default settings has an accuracy of {:0.3f} \
, a precision of {:0.3f} and a recall of {:0.3f}.'.format(np.mean(acc), np.mean(prec), np.mean(rec))
print '\t{:32s}{:s}'.format('feature', 'feature importance')
print '\t'+'-'*51
for i in xrange(len(features_list[1:])):
    print '\t{:32s}{:0.6f}'.format(features_list[i+1], np.mean(fi, axis =0)[i])

features_list.remove('comm_with_poi')
acc, prec, rec, fi = [], [], [], []
for i in xrange(10):
    clf_ = DecisionTreeClassifier(criterion = 'entropy')
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)
    clf_.fit(features_train, labels_train)  
    labels_pred = clf_.predict(features_test)
    acc.append(accuracy_score(labels_test, labels_pred))
    prec.append(precision_score(labels_test, labels_pred))
    rec.append(recall_score(labels_test, labels_pred))
    fi.append(clf_.feature_importances_)
print '\nThe Decision Tree without the newly created feature \'comm_with_poi\' has an accuracy of {:0.3f} \
, a precision of {:0.3f} and a recall of {:0.3f}.'.format(np.mean(acc), np.mean(prec), np.mean(rec))
print '\t{:32s}{:s}'.format('feature', 'feature importance')
print '\t'+'-'*51
for i in xrange(len(features_list[1:])):
    print '\t{:32s}{:0.6f}'.format(features_list[i+1], np.mean(fi, axis =0)[i])
    
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
    

dump_classifier_and_data(clf, my_dataset, features_list)