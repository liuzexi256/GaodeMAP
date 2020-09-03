""" =======================  Import dependencies ========================== """
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm
import matplotlib.image as mpimg
from scipy.stats import pearsonr
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, VarianceThreshold, SelectKBest, chi2
from sklearn.decomposition import PCA, SparsePCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import json
from yolo import YOLO
from PIL import Image
import os

""" ======================  Function definitions ========================== """
def LGB(clf, train_x, train_y, test_x):
    #predictors = list(train_x.columns)
    train_x = train_x.values
    test_x = test_x.values
    folds = 5
    seed = 202
    kf = StratifiedKFold(n_splits = folds, shuffle = True, random_state = seed)
    #kf = KFold(n_splits = folds, shuffle = True, random_state = seed)
    train = np.zeros((train_x.shape[0], 3))
    test = np.zeros((test_x.shape[0], 3))
    test_pre = np.zeros((folds, test_x.shape[0], 3))
    test_pre_all = np.zeros((folds, test_x.shape[0]))
    cv_scores = []
    f1_scores = []
    cv_rounds = []

    for i, (train_index, verify_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        ve_x = train_x[verify_index]
        ve_y = train_y[verify_index]

        train_matrix = clf.Dataset(tr_x, label = tr_y)
        verify_matrix = clf.Dataset(ve_x, label = ve_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            #'metric': 'None',
            'metric': 'multi_logloss',
            'class_weight': 'balanced',
            'min_child_weight': 5.9,
            'num_leaves': 15,
            'lambda_l2': 1,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.8,
            'bagging_freq': 18,
            'learning_rate': 0.18,
            #'seed': 2019,
            'nthread': 6,
            'num_class': 3,
            'verbose': -1,
        }

        num_round = 5000
        early_stopping_rounds = 1000
        if verify_matrix:
            model = clf.train(params, train_matrix, num_round, 
                              valid_sets = verify_matrix, 
                              verbose_eval = 50,
                              #feval=acc_score_vali,
                              early_stopping_rounds = early_stopping_rounds
                              )
            #print("\n".join(("%s: %.2f" % x) for x in
            #                list(sorted(zip(predictors, model.feature_importance("gain")), key=lambda x: x[1],
            #                       reverse=True))[:200]
            #                ))
            verify_res = model.predict(ve_x, 
                                        num_iteration = model.best_iteration
                                        )
            pred = model.predict(test_x, num_iteration = model.best_iteration)
            train[verify_index] = verify_res
            test_pre[i, :] = pred
            ##cv_scores.append(log_loss(te_y, pre))
            pre_y = np.argmax(verify_res, axis = 1)
            f1_list = f1_score(ve_y, pre_y, average = None)
            f1 = 0.2*f1_list[0] + 0.2*f1_list[1] + 0.6*f1_list[2]
            
            f1_scores.append(f1)
            ##cv_rounds.append(model.best_iteration)
            test_pre_all[i, :] = np.argmax(pred, axis=1)


        #print("%s now score is:" % clf_name, cv_scores)
        #print("%s now f1-score is:" % clf_name, f1_scores)
        #print("%s now round is:" % clf_name, cv_rounds)
    f1_mean = np.mean(f1_scores)
    test[:] = test_pre.mean(axis = 0)
    y_test = np.argmax(test, axis = 1)

    return test

""" =======================  Load Training Data ======================= """
path = "I:/TianChi/data/"   #存放数据的地址
result_path = "I:/TianChi/data/"   #存放数据的地址
train_json = pd.read_json(path + "train_target.json")
test_json = pd.read_json(path + "test_target.json")

train_features = pd.read_csv(path + 'train_feature3000_rate0.5_w3_tr0.5_new.csv')
test_features = pd.read_csv(path + 'test_feature3000_rate0.5_w3_tr0.5_new.csv')
train_df = pd.read_csv(path + 'train_feature3000_timecn_new.csv')
test_df = pd.read_csv(path + 'test_feature3000_timecn_new.csv')
X = pd.read_csv(path + 'train_feature3000_fromJ.csv')
X_test = pd.read_csv(path + 'test_feature3000_fromJ.csv')

""" ======================  Variable Declaration ========================== """
select_features = [
                    "rate1_mean",
                    "rate1_std",
                    "rate2_mean",
                    "rate2_std",
                    "number1_mean",
                    "number1_std",
                    "number2_mean",
                    "number2_std",
                    'square_mean', 
                    'square_std',
                    #'number_mean', 
                    #'number_std',
                    'car1_mean',
                    'car2_mean',
                    'truck1_mean',
                    'truck2_mean',
                    #'bus1_mean',
                    #'bus2_mean',
                    #'motorbike1_mean',
                    #'motorbike2_mean',
                    #'bicycle1_mean',
                    #'bicycle2_mean',
                    "gap_mean",
                    "gap_std",
                    "hour_mean",
                    "minute_mean",
                    "dayofweek_mean",
                    "gap_time_today_mean",
                    "gap_time_today_std",
                    '1','2','3',
                    "im_diff_mean_mean","im_diff_mean_std","im_diff_std_mean","im_diff_std_std",
                  ]

""" ========================  Extract Features ============================= """
train_features['number1'] = train_features['car1'] + train_features['truck1'] + train_features['bus1'] + train_features['motorbike1'] + train_features['bicycle1']
train_features['number2'] = train_features['car2'] + train_features['truck2'] + train_features['bus2'] + train_features['motorbike2'] + train_features['bicycle2']
test_features['number1'] = test_features['car1'] + test_features['truck1'] + test_features['bus1'] + test_features['motorbike1'] + test_features['bicycle1']
test_features['number2'] = test_features['car2'] + test_features['truck2'] + test_features['bus2'] + test_features['motorbike2'] + test_features['bicycle2']

train_features['square'] = train_features['rate1'] + train_features['rate2']
test_features['square'] = test_features['rate1'] + test_features['rate2']
train_features['number'] = train_features['number1'] + train_features['number2']
test_features['number'] = test_features['number1'] + test_features['number2']

train_features = train_features.groupby("map_id1").agg({
                                                        "rate1":["mean","std"],
                                                        "rate2":["mean","std"],
                                                        "number1":["mean","std"],
                                                        "number2":["mean","std"],
                                                        "square":["mean","std"],
                                                        'number':['mean','std'],
                                                        'car1':['mean'],'car2':['mean'],'truck1':['mean'],'truck2':['mean'],'bus1':['mean'],'bus2':['mean'],'motorbike1':['mean'],'motorbike2':['mean'],'bicycle1':['mean'],'bicycle2':['mean'],
                                                        "label":["mean"],
                                                        }).reset_index()

test_features = test_features.groupby("map_id1").agg({
                                                        "rate1":["mean","std"],
                                                        "rate2":["mean","std"],
                                                        "number1":["mean","std"],
                                                        "number2":["mean","std"],
                                                        "square":["mean","std"],
                                                        'number':['mean','std'],
                                                        'car1':['mean'],'car2':['mean'],'truck1':['mean'],'truck2':['mean'],'bus1':['mean'],'bus2':['mean'],'motorbike1':['mean'],'motorbike2':['mean'],'bicycle1':['mean'],'bicycle2':['mean'],
                                                        "label":["mean"],
                                                        }).reset_index()
train_features.columns = [
                            "map_id1",
                            "rate1_mean","rate1_std","rate2_mean","rate2_std",
                            "number1_mean","number1_std","number2_mean","number2_std",
                            'square_mean', 'square_std',
                            'number_mean', 'number_std',
                            'car1_mean','car2_mean','truck1_mean','truck2_mean','bus1_mean','bus2_mean','motorbike1_mean','motorbike2_mean','bicycle1_mean','bicycle2_mean',
                            #'1','2','3','4','5','6','7','8','9','10','11',
                            "label"]
test_features.columns = [
                            "map_id1",
                            "rate1_mean","rate1_std","rate2_mean","rate2_std",
                            "number1_mean","number1_std","number2_mean","number2_std",
                            'square_mean', 'square_std',
                            'number_mean', 'number_std',
                            'car1_mean','car2_mean','truck1_mean','truck2_mean','bus1_mean','bus2_mean','motorbike1_mean','motorbike2_mean','bicycle1_mean','bicycle2_mean',
                            "label"]
'''
train_df = get_data(train_json[:],"amap_traffic_train_0712")
test_df = get_data(test_json[:],"amap_traffic_test_0712")
test_df.to_csv(path_or_buf = path + 'test_feature_timecn_new.csv')
train_df.to_csv(path_or_buf = path + 'train_feature_timecn_new.csv')
'''

train_features = pd.concat([train_features, train_df, X], axis = 1)
test_features = pd.concat([test_features, test_df, X_test], axis = 1)

train_features["label"] = train_features["label"].apply(int)
test_features["label"] = test_features["label"].apply(int)


train_x = train_features[select_features].copy()
train_y = train_features["label"]
test_x = test_features[select_features].copy()

""" ========================  Select Features ============================= """
#train_x = MinMaxScaler().fit_transform(train_x)

#方差选择
'''
selector = VarianceThreshold()
selector.fit_transform(train_x)
var = selector.variances_
plt.bar(select_features, var)
#plt.show()
'''
#相关系数法
'''
kbest = SelectKBest(chi2, k = 10)
kbest.fit_transform(abs(train_x), train_y)
a = kbest.scores_
plt.bar(select_features, a)
plt.show()
'''

""" ========================  Oversampling ============================= """

over_samples = BorderlineSMOTE(random_state = 2020) 
over_samples_x, over_samples_y = over_samples.fit_sample(train_x, train_y)
over_samples_x = pd.DataFrame(over_samples_x)
over_samples_x.columns = select_features
#print(pd.Series(over_samples_y).value_counts()/len(over_samples_y))


""" ========================  Decomposition  ============================= """
'''
pca = PCA(n_components = 'mle')
#pca = SparsePCA(n_components = 15)
pca.fit(over_samples_x)
train_x_new = pca.transform(over_samples_x)
test_x_new = pca.transform(test_x)
'''
'''
pca = PCA(n_components = 'mle')
#pca = SparsePCA(n_components = 15)
pca.fit(train_x)
train_x = pca.transform(train_x)
test_x = pca.transform(test_x)
'''

""" ========================  Grid Search ============================= """
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
#test_y = stacking(over_samples_x, over_samples_y, test_x)
#test_y, f1 = lgb(train_x_new, over_samples_y, test_x_new)
test_y = LGB(train_x, train_y, test_x)


a = 0