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
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, VarianceThreshold, SelectKBest, chi2
from sklearn.decomposition import PCA, SparsePCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import json
from yolo import YOLO
from PIL import Image
import os

""" ======================  Function definitions ========================== """
def XGBoost(train_x, train_y, test_x):
    predictors = list(train_x.columns)
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
    f1_scores = []
    verify = np.zeros((3000, 3))
    for i, (train_index, verify_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        ve_x = train_x[verify_index]
        ve_y = train_y[verify_index]

        if 1:
            model = XGBClassifier(
                                    objective = 'multi:softprob',
                                    metric = 'merror',
                                    num_class = 3,
                                    n_estimators = 1000, 
                                    max_depth = 4,
                                    min_child_weight = 1, 
                                    learning_rate= 0.4, 
                                    subsample = 1,
                                    colsample_bytree = 1,
                                    #reg_alpha = 0,
                                    #reg_lambda = 0,
                                    #scale_pos_weight = 5,
                                    random_state = 2019,
                                    #verbosity = 1, 
                                    )
            model.fit(
                        tr_x, 
                        tr_y,
                        eval_set = [(ve_x, ve_y)],
                        #eval_metric = 'mlogloss',
                        early_stopping_rounds = 500,
                        verbose = True
                        )
            verify_res = model.predict_proba(ve_x)
            pred = model.predict_proba(test_x)
            train[verify_index] = verify_res
            test_pre[i, :] = pred

            pre_y = np.argmax(verify_res, axis = 1)
            f1_list = f1_score(ve_y, pre_y, average = None)
            f1 = 0.2*f1_list[0] + 0.2*f1_list[1] + 0.6*f1_list[2]
            
            f1_scores.append(f1)
            test_pre_all[i, :] = np.argmax(pred, axis=1)
            for i, idx in enumerate(verify_index):
                verify[idx] = verify_res[i]
    f1_mean = np.mean(f1_scores)
    test[:] = test_pre.mean(axis = 0)
    y_test = np.argmax(test, axis = 1)

    return verify, test, f1_mean

def AdaBoost(train_x, train_y, test_x):
    predictors = list(train_x.columns)
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
    f1_scores = []
    verify = np.zeros((3000, 3))
    for i, (train_index, verify_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        ve_x = train_x[verify_index]
        ve_y = train_y[verify_index]

        if 1:
            model = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 10, min_samples_split = 2),
                                                                algorithm = "SAMME",
                                                                n_estimators = 200, learning_rate = 0.8)
            model.fit(tr_x, tr_y)
            verify_res = model.predict_proba(ve_x)
            pred = model.predict_proba(test_x)
            #train[verify_index] = verify_res
            test_pre[i, :] = pred

            pre_y = np.argmax(verify_res, axis = 1)
            f1_list = f1_score(ve_y, pre_y, average = None)
            f1 = 0.2*f1_list[0] + 0.2*f1_list[1] + 0.6*f1_list[2]
            
            f1_scores.append(f1)
            #test_pre_all[i, :] = np.argmax(pred, axis=1)
            for i, idx in enumerate(verify_index):
                verify[idx] = verify_res[i]
    f1_mean = np.mean(f1_scores)
    test[:] = test_pre.mean(axis = 0)
    y_test = np.argmax(test, axis = 1)

    return verify, test, f1_mean

def RF(train_x, train_y, test_x):
    predictors = list(train_x.columns)
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
    f1_scores = []
    verify = np.zeros((3000, 3))

    for i, (train_index, verify_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        ve_x = train_x[verify_index]
        ve_y = train_y[verify_index]

        if 1:
            model = RandomForestClassifier(
                                            n_estimators = 600, oob_score = True, 
                                            #min_weight_fraction_leaf = 0,
                                            #class_weight = 'balanced'
                                            )
            model.fit(tr_x, tr_y)
            verify_res = model.predict_proba(ve_x)
            pred = model.predict_proba(test_x)
            test_pre[i, :] = pred

            pre_y = np.argmax(verify_res, axis = 1)
            f1_list = f1_score(ve_y, pre_y, average = None)
            f1 = 0.2*f1_list[0] + 0.2*f1_list[1] + 0.6*f1_list[2]
            
            f1_scores.append(f1)
            for i, idx in enumerate(verify_index):
                verify[idx] = verify_res[i]

    f1_mean = np.mean(f1_scores)
    test[:] = test_pre.mean(axis = 0)
    y_test = np.argmax(test, axis = 1)

    return verify, test, f1_mean


def ET(train_x, train_y, test_x):
    predictors = list(train_x.columns)
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
    f1_scores = []
    verify = np.zeros((3000, 3))
    for i, (train_index, verify_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        ve_x = train_x[verify_index]
        ve_y = train_y[verify_index]

        if 1:
            model = ExtraTreesClassifier(n_estimators=800, bootstrap=True, oob_score=True)
            model.fit(tr_x, tr_y)
            verify_res = model.predict_proba(ve_x)
            pred = model.predict_proba(test_x)
            #train[verify_index] = verify_res
            test_pre[i, :] = pred

            pre_y = np.argmax(verify_res, axis = 1)
            f1_list = f1_score(ve_y, pre_y, average = None)
            f1 = 0.2*f1_list[0] + 0.2*f1_list[1] + 0.6*f1_list[2]
            
            f1_scores.append(f1)
            test_pre_all[i, :] = np.argmax(pred, axis=1)
            for i, idx in enumerate(verify_index):
                verify[idx] = verify_res[i]

    f1_mean = np.mean(f1_scores)
    test[:] = test_pre.mean(axis = 0)
    y_test = np.argmax(test, axis = 1)

    return verify, test, f1_mean

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
    f1_scores = []
    verify = np.zeros((3000, 3))

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
        '''
        params = {
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            #'metric': 'None',
            'metric': 'multi_logloss',
            'class_weight': 'balanced',
            'min_child_weight': 1,
            'num_leaves': 22,
            'lambda_l2': 20,
            'feature_fraction': 0.5,
            'bagging_fraction': 1,
            'bagging_freq': 30,
            'learning_rate': 1,
            'max_depth': 3,
            #'seed': 2019,
            'min_data_in_leaf': 1,
            'nthread': 6,
            'num_class': 3,
            'verbose': -1,
        }
        '''
        num_round = 6000
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
            pre_y = np.argmax(verify_res, axis = 1)
            f1_list = f1_score(ve_y, pre_y, average = None)
            f1 = 0.2*f1_list[0] + 0.2*f1_list[1] + 0.6*f1_list[2]
            
            f1_scores.append(f1)
            test_pre_all[i, :] = np.argmax(pred, axis=1)
            for i, idx in enumerate(verify_index):
                verify[idx] = verify_res[i]

    f1_mean = np.mean(f1_scores)
    test[:] = test_pre.mean(axis = 0)
    y_test = np.argmax(test, axis = 1)

    return verify, test, f1_mean

def stacking(x_train, y_train, x_test):
    verify_LGB, test_LGB, f1_LGB = LGB(lightgbm, x_train, y_train, x_test)
    verify_RF, test_RF, f1_RF = RF(x_train, y_train, x_test)
    verify_ET, test_ET, f1_ET = ET(x_train, y_train, x_test)
    verify_ADB, test_ADB, f1_ADB = AdaBoost(x_train, y_train, x_test)
    verify_XGB, test_XGB, f1_XGB = XGBoost(x_train, y_train, x_test)

    train_features = np.concatenate((verify_LGB, verify_RF, verify_ET, verify_ADB, verify_XGB), axis = 1)
    test_features = np.concatenate((test_LGB, test_RF, test_ET, test_ADB, test_XGB), axis = 1)
    '''
    lr = LogisticRegression(
                            multi_class = 'ovr',
                            random_state = 2020,
                            )
    lr.fit(train_features, y_train)
    y_test = lr.predict(test_features)
    '''
    #svm = LinearSVC(dual = False, class_weight = 'balanced')
    #svm.fit(train_features, y_train)
    #y_test = svm.predict(test_features)
    
    test = test_ADB + test_ET + test_LGB + test_RF + test_XGB
    y_test = np.argmax(test, axis = 1)

    res = np.zeros(3)
    for i in range(len(y_test)):
        if y_test[i] == 0:
            res[0] = res[0] + 1
        if y_test[i] == 1:
            res[1] = res[1] + 1
        if y_test[i] == 2:
            res[2] = res[2] + 1
    f1 = 1
    return y_test, f1

""" =======================  Load Training Data ======================= """
path = "G:/04_Study/TianChi/data/"   #存放数据的地址
result_path = "G:/04_Study/TianChi/data/"   #存放数据的地址
train_json = pd.read_json(path + "train_target.json")
test_json = pd.read_json(path + "test_new_target.json")
label_new = np.load(path + 'label_new.npy')

train_features = pd.read_csv(path + 'train_feature3000_rate0.25_w3_tr0.5_new.csv')
test_features = pd.read_csv(path + 'test_new_feature3000_rate0.25_w3_tr0.5_new.csv')
train_df = pd.read_csv(path + 'train_feature3000_timecn_new.csv')
test_df = pd.read_csv(path + 'test_new_feature3000_timecn_new.csv')
X = pd.read_csv(path + 'train_feature3000_fromJ_new.csv')
X_test = pd.read_csv(path + 'test_new_feature3000_fromJ_new.csv')
train_features_fromM = pd.read_csv(path + 'train_feature1500_fromM.csv')
test_features_fromM = pd.read_csv(path + 'test_feature1500_fromM.csv')

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
                    #"gap_mean",
                    #"gap_std",
                    #"hour_mean",
                    #"minute_mean",
                    #"dayofweek_mean",
                    #"gap_time_today_mean",
                    #"gap_time_today_std",
                    '1','2','3',
                    #"im_diff_mean_mean","im_diff_mean_std","im_diff_std_mean","im_diff_std_std",
                    #'a',
                    #'b','c','d',
                    #'e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v',
                  ]

""" ========================  Extract Features ============================= """
#train_features['rate1'] = train_features['rate1']*100000
#train_features['rate2'] = train_features['rate2']*100000

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

train_features = pd.concat([train_features, train_df, X, train_features_fromM], axis = 1)
test_features = pd.concat([test_features, test_df, X_test, test_features_fromM], axis = 1)

train_features["label"] = train_features["label"].apply(int)
test_features["label"] = test_features["label"].apply(int)


train_x = train_features[select_features].copy()
train_y = train_features["label"]
#train_y = label_new
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

""" ========================  Train the Model ============================= """

#test_y = stacking(over_samples_x, over_samples_y, test_x)
#test_y, f1 = lgb(train_x_new, over_samples_y, test_x_new)
test_y, f1 = stacking(train_x, train_y, test_x)

""" ========================  Submit the result ============================= """
sub = test_features[['map_id']].copy()
sub['pred'] = test_y
result_dic = dict(zip(sub["map_id"],sub["pred"]))

with open(path+"submmit.json","r") as f:
    content = f.read()
content = json.loads(content)
for i in content["annotations"]:
    i['status'] = result_dic[int(i["id"])]
with open(result_path+"sub_%s.json"%f1,"w") as f:
    f.write(json.dumps(content))

a = 0