import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, SparsePCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
import json
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier

""" =======================  Load Training Data ======================= """
path = "I:/TianChi/data/"   #存放数据的地址
result_path = "I:/TianChi/data/"   #存放数据的地址
train_json = pd.read_json(path + "train_target.json")
test_json = pd.read_json(path + "test_target.json")
X = np.load(path + 'X.npy')
X_test = np.load(path + 'X_test.npy')
X = pd.DataFrame(X[:,:3])
X.columns = ['1','2','3']
X_test = pd.DataFrame(X_test[:,:3])
X_test.columns = ['1','2','3']

select_features = [
                    #"rate1_mean",
                    #"rate1_std",
                    #"rate2_mean",
                    #"rate2_std",
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
                    #'truck1_mean',
                    #'truck2_mean',
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
                    #'1','2','3',
                    #"im_diff_mean_mean","im_diff_mean_std","im_diff_std_mean","im_diff_std_std",
                  ]


train_features = pd.read_csv(path + 'train_feature_rate0.25_uw_tr0.2.csv')
test_features = pd.read_csv(path + 'test_feature_rate0.5_uw_tr0.2.csv')
train_df = pd.read_csv(path + 'train_feature_time1.csv')
test_df = pd.read_csv(path + 'test_feature_time1.csv')
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

train_features = pd.concat([train_features, train_df, X], axis = 1)
test_features = pd.concat([test_features, test_df, X_test], axis = 1)

train_features["label"] = train_features["label"].apply(int)
test_features["label"] = test_features["label"].apply(int)

train_x = train_features[select_features].copy()
train_y = train_features["label"]
test_x = test_features[select_features].copy()

""" ========================  Train the Model ============================= """
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

    for i, (train_index, verify_index) in enumerate(kf.split(train_x, train_y)):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        ve_x = train_x[verify_index]
        ve_y = train_y[verify_index]

        if 1:
            model = ExtraTreesClassifier(n_estimators=800, bootstrap=True, oob_score=True)
            model.fit(tr_x, tr_y)
            verify_res = model.predict_proba(ve_x)
            #pred = model.predict(test_x, num_iteration = model.best_iteration)
            #train[verify_index] = verify_res
            #test_pre[i, :] = pred

            pre_y = np.argmax(verify_res, axis = 1)
            f1_list = f1_score(ve_y, pre_y, average = None)
            f1 = 0.2*f1_list[0] + 0.2*f1_list[1] + 0.6*f1_list[2]
            
            f1_scores.append(f1)
            #test_pre_all[i, :] = np.argmax(pred, axis=1)

    f1_mean = np.mean(f1_scores)
    test[:] = test_pre.mean(axis = 0)
    y_test = np.argmax(test, axis = 1)

    return y_test, f1_mean

y_test, f1_mean = ET(train_x, train_y, test_x)
a = 1