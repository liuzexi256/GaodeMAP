import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score, log_loss
from bayes_opt import BayesianOptimization
""" ======================  Function definitions ========================== """
def BayesianSearch(clf, params):
    """贝叶斯优化器"""
    # 迭代次数
    num_iter = 25
    init_points = 5
    # 创建一个贝叶斯优化对象，输入为自定义的模型评估函数与超参数的范围
    bayes = BayesianOptimization(clf, params)
    # 开始优化
    bayes.maximize(init_points=init_points, n_iter=num_iter)
    # 输出结果
    params = bayes.res['max']
    print(params['max_params'])
    
    return params

def GBM_evaluate(min_child_samples, min_child_weight, colsample_bytree, max_depth, subsample, reg_alpha, reg_lambda):
    """自定义的模型评估函数"""

    # 模型固定的超参数
    param = {
        'objective': 'regression',
        'n_estimators': 275,
        'metric': 'rmse',
        'random_state': 2020}

    # 贝叶斯优化器生成的超参数
    param['min_child_weight'] = int(min_child_weight)
    param['colsample_bytree'] = float(colsample_bytree),
    param['max_depth'] = int(max_depth),
    param['subsample'] = float(subsample),
    param['reg_lambda'] = float(reg_lambda),
    param['reg_alpha'] = float(reg_alpha),
    param['min_child_samples'] = int(min_child_samples)

    # 5-flod 交叉检验，注意BayesianOptimization会向最大评估值的方向优化，因此对于回归任务需要取负数。
    # 这里的评估函数为neg_mean_squared_error，即负的MSE。
    val = cross_val_score(lgb.LGBMRegressor(**param),
        train_X, train_y ,scoring='neg_mean_squared_error', cv=5).mean()

    return val

def LGB(params, train_x, train_y):
    predictors = list(train_x.columns)
    train_x = train_x.values
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

        train_matrix = lightgbm.Dataset(tr_x, label = tr_y)
        verify_matrix = lightgbm.Dataset(ve_x, label = ve_y)

        num_round = 6000
        early_stopping_rounds = 1000
        if verify_matrix:
            model = lightgbm.train(params, train_matrix, num_round, 
                              valid_sets = verify_matrix, 
                              verbose_eval = 50,
                              early_stopping_rounds = early_stopping_rounds
                              )
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

    f1_mean = np.mean(f1_scores)

    return f1_mean

def lgb_cv(feature_fraction,bagging_fraction,bagging_freq,learning_rate,num_leaves,min_child_weight,
            min_data_in_leaf,max_depth,min_split_gain,lambda_l2,num_iterations=5000):
        params = {
                    'boosting_type': 'gbdt',
                    'objective':'multiclass',
                    'metric':'multi_logloss',
                    'nthread': 6,'num_class':3,'verbose': -1,}

        params['min_child_weight'] = max(min_child_weight, 0)
        params["num_leaves"] = int(round(num_leaves))
        params['lambda_l2'] = max(lambda_l2, 0)
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['bagging_freq'] = int(round(bagging_freq))
        params['learning_rate'] = max(min(learning_rate, 1), 0)
        params["min_data_in_leaf"] = int(round(min_data_in_leaf))
        params['max_depth'] = int(round(max_depth))
        params['min_split_gain'] = min_split_gain
        
        f1_result = LGB(params, train_x, train_y)
        return f1_result

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
                    #"im_diff_mean_mean","im_diff_mean_std","im_diff_std_mean","im_diff_std_std",
                  ]

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

""" ======================  Random Search ========================== """
bounds = {
        'min_child_weight': (1,10),
        'num_leaves': (8, 150),
        'lambda_l2': (0, 50),
        #'lambda_l1': (0, 50),
        'feature_fraction': (0.2, 1),
        'bagging_fraction': (0.2, 1),
        'bagging_freq': (1, 100),
        'learning_rate': (0.01, 1),
        'min_data_in_leaf': (1,20),
        'max_depth': (3, 30),
        'min_split_gain': (0, 50),
        
        }
lgb_bo = BayesianOptimization(lgb_cv, bounds, random_state = 1111)

lgb_bo.maximize(init_points = 10, n_iter = 100)
best = lgb_bo.max
a = 0