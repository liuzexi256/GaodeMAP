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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA, SparsePCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
import json
from yolo import YOLO
from PIL import Image
import os

""" ======================  Function definitions ========================== """
def get_data(df,img_path):
    map_id_list = []
    label = []
    key_frame_list = []
    jpg_name_1 = []
    jpg_name_2 = []
    gap_time_1 = []
    gap_time_2 = []
    im_diff_mean = []
    im_diff_std = []

    for s in list(df.annotations):
        map_id = s["id"]
        map_key = s["key_frame"]
        frames = s["frames"]
        status = s["status"]
        for i in range(0,len(frames) - 1):
            f = frames[i]
            f_next = frames[i + 1]
            if f_next['gps_time'] - f['gps_time'] > 80000:
                continue
            
            im = mpimg.imread(path+img_path+"/"+map_id+"/"+f["frame_name"])
            im_next = mpimg.imread(path+img_path+"/"+map_id+"/"+f_next["frame_name"])
            
            if im.shape == im_next.shape:
                im_diff = im - im_next
            else:
                im_diff = im
            
            im_diff_mean.append(np.mean(im_diff))
            im_diff_std.append(np.std(im_diff))
            

            map_id_list.append(map_id)
            key_frame_list.append(map_key)

            jpg_name_1.append(f["frame_name"])
            jpg_name_2.append(f_next["frame_name"])
            gap_time_1.append(f["gps_time"])
            gap_time_2.append(f_next["gps_time"])
            label.append(status)
    train_df = pd.DataFrame({
        "map_id":map_id_list,
        "label":label,
        "key_frame":key_frame_list,
        "jpg_name_1":jpg_name_1,
        "jpg_name_2":jpg_name_2,
        "gap_time_1":gap_time_1,
        "gap_time_2":gap_time_2,
        "im_diff_mean":im_diff_mean,
        "im_diff_std":im_diff_std,
    })
    tz = pytz.timezone('Asia/Shanghai')
    train_df["gap"]=train_df["gap_time_2"]-train_df["gap_time_1"]
    train_df["gap_time_today"]=train_df["gap_time_1"]%(24*3600)
    train_df["hour"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x, tz).hour)
    train_df["minute"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x, tz).minute)
    train_df["day"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x, tz).day)
    train_df["dayofweek"]=train_df["gap_time_1"].apply(lambda x:datetime.fromtimestamp(x, tz).weekday())
    
    train_df["key_frame"]=train_df["key_frame"].apply(lambda x:int(x.split(".")[0]))
    
    train_df = train_df.groupby("map_id").agg({"gap":["mean","std"],
                                             "hour":["mean"],
                                             "minute":["mean"],
                                             "dayofweek":["mean"],
                                             "gap_time_today":["mean","std"],
                                             "im_diff_mean":["mean","std"],
                                             "im_diff_std":["mean","std"],
                                             "label":["mean"],
                                            }).reset_index()
    train_df.columns=["map_id","gap_mean","gap_std",
                      "hour_mean",
                      "minute_mean",
                      "dayofweek_mean","gap_time_today_mean","gap_time_today_std",
                      "im_diff_mean_mean","im_diff_mean_std","im_diff_std_mean","im_diff_std_std",
                      "label"
                      ]
    train_df["label"]=train_df["label"].apply(int)
    
    return train_df

def extact_fratures(df, img_path, df_features, key_weight = 3):
    map_id_list = []
    label = []
    key_frame_list = []
    jpg_name_1 = []
    jpg_name_2 = []
    gap_time_1 = []
    gap_time_2 = []
    im_diff_mean = []
    im_diff_std = []

    yolo = YOLO()

    for s in list(df.annotations[:]):
        map_id = s["id"]
        map_key = s["key_frame"]
        frames = s["frames"]
        status = s["status"]

        for i in range(0,len(frames)):
            f = frames[i]
            
            image = Image.open(img_path + "/" + map_id + "/" + f["frame_name"])
            
            if f["frame_name"] == map_key:
                r_image, pic_res = yolo.detect_image(image)
                pic_res = pic_res*key_weight
                
            else:
                r_image, pic_res = yolo.detect_image(image)
            
            #r_image.show()
            #r_image.save(path+'amap_traffic_train_0712_result'+"/"+map_id+"/"+f["frame_name"])
            

            #r_image, pic_res = yolo.detect_image(image)
            pic_res = list(pic_res)
            pic_res.append(map_id)
            pic_res.append(map_key)
            pic_res.append(f["frame_name"])
            pic_res.append(status)
            df_features.loc[df_features.shape[0] + 1] = pic_res

    return df_features

""" =======================  Load Training Data ======================= """
path = "G:/04_Study/TianChi/data/"   #存放数据的地址
train_json = pd.read_json(path + "train_target.json")
test_json = pd.read_json(path + "test_new_target.json")

train_features = pd.read_csv(path + 'train_feature3000_rate0.5_w3_tr0.5_new.csv')
test_features = pd.read_csv(path + 'test_new_feature3000_rate0.5_w3_tr0.5_new.csv')
train_df = pd.read_csv(path + 'train_feature3000_timecn_new.csv')
test_df = pd.read_csv(path + 'test_new_feature3000_timecn_new.csv')
X = pd.read_csv(path + 'train_feature3000_fromJ.csv')
X_test = pd.read_csv(path + 'test_feature3000_fromJ.csv')
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
                    'bus1_mean',
                    'bus2_mean',
                    'motorbike1_mean',
                    'motorbike2_mean',
                    'bicycle1_mean',
                    'bicycle2_mean',
                    "gap_mean",
                    "gap_std",
                    "hour_mean",
                    "minute_mean",
                    "dayofweek_mean",
                    "gap_time_today_mean",
                    "gap_time_today_std",
                    '1','2','3',
                    "im_diff_mean_mean","im_diff_mean_std","im_diff_std_mean","im_diff_std_std",
                    'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v'
                  ]


""" ========================  Extract Features ============================= """

extract_features = ["rate1","rate2",
                    "car1","car2","truck1","truck2","bus1","bus2","motorbike1","motorbike2","bicycle1","bicycle2",
                    ]

#df_features = pd.DataFrame(columns = extract_features + ["map_id", "key_frame", "picture_name", "label"])
#train_features = extact_fratures(train_json[:], path + "amap_traffic_train_0712", df_features)
#train_features.to_csv(path_or_buf = path + 'train_feature3000_rate0.25_w3_tr0.5_new.csv')

df_features = pd.DataFrame(columns = extract_features + ["map_id", "key_frame", "picture_name", "label"])
test_features = extact_fratures(test_json[:], path + "amap_traffic_b_test_0828", df_features)
test_features.to_csv(path_or_buf = path + 'test_new_feature3000_rate0.5_w3_tr0.5_new.csv')

""" ========================  Extract Sub-Features ============================= """
#train_features['rate1'] = train_features['rate1']*1000
#train_features['rate2'] = train_features['rate2']*1000

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
#train_df = get_data(train_json[:],"amap_traffic_train_0712")
test_df = get_data(test_json[:],"amap_traffic_b_test_0828")
test_df.to_csv(path_or_buf = path + 'test_new_feature3000_timecn_new.csv')
train_df.to_csv(path_or_buf = path + 'train_feature3000_timecn_new.csv')
'''

train_features = pd.concat([train_features, train_df, X, train_features_fromM], axis = 1)
test_features = pd.concat([test_features, test_df, X_test, test_features_fromM], axis = 1)

train_features["label"] = train_features["label"].apply(int)
test_features["label"] = test_features["label"].apply(int)


train_x = train_features[select_features].copy()
train_y = train_features["label"]
test_x = test_features[select_features].copy()

""" ========================  Select Features ============================= """
train_x = MinMaxScaler().fit_transform(train_x)
test_x = MinMaxScaler().fit_transform(test_x)
#方差选择

plt.figure()
plt.subplot(2,1,1)
selector1 = VarianceThreshold()
selector1.fit_transform(train_x)
var1 = selector1.variances_
plt.bar(select_features, var1)
plt.subplot(2,1,2)
selector2 = VarianceThreshold()
selector2.fit_transform(test_x)
var2 = selector2.variances_
plt.bar(select_features, var2)
plt.show()


#相关系数法

kbest = SelectKBest(chi2, k = 10)
kbest.fit_transform(abs(train_x), train_y)
a = kbest.scores_
plt.bar(select_features, a)
plt.show()


#Select From Model
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(train_x, train_y)
FI = clf.feature_importances_
model = SelectFromModel(clf, prefit = True)
plt.bar(select_features, FI)
plt.show()
X_new = model.transform(train_x)

a = 1