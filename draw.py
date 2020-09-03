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
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
import json
from yolo1 import YOLO
from PIL import Image
import os

path = "G:/04_Study/TianChi/data/"   #存放数据的地址

train_json = pd.read_json(path + "train_target.json")
test_json = pd.read_json(path + "test_new_target.json")

extract_features = ["rate1","rate2",
                    "car1","car2","truck1","truck2","bus1","bus2","motorbike1","motorbike2","bicycle1","bicycle2",
                    ]
yolo = YOLO()
for s in list(test_json.annotations[:]):
    map_id = s["id"]
    map_key = s["key_frame"]
    frames = s["frames"]
    status = s["status"]
    os.makedirs(path + 'amap_traffic_test_0828_result0.5/' + map_id)
    save_path = path + 'amap_traffic_test_0828_result0.5/' + map_id + '/'
    for i in range(0,len(frames)):
        f = frames[i]
        image = Image.open(path + 'amap_traffic_b_test_0828' + "/" + map_id + "/" + f["frame_name"])
        r_image, pic_res = yolo.detect_image(image)
        r_image.save(save_path + f["frame_name"], quality = 95)