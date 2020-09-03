import numpy as np
import pandas as pd

path = "G:/04_Study/TianChi/data/"
#train1 = np.load(path + '训练集.npy')
#train2 = np.load(path + '训练集-数据增强.npy')
test = np.load(path + 'test_logit.npy')
#train = np.vstack((train1, train2))

#train = pd.DataFrame(train)
test = pd.DataFrame(test)

#train.columns = ['1', '2', '3']
test.columns = ['1', '2', '3']

#train.to_csv(path_or_buf = path + 'train_feature3000_fromJ_new.csv')
test.to_csv(path_or_buf = path + 'test_new_feature3000_fromJ_new.csv')
a = 1