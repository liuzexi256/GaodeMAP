import pandas as pd
import numpy as np
path = "I:/TianChi/data/"
res = []

with open(path + '全部修改完毕.txt', 'r') as f:
    data = f.readlines()  #txt中所有字符串读入data
 
    for line in data:
        if line[0] == '/':
            if line[8] == '1':
                res.append(int(line.split()[1]))
        else:
            if line[7] == '1':
                res.append(int(line.split()[1]))
res1 = np.array(res + res)
np.save(path + 'label_new.npy', res1)

a = 1 