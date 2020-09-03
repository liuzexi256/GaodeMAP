import pandas as pd

path = "I:/TianChi/data/"
res = []

with open(path + 'test_feature1500_fromM.txt', 'r') as f:
    data = f.readlines()  #txt中所有字符串读入data
 
    for line in data:
        res.append(line.split())        #将单个数据分隔开存好
        #numbers_float = map(float, odom) #转化为浮点数

for i in range(len(res)):
    temp = 0
    for j in range(1,21):
        temp += float(res[i][j])
    res[i].pop(0)
    res[i].insert(0,temp)

name = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v']

#res1 = res + res
test = pd.DataFrame(columns=name,data=res)#数据有三列，列名分别为one,two,three
#test['a'] = test['b'] + test['c'] + test['d'] + test['e'] + test['f'] + test['g'] + test['h'] + test['i'] + test['j'] + test['k'] + test['l'] + test['m'] + test['n'] + test['o'] + test['p'] + test['q'] + test['r'] + test['s'] + test['t'] + test['u'] + test['v']
#test['a'] = test.iloc[0,1:].sum()

##test = pd.read_csv(path + 'train_feature1500_fromM.csv')

test.to_csv(path + 'test_feature1500_fromM.csv')
a = 1