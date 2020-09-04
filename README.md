# TianChi_GaodeMAP
## 使用YOLOv4提取特征_代码细节  
extract_features调用yolo.py中的YOLO类  
### 1. 设置识别阈值  
```
"confidence": 0.5
```
### 2. generate函数加载预训练权重  

### 3. detect_image函数检测图片输出检测结果  
```
predict = np.zeros(12) #初始化检测结果  
```
```
predict = self.calsquare(image, boxes, top_label, predict) #调用calsuqare函数
```
### 4. calsquare函数计算各种特征值  
```
    def calsquare(self, image, boxes, top_label, res):
        dic = {2:2, 7:4, 5:6, 3:8, 1:10} #选择检测目标中的有用目标，检测目标文件在'model_data/coco_classes.txt'
        w, h = image.size #计算图片长宽
        top_main = h/4 #设置中心框的上边缘
        bottom_main = 3*h/4 #设置中心框的下边缘
        left_main = w/4 #设置中心框的左边缘
        right_main = 3*w/4 #设置中心框的右边缘
        box_main = [top_main, left_main, bottom_main, right_main]
        square_rate = 0
        square1 = 0
        square2 = 0
        pre = []
        for i, c in enumerate(top_label):
            if c not in dic: #判断检测目标是否是有用目标
                continue
            predicted_class = self.class_names[c]
            top, left, bottom, right = boxes[i]
            
            square = (bottom - top)*(right - left) #计算目标框面积
            if square >= 0.4*w*h: #过滤异常检测
                continue
            if bottom > h - 30 and left < 30 and right > w - 30: #过滤异常检测
                continue
            flag = 0
            for j in range(len(pre)): #过滤重复框
                temp = abs(pre[j] - boxes[i])
                if sum(temp) < 20:
                    flag = 1
                    break
            if flag:
                continue
            pre.append(list(boxes[i]))

            if self.mat_inter(box_main, boxes[i]): #判断目标是否与中心框重合
                top, left, bottom, right = boxes[i]
                w_con = min(right,right_main) - max(left,left_main) #计算重合宽
                h_con = min(bottom,bottom_main) - max(top,top_main) #计算重合高
                square_con = w_con * h_con #计算重合面积

                square1 = square1 + square_con #目标中心面积
                square2 = square2 + (square - square_con) #目标边缘面积

                if square_con >= 0.5*square: #判断目标整体在中心还是在边缘
                    idx = dic[c]
                    res[idx] = res[idx] + 1
                else:
                    idx = dic[c] + 1
                    res[idx] = res[idx] + 1
            else:
                square2 = square2 + square
                idx = dic[c] + 1
                res[idx] = res[idx] + 1

        res[0] = square1/(h*w/4) #计算中心比率
        res[1] = square2/(3*h*w/4) #计算边缘比率

        return res
```
  + input:  
    - image: 图片矩阵
    - boxes: yolo识别出的目标框
    - top_label: boexes中每个框的分类
    - res: 储存各种特征值
  + output:
    - res: 特征值

### 5. mat_inter函数判断两个矩形是否相交  
  
  
## 使用YOLOv4提取特征_使用说明  
### 所需环境  
torch == 1.2.0  
### 文件下载  
下载链接:https://drive.google.com/file/d/1pzLZMHPMOGQtjCc1KSVg6YT70bXIcbR1/view?usp=sharing  
权重文件下载好后放在'model_data'文件夹下  

### 提取特征步骤  
  1. 打开extract_features.py

  2. 加载图片  
  ```
path = "I:/TianChi/data/"   #存放数据的地址  
train_json = pd.read_json(path + "train_target.json")  
test_json = pd.read_json(path + "test_target.json")  
```
  3. 选择需要提取的特征  
  ```
extract_features = ["rate1","rate2",
                    "car1","car2","truck1","truck2","bus1","bus2","motorbike1","motorbike2","bicycle1","bicycle2",
                    ]
```
                    
  4. 将要提取的特征和图片属性特征合并  
  ```
df_features = pd.DataFrame(columns = extract_features + ["map_id", "key_frame", "picture_name", "label"])  
```  
  5. 执行extact_fratures(df, img_path, df_features, key_weight = 3)  
  + input:  
      + df: 从json文件读取的dataframe
      + img_path: 图片集文件夹路径
      + df_features: 需要提取的特征再加上图片属性特征
      + key_weight: 关键帧权重，默认为3  
  + output:  
      + 提取后的特征  
      
  6. 保存特征  
  ```
train_features.to_csv(path_or_buf = path + 'train_feature3000_rate0.25_w3_tr0.5_new.csv')
test_features.to_csv(path_or_buf = path + 'test_feature3000_rate0.25_w3_tr0.5_new.csv')
```
