import os
import cv2
import json
import numpy as np


def sequence_imgs_mean(ann_file, src_path, dst_path):
    anns = json.load(open(ann_file, "r"))
    anns = anns["annotations"]
    
    for ann in anns:
        # print(ann)
        cls = ann["status"]
        id = ann["id"]
        print(id)
        
        imgs = os.listdir(os.path.join(src_path, str(id)))
        imgs_num = 1
        total_img = cv2.imread(os.path.join(src_path, str(id),ann["key_frame"]))
        # total_img = np.empty(img_shape, dtype = float, order = 'C')
        for img_name in imgs:
            if img_name == ann["key_frame"]:
                continue
            img = cv2.imread(os.path.join(src_path, str(id), img_name))
            # print(img)
            # print(img.shape)
            if img.shape != total_img.shape:
                continue
            imgs_num += 1
            img = img.astype(np.float32)
            total_img = total_img.astype(np.float32)
            total_img += img
        #mean_img = total_img/ imgs_num
        mean_img = total_img
        mean_img = mean_img.astype(np.uint8)
        if not os.path.exists(os.path.join(dst_path, str(cls))):
            os.makedirs(os.path.join(dst_path, str(cls)))
        a = os.path.join(dst_path, str(cls), id+".jpg")
        cv2.imwrite(os.path.join(dst_path, str(cls), id+".jpg"), mean_img)
path = "I:/TianChi/data/"
sequence_imgs_mean(path +"train_target.json", path + "amap_traffic_train_0712", "mean_train_torch")