from PIL import Image
import os
import pandas as pd
import numpy as np
import json

path = "I:/TianChi/data/"
train_json = pd.read_json(path + "train_target.json")
'''
for s in list(train_json.annotations[:]):
    map_id = s["id"]
    map_key = s["key_frame"]
    frames = s["frames"]
    status = s["status"]
    os.makedirs(path + 'amap_traffic_train_0712/00' + str(int(map_id) + 1500))
    save_path = path + 'amap_traffic_train_0712/00' + str(int(map_id) + 1500) + '/'
    for i in range(0,len(frames)):
        f = frames[i]
        #image = Image.open(path + 'amap_traffic_train_0712' + "/" + map_id + "/" + f["frame_name"])
        #out = image.transpose(Image.FLIP_LEFT_RIGHT)
        #out.save(save_path + f["frame_name"], quality = 95)
'''

with open(path+"train_target.json","r") as f:
    content = f.read()
content = json.loads(content)
for i in content["annotations"]:
    i['id'] = '00' + str(int(i['id']) + 1500)
with open(path+"train_targrt1.json","w") as f:
    f.write(json.dumps(content))