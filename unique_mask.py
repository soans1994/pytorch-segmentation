import cv2
import numpy as np
import glob
mask_path = glob.glob("camvid/train_GT/*")
masks=[]
for i in mask_path:
    mask = cv2.imread(i)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    mask = np.transpose(mask, (2, 0, 1)).astype(np.long)#float32 14 unique same
    masks.append(mask)
masks2 = np.array(masks)
print(masks2.shape)
print(np.unique(masks2))#14 #46