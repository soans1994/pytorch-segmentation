import numpy as np
import torch
import cv2
import glob
from matplotlib import pyplot as plt


class custom_dataset:
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):#class fucntions
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#h,w,c
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
        image = np.transpose(image, (2,0,1)).astype(np.float)#pytorch model input(b,c,h,w)
        image = image/255
        mask = cv2.imread(self.mask_paths[idx],0)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0)
        #mask = np.transpose(mask, (2, 0, 1)).astype(np.float)
        mask = mask/255
        #return {"image":torch.tensor(image), "mask":torch.tensor(mask)}
        return torch.tensor(image, dtype=torch.float), torch.tensor(mask, dtype=torch.float) #torch.tensor(mask, dtype=torch.uint8)(mask, dtype=torch.long)
        #dtype=torch.tensor()[0-1] dtype=torch.float32 original
#"""
#image_path = glob.glob("resized/training/image_2/*")
#mask_path = glob.glob("resized/training/gt_image_2/*")
image_path = glob.glob("camvid/train/*")
mask_path = glob.glob("camvid/train_GT/*")
dataset = custom_dataset(image_path, mask_path)
print(len(image_path), len(mask_path))
print(len(dataset))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
print(len(train_loader))


for image, mask in train_loader:
    print(image.shape, image.dtype)
    print(mask.shape, mask.dtype)
    mask = np.transpose(mask, (0, 2, 3, 1))
    plt.imshow(mask[15])
    plt.show()
    break
print("sdf")
print("sdf")
#"""