import numpy as np
import torch
import cv2
import glob
from matplotlib import pyplot as plt
color_map = {
        0: (64, 128, 64),	#Animal
        1: (192, 0, 128),	#Archway
        2: (0, 128, 192),	#Bicyclist
        3: (0, 128, 64),	#Bridge
        4: (128, 0, 0),		#Building
        5: (64, 0, 128),	#Car
        6: (64, 0, 192),	#CartLuggagePram
        7: (192, 128, 64),	#Child
        8: (192, 192, 128),	#Column_Pole
        9: (64, 64, 128),	#Fence
        10: (128, 0, 192),	#LaneMkgsDriv
        11: (192, 0, 64),	#LaneMkgsNonDriv
        12: (128, 128, 64),	#Misc_Text
        13: (192, 0, 192),	#MotorcycleScooter
        14: (128, 64, 64),	#OtherMoving
        15: (64, 192, 128),	#ParkingBlock
        16: (64, 64, 0),	#Pedestrian
        17: (128, 64, 128),	#Road
        18: (128, 128, 192),	#RoadShoulder
        19: (0, 0, 192),		#Sidewalk
        20: (192, 128, 128),	#SignSymbol
        21: (128, 128, 128),	#Sky
        22: (64, 128, 192),	#SUVPickupTruck
        23: (0, 0, 64),		#TrafficCone
        24: (0, 64, 64),		#TrafficLight
        25: (192, 64, 128),	#Train
        26: (128, 128, 0),	#Tree
        27: (192, 128, 192),	#Truck_Bus
        28: (64, 0, 64),		#Tunnel
        29: (192, 192, 0),	#VegetationMisc
        30: (0, 0, 0),		#Void
        31: (64, 192, 0)	#Wall
}
def rgb_to_mask(img, color_map):
#    Converts a RGB image mask of shape [batch_size, h, w, 3] to Binary Mask of shape [batch_size, classes, h, w]
#Parameters:img: A RGB img mask
#color_map: Dictionary representing color mappings
#returns:out: A Binary Mask of shape [batch_size, classes, h, w]
     num_classes = len(color_map)
     shape = img.shape[:2]+(num_classes,)
     out = np.zeros(shape, dtype=np.int8)
     for i, cls in enumerate(color_map):
         out[:,:,i] = np.all(img.reshape( (-1,3) ) == color_map[i], axis=1).reshape(shape[:2])
     return out

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
        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = rgb_to_mask(mask, color_map)
        mask = np.transpose(mask, (2,0,1))

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
    target = torch.argmax(mask, dim=1)
    print(target.shape, target.dtype)
    #mask = np.transpose(mask, (0, 2, 3, 1))
    #plt.imshow(mask[4])
    #plt.show()
    plt.imshow(target[4])
    plt.show()
    #target = np.transpose(target, (0, 2, 3, 1)))
    #target = target.argmax(axis=3)
    #print(target.shape, target.dtype)
    #target = np.expand_dims(target, axis=-1)
    #print(target.shape, target.dtype)
    #plt.imshow(target[4])
    #qplt.show()
    break
print("sdf")
print("sdf")
#"""