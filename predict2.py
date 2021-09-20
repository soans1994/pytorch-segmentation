import torch
torch.cuda.empty_cache()
from unet import unet
from fcn import FCNs, VGGNet
from fcnown import fcn
import glob
from load_data2 import custom_dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import numpy as np
from matplotlib import pyplot as plt
def load_checkpoint(checkpoint, model):
    print("loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16

#model = unet()
model = fcn()
#vgg_model = VGGNet(requires_grad=True)
# test a random batch, loss should decrease
#model = FCNs(pretrained_net=vgg_model, n_class=32)
model.to(device)
#load_checkpoint(torch.load("my_checkpoint.pth"), model)
#model.load_state_dict(torch.load("checkpoint_model_unet2_cel.pth"))
#model.load_state_dict(torch.load("checkpoint_model_unet2_mse.pth"))
#model.load_state_dict(torch.load("checkpoint_model_fcn_cel.pth"))
#model.load_state_dict(torch.load("checkpoint_model_fcn_mse.pth"))
#model.load_state_dict(torch.load("checkpoint_model_unet_mse.pth"))
#model.load_state_dict(torch.load("checkpoint_model_unet_cel.pth"))
#model.load_state_dict(torch.load("checkpoint_model_fcnown_mse.pth"))
#model.load_state_dict(torch.load("checkpoint_model_fcnownresnet_mse.pth"))

#img_path = glob.glob("data_road/training/image_2/*")
#mask_path = glob.glob("data_road/training/gt_image_2/*")
img_path = glob.glob("camvid/train/*")
mask_path = glob.glob("camvid/train_GT/*")
dataset = custom_dataset(img_path, mask_path)
print(len(img_path), len(mask_path))
print(len(dataset))
#train_set, test_set = random_split(dataset, [200,89], generator=torch.Generator().manual_seed(42) )
train_set, test_set = random_split(dataset, [300,120], generator=torch.Generator().manual_seed(42) )
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
print(len(train_loader), len(test_loader))

def check_accuracy(dataloader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            y = torch.argmax(y, dim=1)  # ccel torch.Size([16, 256, 256]) torch.int64
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            pred = scores
            pred = torch.argmax(scores, dim=1)  # ccel torch.Size([16, 256, 256]) torch.int64
            print(pred.shape, pred.dtype, y.shape, y.dtype)
            num_correct += (pred==y).sum()
            num_samples += pred.size(0)
        print(f"got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}")
    model.train()
print("check acc on train set")
check_accuracy(train_loader, model)
print("check acc on test set")
check_accuracy(test_loader, model)

for x, y in train_loader:
    x = x.to(device)
    y = y.to(device)
    pred = model(x)
    print(pred.shape)
    target = torch.argmax(pred, dim=1)#cel
    y = torch.argmax(y, dim=1)#cel
    #print(target.shape)
    y = y.detach().cpu().numpy()#cel
    target = target.detach().cpu().numpy()#cel
    plt.imshow(np.squeeze(target[10]))
    plt.show()
    plt.imshow(np.squeeze(y[10]))
    plt.show()
    plt.imshow(target[10])
    plt.show()
    plt.imshow(y[10])
    plt.show()
    break
print("test")