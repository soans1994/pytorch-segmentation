import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from unet import unet
#from fcnown import fcn
from fcnown_resnet50 import fcn
from fcn import FCNs, VGGNet
from torchvision.models.vgg import VGG
import glob
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from load_data2 import custom_dataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
from callbacks import EarlyStopping


lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
epochs = 500
num_workers = 2
img_height = 256
img_width = 256
pin_memory = True
load_model = False
#img_path = glob.glob("data_road/training/image_2/*")
#mask_path = glob.glob("data_road/training/gt_image_2/*")
img_path = glob.glob("camvid/train/*")
mask_path = glob.glob("camvid/train_GT/*")
dataset = custom_dataset(img_path, mask_path)
print(len(img_path), len(mask_path))
print(len(dataset))
#train_set, test_set = random_split(dataset, [200,89], generator=torch.Generator().manual_seed(42) )
train_set, test_set = random_split(dataset, [300,120], generator=torch.Generator().manual_seed(42) )
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
print(len(train_loader), len(test_loader))
"""
for image, mask in train_loader:
    print(image.shape, image.dtype)
    print(mask.shape, mask.dtype)
    #break
"""
#model = unet()
model = fcn()
#vgg_model = VGGNet(requires_grad=True)
# test a random batch, loss should decrease
#model = FCNs(pretrained_net=vgg_model, n_class=32)
model.to(device)
loss_fn = nn.CrossEntropyLoss()#does auto onehot encoding
#loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def save_checkpoint(state, filename="my_checkpoint.pth"):
    print("saving checkpoint")
    torch.save(state, filename)

n_epochs_stop = 10
epochs_no_improve = 0
early_stop = False
min_val_loss = np.Inf
earlystopping = EarlyStopping(patience=10, verbose=True)

for epoch in range(epochs):
    losses = []
    for batch_idx, (data, labels) in enumerate(train_loader):
        labels = torch.argmax(labels, dim=1)#ccel torch.Size([16, 256, 256]) torch.int64.... 4,4,4......21,21,21...
        data = data.to(device=device)
        labels = labels.to(device=device)
        #forward
        predictions = model(data)
        #print(predictions.shape,predictions.dtype, labels.shape, labels.dtype)
        loss = loss_fn(predictions, labels)
        losses.append(loss.item())
        #backward
        optimizer.zero_grad()
        loss.backward()
        #grad desc/ adam step
        optimizer.step()
    print(f"Loss at epoch {epoch} is {sum(losses)/len(losses)}")

#"""
    earlystopping(sum(losses), model)  # callメソッド呼び出し
    if earlystopping.early_stop:  # ストップフラグがTrueの場合、breakでforループを抜ける
        print("Early Stopping!")
        break
#"""
"""
checkpoint = {"state_dict": model.state_dict(),
              "optimizer": optimizer.state_dict()
}
save_checkpoint(checkpoint)
"""
def check_accuracy(dataloader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            #_, pred = scores.max(1)
            pred = scores
            #print(pred.shape, pred.dtype, y.shape, y.dtype)
            num_correct += (pred==y).sum()
            num_samples += pred.size(0)
        print(f"got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}")
    model.train()
print("check acc on train set")
check_accuracy(train_loader, model)
print("check acc on test set")
check_accuracy(test_loader, model)

print("test")




