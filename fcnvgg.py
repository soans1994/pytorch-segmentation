import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_model_summary import summary

vgg16 = models.vgg16(pretrained=True)
for param in vgg16.features.parameters():
  param.requires_grad = False
#False Total params: 185,771,904 Trainable params: 171,057,216 Non-trainable params: 14,714,688
#true  Total params: 185,771,904 Trainable params: 185,771,904 Non-trainable params: 0

class fcn(nn.Module):
  def __init__(self):
    super(fcn, self).__init__()
    self.features = vgg16.features
    self.classifier = nn.Sequential(
      nn.Conv2d(512, 4096, 7),
      nn.ReLU(inplace=True),
      #nn.Dropout2d(),
      nn.Conv2d(4096, 4096, 1),
      nn.ReLU(inplace=True),
      #nn.Dropout2d(),
      nn.Conv2d(4096, 32, 1),
      nn.ConvTranspose2d(32, 32, 224, stride=32)
    )

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    #print(x.shape)
    return x

class fcn16(nn.Module):
  def __init__(self):
    super(fcn16, self).__init__()
    self.features = vgg16.features
    self.classifier = nn.Sequential(
      nn.Conv2d(512, 4096, 7),
      nn.ReLU(inplace=True),
      nn.Conv2d(4096, 4096, 1),
      nn.ReLU(inplace=True),
      nn.Conv2d(4096, 21, 1)
    )
    self.score_pool4 = nn.Conv2d(512, 32, 1)
    self.upscore2 = nn.ConvTranspose2d(32, 32, 14, stride=2, bias=False)
    self.upscore16 = nn.ConvTranspose2d(32, 32, 16, stride=16, bias=False)

  def forward(self, x):
    pool4 = self.features[:-7](x)
    pool5 = self.features[-7:](pool4)
    pool5_upscored = self.upscore2(self.classifier(pool5))
    pool4_scored = self.score_pool4(pool4)
    combined = pool4_scored + pool5_upscored
    res = self.upscore16(combined)
    return res

if __name__ == "__main__":
    input = torch.randn(8, 3, 512, 512)
    model = fcn()
    # model.load=stat=dicr(modelpsth)
    print(model)
    print(summary(model, input))
    model(input)
