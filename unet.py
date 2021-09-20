import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        #nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        #nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )
    return conv

class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.down_conv1 = double_conv(3, 64)
        self.down_conv2 = double_conv(64, 128)
        self.down_conv3 = double_conv(128, 256)
        self.down_conv4 = double_conv(256, 512)
        self.down_conv5 = double_conv(512, 1024)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_trans1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv1 = double_conv(1024, 512)
        self.up_trans2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv2 = double_conv(512, 256)
        self.up_trans3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv3 = double_conv(256, 128)
        self.up_trans4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv4 = double_conv(128, 64)
        self.out = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)

    def forward(self, image):
        #encoder(b,c,h,w)
        x1 = self.down_conv1(image)#64 skip
        #print(x1.shape)
        x2 = self.max_pool(x1)
        #print(x2.shape)
        x3 = self.down_conv2(x2)#128 skip
        #print(x3.shape)
        x4 = self.max_pool(x3)
        #print(x4.shape)
        x5 = self.down_conv3(x4)#256 skip
        #print(x5.shape)
        x6 = self. max_pool(x5)
        #print(x6.shape)
        x7 = self.down_conv4(x6)#512 skip
        #print(x7.shape)
        x8 = self.max_pool(x7)
        #print(x8.shape)
        x9 = self.down_conv5(x8)#1024
        #print(x9.shape)

        #decoder
        x = self.up_trans1(x9)
        #print(x.shape)
        x = self.up_conv1(torch.cat([x, x7], dim=1))
        #print(x.shape)
        x = self.up_trans2(x)
        #print(x.shape)
        x = self.up_conv2(torch.cat([x, x5], 1))
        #print(x.shape)
        x = self.up_trans3(x)
        #print(x.shape)
        x = self.up_conv3(torch.cat([x, x3], 1))
        #print(x.shape)
        x = self.up_trans4(x)
        #print(x.shape)
        x = self.up_conv4(torch.cat([x, x1], 1))
        #print(x.shape)
        x = self.out(x)
        #print(x.shape)
        return x


if __name__ == "__main__":
    image = torch.rand((16,3,256,256))#.to("cuda")
    model = unet()
    print(model)
    #model.to("cuda")
    #model.forward(image)
    #print(model(image))
    pred = model(image)
    print(pred.shape)
    print(image.device)

