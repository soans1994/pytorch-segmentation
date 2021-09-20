import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}
class fcn(nn.Module):
    def __init__(self):
        super(fcn, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        #self.bn1_1 = nn.BatchNorm2d()
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        #self.bn1_2 = nn.BatchNorm2d()
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)#/2

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        # self.bn2_1 = nn.BatchNorm2d()
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        # self.bn2_2 = nn.BatchNorm2d()
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)#/4

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        # self.bn3_1 = nn.BatchNorm2d()
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        # self.bn3_2 = nn.BatchNorm2d()
        self.relu3_2 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)#/8

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        # self.bn4_1 = nn.BatchNorm2d()
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        # self.bn4_2 = nn.BatchNorm2d()
        self.relu4_2 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)  # /16

        self.conv5_1 = nn.Conv2d(512, 1024, 3, 1, 1)
        # self.bn5_1 = nn.BatchNorm2d()
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, 1, 1)
        # self.bn5_2 = nn.BatchNorm2d()
        self.relu5_2 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2)  # /32

        self.fc1 = nn.Conv2d(1024, 4096, 3, 1, 1)
        self.relu6 = nn.ReLU()
        self.drop6 = nn.Dropout2d()

        self.fc2 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU()
        self.drop7 = nn.Dropout2d()

        self.fc3 = nn.Conv2d(4096, 32, 1)
        self.up = nn.ConvTranspose2d(32, 32, kernel_size=32, stride=32)

    def forward(self, input):
        #print(input.shape)
        x = self.conv1_1(input)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        #print(x.shape)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        #print(x.shape)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.pool3(x)
        #print(x.shape)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.pool4(x)
        #print(x.shape)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.pool5(x)
        #print(x.shape)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.drop6(x)
        #print(x.shape)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.drop7(x)
        #print(x.shape)
        x = self.fc3(x)
        #print(x.shape)
        x = self.up(x)
        #print(x.shape)
        return x

if __name__ == "__main__":
    input = torch.randn(8, 3, 512, 512)
    model = fcn()
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    print(model)
    model(input)
