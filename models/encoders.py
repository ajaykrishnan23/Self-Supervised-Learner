from torch.nn import functional as F
from torch import nn
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50
from torchvision.models import mnasnet1_0, mobilenet_v2

class NasNet(nn.Module):

  def __init__(self, pretrained):
    super().__init__()
    self.nasnet = mnasnet1_0(pretrained = pretrained)
    self.nasnet.classifier = nn.Identity()

  def forward(self, x):
    x = self.nasnet(x)
    return [x]

class MobileNet(nn.Module):

  def __init__(self, pretrained):
    super().__init__()
    self.mobilenet = mobilenet_v2(pretrained = pretrained)
    self.mobilenet.classifier = nn.Identity()

  def forward(self, x):
    x = self.mobilenet(x)
    return [x]

class miniCNN(nn.Module):

  def __init__(self, output_dim):
    super().__init__()
    self.output_dim = output_dim

    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1)
    self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 56, kernel_size = 3, stride = 2, padding = 1)
    self.conv3 = nn.Conv2d(in_channels = 56, out_channels = 98, kernel_size = 3, stride = 2, padding = 1)
    self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size = (16, 16))
    self.conv4 = nn.Conv2d(in_channels = 98, out_channels = 64, kernel_size = 5, stride = 2, padding = 2)
    self.pool = nn.MaxPool2d(2, 2)
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(64*8*8, self.output_dim)

  def forward(self, x):
      x = F.relu(self.conv1(x))
      # x = self.pool(x)
      x = F.relu(self.conv2(x))
      # x = self.pool(x)
      x = F.relu(self.conv3(x))
      x = F.relu(self.adaptive_pool(x))
      x = F.relu(self.conv4(x))
      # x = self.pool(x)
      x = self.flatten(x)
      # print(x.shape)
      x = F.relu(self.fc1(x))
      return [x]
