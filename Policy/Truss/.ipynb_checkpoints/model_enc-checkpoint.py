import torch
import torch.nn as nn
import torch.nn.functional as F
from Policy.Truss.CoordConv_layer import CoordConv

# class TrussEnc(nn.Module):
    
#     def __init__(self, in_channels = 3):
#         super(TrussEnc, self).__init__()
#         # game params
#         self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#         self.fc4 = nn.Linear(12 * 12 * 64, 512)
        
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
        
#         return F.relu(self.fc4(x.view(x.size(0), -1)))
    
    
    
class TrussEnc(nn.Module):
    
    def __init__(self, in_channels = 3):
        super(TrussEnc, self).__init__()
        # game params
        self.conv1 = CoordConv(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(12 * 12 * 64, 512)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        return F.relu(self.fc4(x.view(x.size(0), -1)))