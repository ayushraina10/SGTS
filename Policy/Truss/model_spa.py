import torch
import torch.nn as nn
import torch.nn.functional as F

class TrussSpa(nn.Module):
    
    def __init__(self):
        super(TrussSpa, self).__init__()
        
        #reduced size
        linear_size = [64, 32]
        latent_size = 512
        action_size = 4 #X1, Y1 and X2, Y2 dimension for the spatial region
            
        self.fc1 = nn.Linear(latent_size,  linear_size[0])
        self.fc_bn1 = nn.BatchNorm1d( linear_size[0])
        
        self.fc2 = nn.Linear( linear_size.pop(0),  linear_size[0])
        self.fc_bn2 = nn.BatchNorm1d( linear_size[0])

        self.pfc3 = nn.Linear( linear_size[0],  action_size)
        
        self.vfc4 = nn.Linear( linear_size[0], 1)#change to 1 for the skill model
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, lat):
        s = self.relu(self.fc_bn1(self.fc1(lat)))  # batch_size x 1024
        s = self.relu(self.fc_bn2(self.fc2(s))) # batch_size x 512

        pi = self.pfc3(s)                                                                         # batch_size x action_size
        v = self.vfc4(s)

        return self.tanh(pi).squeeze(), self.tanh(v).squeeze()
