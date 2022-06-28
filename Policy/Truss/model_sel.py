import torch
import torch.nn as nn

import torch.nn.functional as F

class TrussSel(nn.Module):
    def __init__(self, latent_size = 512, priority=False, action_size = (50,7)):
        super(TrussSel, self).__init__()
        
        #action category
        kernel = (1,3)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel)
        self.bn1 = nn.BatchNorm2d(64)
        
        #action parameters
        kernel = (1,5) if priority else (1,4)
        self.conv2 = torch.nn.Conv2d(1, 64, kernel)
        self.conv3 = torch.nn.Conv2d(1, 128, (64, 1))
        self.conv5 = torch.nn.Conv2d(1, 256, (128, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.convc = torch.nn.Conv2d(1, latent_size, (256+64, 1))
        self.bnc = nn.BatchNorm2d(latent_size)
        
        self.conv6 = torch.nn.Conv2d(1, 128, (2*latent_size, 1))
        self.conv7 = torch.nn.Conv2d(1, 32, (128, 1))
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(32)

        self.avgpool = nn.AvgPool1d(32)

        self.relu = nn.ReLU()
    
    def forward(self, enc_state, inp):
        #unprocessed action input, variable size set of feasible actions
        batch, n_pts, act_size = inp.size()
        
        x = self.relu(self.bn1(self.conv1(inp[:,:,:3].unsqueeze(1)))).squeeze(3)
        y = self.relu(self.bn2(self.conv2(inp[:,:,3:].unsqueeze(1)))).squeeze(3)
        
        y = self.relu(self.bn3(self.conv3(y.unsqueeze(1)))).squeeze(2)
        
        y = self.relu(self.bn5(self.conv5(y.unsqueeze(1)))).squeeze(2)
        
        x = torch.cat((x, y), 1)
        
        x = self.relu(self.bnc(self.convc(x.unsqueeze(1)))).squeeze(2)
        
        enc_state = enc_state.unsqueeze(2).repeat(1, 1, n_pts)
        x = torch.cat((x, enc_state), dim = 1)
        
        x = self.relu(self.bn6(self.conv6(x.unsqueeze(1)))).squeeze(2)
        
        x = self.conv7(x.unsqueeze(1)).squeeze(2)
        x = self.avgpool(x.transpose(2,1)).squeeze()
    
        x = torch.where(torch.sum(inp[:,:,:3], dim = 2) == 0, torch.tensor(-1e+8).to(enc_state.device), x)
        
        x = F.log_softmax(x, dim = 1)
        return x