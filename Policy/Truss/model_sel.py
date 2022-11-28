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
#         self.maxpool = nn.MaxPool1d(50)

        self.relu = nn.ReLU()
        # print("new sel model")
    
    def forward(self, enc_state, inp):
        #unprocessed action input, variable size set of feasible actions
        batch, n_pts, act_size = inp.size()
        
        x = self.relu(self.bn1(self.conv1(inp[:,:,:3].unsqueeze(1)))).squeeze(3)
        # print('1', x.shape, x.stride())
        #batch, 1, N, 3 -> batch 1, 64, N, 1
        # x = x.permute(0,3,2,1)
        # print('1', x.shape, x.stride())
        
        y = self.relu(self.bn2(self.conv2(inp[:,:,3:].unsqueeze(1)))).squeeze(3)
        # print('2', y.shape, y.stride())
        #batch, 1, N, 4 -> batch, 1, 64, N, 1 -> batch, 1, 64, N
        # y = y.permute(0,3,2,1)
        # y = y.permute(0,3,2,1)
        
        # print('2', y.shape, y.stride())
        
        y = self.relu(self.bn3(self.conv3(y.unsqueeze(1)))).squeeze(2)
        # print('3', y.shape, y.stride())
        #batch, 1, 64, N -> batch, 1, 64, N, 1 -> batch, 1, 64, N
        # y = y.permute(0,3,2,1)
        # print('3', y.shape, y.stride())
        
        y = self.relu(self.bn5(self.conv5(y.unsqueeze(1)))).squeeze(2)
        # print('4', y.shape, y.stride())
        #batch, 1, 64, N
        # y = y.permute(0,3,2,1)
        # print('4', y.shape, y.stride())
        
        
        # print('4-5', x.size(), y.size())
        
        x = torch.cat((x, y), 1)
        # print('4.5', x.size(), x.stride())
        
        # x = x.permute(0,3,2,1)
        # print('4.5', x.shape, x.stride())
    
        x = self.relu(self.bnc(self.convc(x.unsqueeze(1)))).squeeze(2)
        # print('5', x.size(), x.stride())
        # x = x.permute(0,3,2,1)
        # print('5', x.shape, x.stride())
        
        # print(enc_state.size(), "enc size")
        enc_state = enc_state.unsqueeze(2).repeat(1, 1, n_pts)
        # enc_state = enc_state.permute(0,3,2,1)
        # print('5.5', enc_state.shape, enc_state.stride())
        # print(enc_state.size(), "enc size")
        
        x = torch.cat((x, enc_state), dim = 1)
        # print('6', x.size(), x.stride())
        # x = x.permute(0, 2, 1).unsqueeze(1).detach().cpu().numpy()
        # x = torch.from_numpy(x).float().to(enc_state.device)
        # x = x.permute(0, 2, 1).unsqueeze(1)
        # print('7', x.size(), x.stride())
        
        x = self.relu(self.bn6(self.conv6(x.unsqueeze(1)))).squeeze(2)
        # print('8', x.size(), x.stride())
        # x = x.permute(0, 2, 1).unsqueeze(1).detach().cpu().numpy()
        # x = torch.from_numpy(x).float().to(enc_state.device)
        # x = x.permute(0,3,2,1)
        # print('8', x.shape, x.stride())
        
        # x = x.permute(0, 2, 1).unsqueeze(1)
        # print('9', x.size(), x.stride())
        
        # x = self.conv7(x).squeeze(3).transpose(2, 1).detach().cpu().numpy()
        # x = torch.from_numpy(x).float().to(enc_state.device)
        
        x = self.conv7(x.unsqueeze(1)).squeeze(2)#.permute(0, 2, 1)
        # print('10', x.size(), x.stride())
        
        # x = x.permute(0,3,2,1)
        # print('10', x.shape, x.stride())

        x = self.avgpool(x.transpose(2,1)).squeeze()
        # print('11', x.shape, x.stride())
        
#         #high negative value to the infeasible action to prevent backprop of loss terms
        # print(torch.sum(inp[:,:,:3], dim = 2).size(), x.size(), torch.sum(inp[:,:,:3], dim = 2) == 0)
    
        x = torch.where(torch.sum(inp[:,:,:3], dim = 2) == 0, torch.tensor(-1e+8).to(enc_state.device), x)
        
        # x = torch.rand((batch, n_pts)).to(enc_state.device)
        x = F.log_softmax(x, dim = 1)
        # print('12', x.shape, x.stride())
        
        return x