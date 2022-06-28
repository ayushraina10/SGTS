import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time
import os

#models
from Policy.Truss.model_enc import TrussEnc
from Policy.Truss.model_spa import TrussSpa
from Policy.Truss.model_sel import TrussSel

from Env.EnvWrapper import EnvWrapper

class DSN(nn.Module):
    
    def __init__(self, env_params, device, checkpoint_dir):
        super().__init__()
        
        self.env = EnvWrapper(**env_params)
        # print("New environment has been initialized")
        
        #initialize the networks
        self.encoder = TrussEnc()
        self.spatial = TrussSpa()
        self.selection = TrussSel() #HYPE 64 is the batch size, can be changed depending on experiments

        self.noise_mixing = 0.5 #HYPE how much to explore and add randomness to it

        self.device = device
        self.checkpoint_dir = checkpoint_dir

        if checkpoint_dir != "":
            # print("LOAD THE INDIVIDUAL MODEL WEIGHTS")
#             import pdb; pdb.set_trace()
            checkpoint_enc = torch.load(checkpoint_dir+'_enc.pt', map_location = "cpu")
            checkpoint_spa = torch.load(checkpoint_dir+'_spa.pt', map_location = "cpu")
            checkpoint_sel = torch.load(checkpoint_dir+'_sel.pt', map_location = "cpu")

            self.encoder.load_state_dict(checkpoint_enc)
            self.spatial.load_state_dict(checkpoint_spa)
            self.selection.load_state_dict(checkpoint_sel)

        self.encoder.to(device)
        self.spatial.to(device)
        self.selection.to(device)

        self.encoder.eval()
        self.spatial.eval()
        self.selection.eval()
    
    def get_action(self, compState, explore = False):
        """
        Input: compState = [Image, members, nodes, iteration]
        Output: action_distribution, feasible_actions"""
        
        #convert into torch tensors, convert to cuda depending on the requirement
        image = torch.from_numpy(compState[0]).type(torch.float32).to(self.device).unsqueeze(0)/255.0
        
        latent = self.encoder(image)
        sr, value = self.spatial(latent)
        
        #prepare spatial region for interaction with environment
        spatial_region = sr.detach().cpu().numpy()
        
        if explore:
            #only occurs when progressive widening is required during training....
            #add some random noise to the predicted spatial region....
            noise = np.clip(np.random.normal(loc = 0, scale = 0.5, size = spatial_region.shape), -1, 1)
            spatial_region = (self.noise_mixing)*spatial_region + noise*(1-self.noise_mixing)
            #noise mixing = 0 is completely random spatial region

        # if defined_actions == None:
        valid_actions = self.env.env.getValidMoves(state = compState, spatial_region = spatial_region)
        if np.all(valid_actions) ==None:
            return 0, [None]
        valid_actions_inp = np.hstack((np.eye(3)[valid_actions[:,0].astype(np.int)], valid_actions[:,1:-1]))
        
        # print("see what they are using", sr, spatial_region)
        # else:
        #     valid_actions_inp = np.array(defined_actions)

        valid_actions_comp = np.zeros((1, 50, 7))
        valid_actions_comp[0, :len(valid_actions_inp), :] = valid_actions_inp

        valid_actions_torch = torch.FloatTensor(valid_actions_comp.astype(np.float32))

        #final probabilities of the valid actions (after log softmax from the model)
        pi = self.selection(latent, valid_actions_torch.to(self.device))

        return [torch.exp(pi[0, :len(valid_actions_inp)]).data.cpu().numpy(), valid_actions_torch.detach().cpu().numpy()[0,:len(valid_actions_inp)]]

    
    def get_value(self, compState):
        """
        Input: compState = [Image, members, nodes, iteration]
        Output: value of state 
        """
        #convert into torch tensors, convert to cuda depending on the requirement
        image = torch.from_numpy(compState[0]).type(torch.float32).to(self.device).unsqueeze(0)/255.0
        
        latent = self.encoder(image)
        _, value = self.spatial(latent)
        
        return 0*(value.data.cpu().numpy()+1)

    def train_step_policy(self, image_batch, action_batch, policy_batch):
        pass

    def train_step_value(self, image_batch, value_batch):
        pass