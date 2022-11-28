import gym
import gym_truss_design
import numpy as np

from scipy.stats import multivariate_normal

#any specific functions for interfacing the environment with the wrapper

class TGame():
    """
    This class defines the functions used for an MCTS implementation for Truss design. It calls the environment truss-v0 and provides access to the inbuild built functions using the env variable.
    """

    def __init__(self, max_iterations = 50, action_limit = 50, scenario = 1):
        
        pass
        
    def reset(self, state=[], random_start = 0, seed = 0):
        
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        
        '''Returns: 128x128x3 image of the current design state'''
        pass

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        
        '''Returns: image size,
        TODO: check where and why it is used, adjustt accordingly'''
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of possible actions in a particular state
        """
        '''
        Input:
            Nodes
            Members
            
        Returns: All possible actions in the particular state / Possibly a constant number that we define as the computational limit TODO: rethink this?'''
        pass
    

    def getNextState(self, action = [0,0,0,0,0], state = []):
        ''' Input: 
                Image: current visual state (not needed)
                Nodes: node locations
                Members: dict containing node pointers and thickness
                
                Action: finalized action taken by the agent [0,1,0 , x1,y1,x2,y2] - normalized form
        
            Returns:
                Next visual state
                Next node state
                Next member state
                next iteration number
                
                reward
                
                if done
                
                
            TODO: wrap around the step function in the environment (possibly even just call the environment here and then "close" it)'''
        pass
    
    def getSymmetries(self, compState, pi, actions):
        pass

    def getValidMoves(self, state = [], spatial_region = [], defined_action = [], defined_list = [], k_min = 1):

        '''Input:
                Nodes
                Members
                
                Spatial regions (optional) or discrete action
                
            Returns"
                valid set of moves (with preferance rank)
                
                (defined_action is the special case when a aprticular continuous action needs to be in the list, important for the generation of dataset)
                
                (defined list is a special case when a particular list of actions is required, the fuinction verifies of the indifidual actions are valid and then assign a particular importance value to each actions and outputs a shuffled list), used for predicting over a set of visited actions in mcts
                '''
        pass
        
        
    def getMoveIndex(self, defined_action, action_list):
        """Input:
                Action vector
                Action list
                
        Output:
                Index
        Used for creation of the dataset
        this became simpler since define_action will ensure that the action lies in the feasible set
        """
        pass
        
        
    def nearestAction(self, action_list, action, params):
        """Input:
                Action list
                Action token
                Action params
           
           Output:
                Finds the nearest action in the feasible list"""
        pass
        
        
        
    def getGameEnded(self, state=[]):
        """
        Input:
            state = [members, nodes, iteration]

        Returns:
            reward, ended(True or False)
               
        """
        pass

    def getCanonicalForm(self, board = []):
        """
        Input:
            all state input: image, member_info, node_info

        Returns:
            canonicalForm: if state == None then generate an image and output it, otherwise just use the image itself. 
        """
        
        '''TODO: possibly dont need and can be removed'''
        pass
    
    def stringRepresentation(self, state = []):
        """
        Input:
            state = [members, nodes, iteration]

        Returns:
            boardString: a quick conversion of board to a string format. Required by MCTS for hashing.
                         represent all the nodes and members components together as a string. Sort the nodes and member locations and put it in a string
        example: "iter: x1, y1 : x2, y2 M x1, y1, x2, y2, t1 :"
        """
        
        '''TODO: equivalent of state representation for hashing, ideally replace with image input or latent representation
        
        
        '''
        pass
    
    def stringToParametericRepresentation(self, string = []):
        '''Input: string representation of state
           Output: nodes and member matrix representation'''
        pass
        
        
    def actionToString(self, array):
        pass

            
            
            