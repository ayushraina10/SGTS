from Env.Environments.truss.truss import TrussWrapper
import numpy as np

#function to convert matrices to dtype formulations
def sort_state(members, nodes):
    if len(nodes)>5:
        arg = np.lexsort((nodes[5:, 1], nodes[5:, 0]))
        nodes = np.vstack((nodes[:5], nodes[5:][arg]))
        if len(members)>0:
            convert = {0:0, 1:1, 2:2, 3:3, 4:4}
            for ctr, a in enumerate(arg):
                convert[a+5] = ctr+5

            new_members = {}
            for idd in members:
                new_members[(convert[idd[0]], convert[idd[1]])] = members[idd]
            members = new_members.copy()
    
    return members, nodes

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    This is a 1 player problem and hence dont require self play
    """
    def __init__(self, scenario, max_iterations, random_start = 0, seed = 0):
        self.max_iterations = max_iterations
        self.scenario = scenario
        self.env = TrussWrapper(scenario = scenario, max_iterations = max_iterations, random_start = random_start)
        self.seed = seed
        self.env.seed = seed
        self.done = False

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        raise NameError('Not Implemented')

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return 3, 128, 128

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
            50 possible actions or 3+4 dimensions each
        """
        return 50, 7

    def get_state(self):
        """return the current compState"""
        return [self.env.tu._plot_state([self.env.members, self.env.nodes]), self.env.members, self.env.nodes, self.env.iteration]
    
    def render(self):
        return self.env.tu._plot_state([self.env.members, self.env.nodes])
    
    def resetState(self, compState = None):
        """
        input:
            state = [observation, members, nodes, iteration]
        """
        if compState == None:
            self.env.members, self.env.nodes, self.env.iteration = self.env.tu._loadScenario(compState, scenario=self.scenario)
        else:
            self.env.members, self.env.nodes, self.env.iteration = compState[1], compState[2], compState[3]
        
        self.done = False
        self.env.seed = self.seed
        
        return [self.env.tu._plot_state([self.env.members, self.env.nodes]), self.env.members, self.env.nodes, self.env.iteration]
    
    def getGameEnded(self, compState):
        """
        Input: None

        Returns: reward (in the range of -1 to 1), True if end of episode"""
        reward, done = self.env.tu.calcReward(compState[1], compState[2], compState[3], self.max_iterations)
        return np.around(reward, decimals = 3), done
        
    def getNextState(self, action = [], state = [], update = True):
        """
        Input:
            action: the action to be taken (noramlized and one-hot)
            state: the parameteric representation of the state (memebers, nodes)
            update: if the state needs to be updated after this action is taken

        Returns:
             [observation, members, nodes, self.iteration], reward (in the range of -1 to 1), done, None
        """
        #action = None is a way of saying no feasible actions exist and hence the trajectory should end
        if np.all(action) != None:
            action = np.array([np.argmax(action[:3]), action[3], action[4], action[5], action[6]])*np.array([2,6,5,6,5])
        
        compState, reward, self.done, _ = self.env.step(action = action, state = state[1:], update = update)
        
        return compState, np.around(reward, decimals = 3), self.done, None

    def getValidMoves(self, state = [], spatial_region = []):
        """
        Input:
            state: the parameteric representation of the state (memebers, nodes)
            spatial_region: the aptial region for biasing the the sampling of feasible actions 

        Returns:
            validMoves: N x action definition (where N is the number of feasible actions in the state)
            action fmt: [0or1or2, -1:1, -1:1, -1:1, -1:1, 0:1]
        """
        actions = self.env.validActions(state = state[1:], spatial_region = spatial_region*np.array([6,5,6,5]))
        if np.all(actions) == None:
            return [None]
        return actions/np.array([2,6,5,6,5,1])

    def getValue(self):
        """
        Input:
            None

        Returns:
            fos, swr, mass of the current state of truss   
        """
        return self.env.tu.evaluate(self.env.members, self.env.nodes)

    def getCanonicalForm(self, state):
        """
        Input:
            state: the parameteric representation of the state (memebers, nodes)

        Returns:
            canonicalBoard:
        """
        raise NameError('Not Implemented')

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: 
        """
        raise NameError('Not Implemented')

    def stringRepresentation(self, state = []):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        if state == []:
            members, nodes, iteration = self.env.env.member_info, self.env.env.node_info, self.env.env.iteration
        else:
            
            _, members, nodes, iteration = state
        nodes = np.around(nodes, decimals = 3)
        
        #nodes
        members, nodes = sort_state(members, nodes)
        
#         string_rep = str(iteration).zfill(3)+':'
        string_rep = ''

        #convert to string
        for i in range(len(nodes)):
            string_rep+=(str(nodes[i,0])+','+str(nodes[i,1]))
            string_rep+=':'
        string_rep=string_rep[:-1]
        
        string_rep+='M'
        if members == {}:
            return string_rep
        
        #member
        for key in members:
            string_rep+=str(key)+';'+str(members[key])+':'
        string_rep=string_rep[:-1]
        
        return string_rep

    def actionToString(self, array):
        string = ''
        if len(array.shape) == 2:
            array = array[0]
        for item in array:
            string+=str(np.around(item, decimals = 3))
            string+=','
        return string[:-1]
    
    def stringToAction(self, string):
        return np.array([float(x) for x in string.split(',')])
    
    