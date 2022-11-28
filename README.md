# Notes for the overall working of the WU-UCT code and how to implement it for the spatially guided version of the formula
- /Tree/WU-UCT.py 
    - has all the code related top the MCTS search and the parallelization is built into the functions for simulation etc.
    - These functions will have to be changed based on how I want to implement the spatially guided version of the code
    - Also note that they are not using the dictionary based methodology which was potentially helpful in the previous code so explore how they are storing state imformation and how it can be implemented for the truss design case
    - Since this code is valid for trees, think about how being a DAG can be different and possibly problematic for this case since referring to "parents" of a node may lead to inconsistency
    
        
- /Node/WU-UCTnode.py
    - This file has all the functions for defining every state in the tree and also associated methods about cloning, selecting action based on UCB, info about children and parents etc (may be memory intensive)
    - 
    
## Flow of the algorithm
- main.py
    - multiprocessing.set_start_method("forksever") <--- // what is this??
    - Initialize tree = /Tree/WU_UCT
    
    - Simulate trajectory using WU_UCT (MCTStree.simuilate_trajectory()) [star]
    
    - save file and results
    
    - Close Tree
    
    
- /Tree/WU_UCT.py
    
    - simulate_trajectory(episode_length)
        - this function not just runs the WU_UCT algorithm but also interacts with the environment and makes iterative decision untill the end of the episode
        - self.wrapped_env() function is used to simplify the design environment such that it can be used for multiprocesing as well?
        - simulate_single_mode(state)
            - this function runs the actual MCTS algorithm and makes the action decision (similar to MCTS.ActionProbs)
        - calculate and store the rewards associated and run until done
        
     - simulate_single_move(state)
         - this function is a wrapper for the actual distributed MCTS algorithm as it: 
             - clears out the cache and recorders
             - free the workers
             - intializes the root node Node/WU_UCTnode
             - runs multiple simulations of the algorithm
                 - runs simulate_single_step(simulation_index)
             - select the best action from the constructed tree
             
     - simulate_single_step(simulation_index)
         the 4 sequential steps of selection, expansion, Assigning simulation tasks, Update the nodes
         
         #important node functions to note
         - no_child_available (checks if no children of the node are available)
         - all_child_visited
         - all_child_visited
         
         - update_history
         - rewards
         
         - dones
         - children
         - checkpoint_idx
         
         - update_history
         - add_child
         
         - update_incomplete
         - parent
    
    
- /Node/WU_UCTnode.py

    - init variables:
        - action_n (list of feasible actions)
        - state (comprehensive representation of the design state)
        - checkpoint_idc (information about the checkpoint index, possibly the info about the state as well)
        - parent (info about the parent to maintain the linked list)
        - tree (???) (possibly the set of connected nodes (children), it defined max_width of tree)
        - is_head (possibly bool variable to define if the node is a root node)
        
        - children (initialized as a list of None based on feasible set of action associated with the state)
        - rewards (initialized as 0.0 for every feasible action)
        - dones (initialized as False for every feasible action)
        - children_visit_count (initialized as 0 for every feasible action)
        - children_completed_visit_count (initialized as 0 for every feasible action) ???
        - Q_values (initialized as 0s)
        - visit_count (specific state's visitation count)
        
        - prior prob (prior probabilities for the set of feasible actions, uniform prior is used if none available)
        - traverse_history = dict() (tree traversal history record)
        - visited_node_count ???
        - updated_node_count ???
        - moving_aveg_calculator ??? (contains the mean and sample variances for some particular value)
        
        
        
- Worker class

    - run????
    - expand node
    - simulate
    
    - get action
    - get value
    - get prior prob
    
    
    
    
## files to edit    
- WU_UCT Tree
- WU_UCT Node

- Worker


- Env Wrapper




## files that have been edited

- Policy/Truss/DSNPolicy.py
    
    



