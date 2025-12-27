# ==========================================================
# This file includes the agent class
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np
import random
from itertools import product
# =============================================================================
# 1. Agent class
# =============================================================================
class agent_class:
    def __init__(self, P, V, U, num_grid_p, num_grid_v, gamma, total_interaction, calculate_return_immediate, epsilon_decay, T_decay):
        """
        Initialize an agent.

        Args:
            S : np.ndarray
                Array of states
            A : np.ndarray
                Array of actions
            gamma : float
                Discount factor in [0, 1]
        
        Returns:
            agent object
        """
        self.P = P
        self.V = V
        self.A = U
        self.num_grid_p = num_grid_p
        self.num_grid_v = num_grid_v
        self.p_bins, self.v_bins, self.grid_dict = self.rectangle_discretized_state_space(P, V, num_grid_p, num_grid_v)
        self.S = list(set(self.grid_dict.keys()))
        self.gamma = gamma  # Discount factor

        self.total_interaction = total_interaction
        self.calculate_return_immediate = calculate_return_immediate
        self.epsilon_decay = epsilon_decay
        self.T_decay = T_decay
        self.list_of_returns = []

        # Check discount factor validity
        if not (0 <= gamma <= 1):
            raise ValueError("Discount factor γ must be between 0 and 1")
        

    def Q_learning_func(self, obj_env, initial_state_idx , select_action_strategy, EPISODE_BLOCK, epsilon=0.9, T=1, epsilon_min=0.01, T_min = 0.05):
        """
        Q_learning algorithm.

        Args:
            obj_env : object
                An object from environment.
            select_action_strategy : string
                Name of function to choose the action.
            max_episodes : int
                Number of max iteration.

        Returns:
            Q_table : numpy array 2D
                The Q value for all states and actions pairs.
            epsilon_values: List
                List of all epsilons.
        """
        
        # Initializing
        Q_hat = np.zeros((len(self.S),len(self.A)), dtype=float)
        num_visit = np.zeros((len(self.S),len(self.A)), dtype=int) 
        epsilon_values = [] # to check the condition of epsilon later
        total_interaction_manual = 0 # in all episodes

        # inside of a block of episodes
        for episode in range(EPISODE_BLOCK):
            
            # reset env

            obj_env.reset(initial_state_idx, self.S)
            s_idx = obj_env.state

            print(obj_env.done)
            print(obj_env.state, s_idx)
            print(obj_env.steps)
            
            R = 0
        
            # Inside an episode
            while not obj_env.done:
                
                # choose an action by one of the strategies
                if select_action_strategy == "epsilon_greedy":
                    
                    a_idx = self.epsilon_greedy(s_idx, Q_hat, self.A, epsilon) 
                
                elif select_action_strategy == "Boltzmann":
                    
                    a_idx = self.boltzmann(s_idx, Q_hat, self.A, T) 

                else:
                    raise ValueError("No action strategy was provided.")
                
                # observe st+1 and rt
                if obj_env.transition_fn and obj_env.reward_fn:
                    current_p, current_v = self.find_p_v(cell_id=obj_env.state, grid_dict=self.grid_dict)
                    next_p, next_v, reward = obj_env.interaction(current_p, current_v, self.A[a_idx])

                else:
                    raise ValueError("No transition of reward function was provided.")
                
                # calculate the TD error
                next_state_idx = self.find_cell(next_p, next_v, self.p_bins, self.v_bins)
                TD_error = reward + (self.gamma * (np.max(Q_hat[next_state_idx,:]))) - Q_hat[s_idx, a_idx]
                
                # calculate learning step
                if obj_env.deterministic: 
                    learning_step = 1
                else: # stochastic
                    learning_step = 1/(num_visit[s_idx,a_idx] + 1)

                # Calculate reward immediate
                if self.calculate_return_immediate:
                    R += self.calculate_return_immediate_func(self.gamma, obj_env.steps, reward)

                # update Q_hat
                Q_hat[s_idx, a_idx] = Q_hat[s_idx, a_idx] + learning_step * (TD_error)
                print(Q_hat[s_idx, a_idx])
                # update number of visited
                num_visit[s_idx,a_idx] += 1

                # take the next step
                s_idx = next_state_idx

                obj_env.steps += 1 # increase 

            # end of an episode
            self.list_of_returns.append(R)
            
            # save the total interaction
            total_interaction_manual += obj_env.steps

            # save epsilon
            epsilon_values.append(epsilon)
        
            # Decreasing epsilon for epsilon greedy
            epsilon = max(epsilon_min, epsilon * self.epsilon_decay)

            # Decreasing temperture
            T = max(T_min, T * self.T_decay)

        # end of a block of episodes
        
        if self.total_interaction:
           total_interaction = self.total_interaction
        else:
            total_interaction = total_interaction_manual

        return Q_hat, epsilon_values, total_interaction
    
    def rectangle_discretized_state_space(self, p, v, num_grid_p, num_grid_v):
        """
        Discretizing the state space into a rectangle.

        Args:
            p : current postion
            v : current velocity
            num_grid_p : number of cells (horizontal)
            num_grid_v : number of cells (vertical)

        Returns:
            p_bins : Bins of postions
            v_bins : Bins of velocity
            grid_dict : A dictionary of cells in the grid; 
                        keys : cell_id.
                        values : List of 4 corners tuples.
        """
        
        # Discretization parameters
        p_min, p_max = p[0], p[1]
        v_min, v_max = v[0], v[1]
        
        # Define bins
        p_bins = np.linspace(p_min, p_max, num_grid_p + 1)
        v_bins = np.linspace(v_min, v_max, num_grid_v + 1)
        
        # Create dictionary of grid cells
        grid_dict = {} # key = cell_id , value = list of corners
        cell_id = 0
        for i in range(num_grid_p):
            for j in range(num_grid_v):
                corners = [
                    (p_bins[i],   v_bins[j]),     # bottom-left
                    (p_bins[i+1], v_bins[j]),     # bottom-right
                    (p_bins[i+1], v_bins[j+1]),   # top-right
                    (p_bins[i],   v_bins[j+1])    # top-left
                ]
                grid_dict[cell_id] = corners
                cell_id += 1

        return p_bins, v_bins, grid_dict
    
    def find_cell(self, p, v, p_bins, v_bins):
        """
        Get (p,v), find the cell_id corresponding.

        Args:
            p : current postion
            v : current velocity
            p_bins : np.linspace of postion
            v_bins : np.linspace of velocity

        Returns:
            cell_id : key 
        """

        # Find the bin index
        i = np.digitize(p, p_bins) - 1
        j = np.digitize(v, v_bins) - 1

        # Clip indices to stay inside the grid
        i = np.clip(i, 0, len(p_bins) - 2)
        j = np.clip(j, 0, len(v_bins) - 2)

        # Flatten 2D grid to 1D index
        num_v = len(v_bins) - 1   # number of vertical cells (velocity)
        cell_id = i * num_v + j

        return cell_id

    def find_p_v(self, cell_id, grid_dict):
        """
        Get cell_id of dictionary grid, find the p and v corresponding.

        Args:
            cell_id : Key of grid dictionary
            grid_dict : grid dictionary (discretized state space)

        Returns:
            p : position corresponding
            v : velocity corresponding 
        """

        corners = grid_dict[cell_id]
        p = (corners[0][0] + corners[1][0]) / 2
        v = (corners[0][1] + corners[3][1]) / 2

        return p, v
    
    def epsilon_greedy(self, s_idx, Q_hat, A, epsilon):
        if np.random.rand() < epsilon:
            # explore: random action
            return np.random.randint(0, len(A))
        else:
            # exploit: choose best action
            a = np.argmax(Q_hat[s_idx])
            return a
        
    def boltzmann(self, s_idx, Q_hat, A, T):
            
        """
        Boltzmann (softmax) action selection.

        Args:
            s_idx : int
                Current state index
            Q_hat : np.ndarray
                Q-table of shape (num_states, num_actions)
            A : np.ndarray or list
                Action space
            T : float
                Temperature (T > 0)

        Returns:
            a_idx : int
                Selected action index
        """
        if T <= 0:
            raise ValueError("Temperature T must be > 0")

        # Extract Q-values for the current state
        q_values = Q_hat[s_idx]

        # Numerical stability trick: subtract max
        q_values_stable = q_values - np.max(q_values)

        # Compute softmax probabilities
        exp_q = np.exp(q_values_stable / T)
        probs = exp_q / np.sum(exp_q)

        # Sample action according to the probabilities
        a_idx = np.random.choice(len(A), p=probs)

        return a_idx
    
    def stop_condition(self, gamma, step, current_state_idx, total_interaction):
        
        if total_interaction != None: # if there is a number of interactions
            if step == total_interaction:
                return True
            
        if self.terminal_state_idx == None: # there is No terminal state
            max_steps = 5000
            return gamma**step <= 1e-6 or step >= max_steps
        else:
            if current_state_idx == self.terminal_state_idx:
                return True
            
        if current_state_idx in [5,7,11,12]: # for TUX penguin
            return True
    
    def calculate_return_immediate_func(self, gamma, step, reward):
        return (gamma ** step) * reward
