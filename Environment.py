# ==========================================================
# This file includes the environment class
# ==========================================================

# =============================================================================
# 0. Libraries
# =============================================================================
import numpy as np

# =============================================================================
# 1. Environment class
# =============================================================================
class continues_environment_class:
    def __init__(self, transition_fn, reward_fn, stop_func, deterministic):
        """
        Initialize a Markov Decision Problem (MDP)

        Args:
            transition_fn : Function
                Function to calculate the next position and velocity
            reward_fn : Function
                Function to calculate the reward of the position and velocity
        Returns:
            env object
        """
        self.transition_fn = transition_fn  # Transition probabilities P[s, a, s']
        self.reward_fn = reward_fn          # Reward function R[s, a, s']
        self.stop_func = stop_func # stop condition
        self.state = None
        self.done = None
        self.steps = None
        self.deterministic = deterministic
    
    def reset(self, initial_state_idx, P, V):
        
        self.done = False
        self.steps = 0

        # choose the start state
        if initial_state_idx is None:
            # Choose random continuous state within bounds
            p = np.random.uniform(P[0], P[1])
            v = np.random.uniform(V[0], V[1])
            self.state = (p, v)  # Store as continuous tuple
        else:
            # Convert discrete index to continuous state
            # You need a method to convert discrete to continuous
            p, v = initial_state_idx  # Assuming it's already (p, v)
            self.state = (p, v)

    def interaction(self, current_p, current_v, action):
        """
        response of env to agent taking action 'a' in a state 's' "

        Args:
            current_p: float
                current position
            current_v: float
                current velocity
            action: int
                force as an action
        Returns:
            r : the immediate reward.
            s t+1 : the next state.
        """
        # Transition function
        next_p, next_v = self.transition_fn(current_p, current_v, action)

        # Reward function
        reward = self.reward_fn(current_p, current_v, action, next_p, next_v)

        # Update stop condition with tolerance
        self.done = self.stop_func(current_p, current_v, action, next_p, next_v)

        return next_p, next_v, reward