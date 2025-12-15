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
    def __init__(self, transition_fn, reward_fn):
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
        next_p, next_v = self.transition_fn(current_p, current_v, action, next_p, next_v)

        # Reward function
        reward = self.reward_fn(next_p, next_v, action)
        done = self.stop_condition()

        return next_p, next_v, reward, done
    
    def stop_conditon(self):
        pass

