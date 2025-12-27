# create problem
from Environment import continues_environment_class
from agent import agent_class
import numpy as np

# Transition function
def transition_calculation_pendulum(current_p, current_v, current_u):
    """
    Calculates the next angular position and velocity of a pendulum given the current state and applied torque.
    
    Args:
        current_p (float) – Current angular position of the pendulum.

        current_v (float) – Current angular velocity of the pendulum.

        u (float) – Applied torque.

        delta_t (float, optional) – Time step for the state update. Default is 0.01.

    Returns:

        new_p (float) – Updated angular position after applying the dynamics.

        new_v (float) – Updated angular velocity after applying the dynamics.
    """
    m = 1
    l=1
    µ = 0.01
    g = 9.81
    delta_t = 0.01

    # 1. Compute acceleration
    a =  (1/m*pow(l,2)) * (-(µ*current_v) + (m*g*l*(np.sin(current_p))) + current_u)

    # 2. Compute next continuous state
    new_v = current_v + (a * delta_t) 
    new_p = current_p + (new_v * delta_t) 
    return new_p, new_v

# Reward function
def reward_calculation_pendulum(current_p, current_v, u, next_p, next_v):

    return np.cos(current_p) - 0.01 * abs(u)

def stop_func(current_p, current_v, action, next_p, next_v):
    # Check if pendulum is upright (within tolerance)
    if abs(current_p) < 0.1 and abs(current_v) < 0.5:
        return True
    return False
    
env_obj = continues_environment_class(transition_calculation_pendulum, reward_calculation_pendulum, stop_func, True)
agent_obj = agent_class(P=(-np.pi, np.pi), V=(-10,10), U=[-5, 0 ,5], 
                        num_grid_p=20, num_grid_v =20, gamma=0.95,
                        total_interaction=None, calculate_return_immediate=False, 
                        epsilon_decay=0.995, T_decay=0.8)

Q_table, epsilon_values, steps = agent_obj.Q_learning_func(env_obj, None, "epsilon_greedy", 
                                                           EPISODE_BLOCK=200, epsilon=1.0, epsilon_min=0.1 )

print(f"Q-table shape: {Q_table.shape}")
print(f"Min Q-value: {np.min(Q_table):.4f}")
print(f"Max Q-value: {np.max(Q_table):.4f}")
print(f"Mean Q-value: {np.mean(Q_table):.4f}")