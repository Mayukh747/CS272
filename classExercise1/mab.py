from random import random

import numpy as np
import matplotlib.pyplot as plt

def ucb1_bandit(arms, num_steps):
    """
    UCB1 algorithm for the multi-armed bandit problem.

    Args:
    arms (list): List of arms (bandit machines) with their true reward probabilities.
    num_steps (int): Number of steps (iterations) to run the algorithm.

    Returns:
    selected_arms (list): List of arms selected at each step.
    rewards (list): List of rewards obtained at each step.
    """
    num_arms = len(arms)
    num_pulls = np.zeros(num_arms)  # Number of times each arm has been pulled.
    total_rewards = np.zeros(num_arms)  # Total rewards obtained from each arm.
    selected_arms = []  # List to store the selected arms.
    rewards = []  # List to store the rewards obtained.

    action_value_estimates = np.zeros(num_arms)

    for step in range(num_steps):
        pass

    return selected_arms, rewards


def epsilon_greedy_bandit(arms, num_steps, epsilon):
    """
    Epsilon-Greedy algorithm for the multi-armed bandit problem.

    Args:
    arms (list): List of arms (bandit machines) with their true reward probabilities.
    num_steps (int): Number of steps (iterations) to run the algorithm.
    epsilon (float): The exploration-exploitation trade-off parameter (0 <= epsilon <= 1).

    Returns:
    selected_arms (list): List of arms selected at each step.
    rewards (list): List of rewards obtained at each step.
    """
    num_arms = len(arms)
    num_pulls = np.zeros(num_arms)  # Number of times each arm has been pulled.
    total_rewards = np.zeros(num_arms)  # Total rewards obtained from each arm.
    selected_arms = []  # List to store the selected arms.
    rewards = []  # List to store the rewards obtained.

    action_value_estimates = np.zeros(num_arms)

    # Helper Function
    def bandit(arm_num):
        """
        Given an arm # return the reward for choosing the arm.
        Increment the #  times the arm has been pulled + update the action value.
        """
        num_pulls[arm_num] += 1
        current_reward = np.random.choice([0,1], p=[1 - arms[arm_num], arms[arm_num]])
        rewards.append(current_reward)
        total_rewards[arm_num] += current_reward

    for step in range(num_steps):
        # greedy initialization
        arm_num = np.argmax(action_value_estimates)
        # reassign with epsilon exploration
        if np.random.rand() <= epsilon:
            arm_num = np.random.randint(0,num_arms-1)

        bandit(arm_num)
        # R, N(A) are updated in the bandit() function
        action_value_estimates[arm_num] += (1/num_pulls[arm_num]) * [rewards[-1] - action_value_estimates[arm_num]]

    return selected_arms, rewards
