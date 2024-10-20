from typing import Any
import random
import gymnasium as gym
import numpy as np


def argmax_action(d: dict[Any,float]) -> Any:
    """return a key of the maximum value in a given dictionary 

    Args:
        d (dict[Any,float]): dictionary

    Returns:
        Any: a key
    """
    return max(d, key=d.get)

class ValueRLAgent():
    def __init__(self, env: gym.Env, gamma : float = 0.98, eps: float = 0.2, alpha: float = 0.02, total_epi: int = 5_000) -> None:
        """initialize agent parameters
        This class will be a parent class and not be called directly.

        Args:
            env (gym.Env): gym environment
            gamma (float, optional): a discount factor. Defaults to 0.98.
            eps (float, optional): the epsilon value. Defaults to 0.2. Note: this pa uses a simple eps-greedy not decaying eps-greedy.
            alpha (float, optional): a learning rate. Defaults to 0.02.
            total_epi (int, optional): total number of episodes an agent should learn. Defaults to 5_000.
        """
        self.env = env
        self.q = self.init_qtable(env.observation_space.n, env.action_space.n)
        self.gamma = gamma
        self.eps = eps
        self.alpha = alpha
        self.total_epi = total_epi
    
    def init_qtable(self, n_states: int, n_actions: int, init_val: float = 0.0) -> dict[int,dict[int,float]]:
        """initialize the q table (dictionary indexed by s, a) with a given init_value

        Args:
            n_states (int, optional): the number of states. Defaults to int.
            n_actions (int, optional): the number of actions. Defaults to int.
            init_val (float, optional): all q(s,a) should be set to this value. Defaults to 0.0.

        Returns:
            dict[int,dict[int,float]]: q table (q[s][a] -> q-value)
        """
        q = {}
        for s in range(n_states):
            q[s] = {}
            for a in range(n_actions):
                q[s][a] = init_val
        return q

    def eps_greedy(self, state: int, exploration: bool = True) -> int:
        """epsilon greedy algorithm to return an action

        Args:
            state (int): state
            exploration (bool, optional): explore based on the epsilon value if True; take the greedy action by the current self.q if False. Defaults to True.

        Returns:
            int: action
        """
        if exploration and random.random() <= self.eps:
            return random.choice(list(self.q[state].keys()))
        else:
            return argmax_action(self.q[state])


    def choose_action(self, ss: int) -> int:
        """a helper function to specify a exploration policy
        If you want to use the eps_greedy, call the eps_greedy function in this function and return the action.

        Args:
            ss (int): state

        Returns:
            int: action
        """
        return self.eps_greedy(ss)


    def best_run(self, max_steps: int = 100) -> tuple[list[tuple[int,int,float]], bool]:
        """After the learning, an optimal episode (based on the latest self.q) needs to
        be generated for evaluation. From the initial state, always take the greedily best
        action until it reaches a goal.

        Args:
            max_steps (int, optional): Terminate the episode generation if the agent cannot
            reach the goal after max_steps.
            One step is (s,a,r) Defaults to 100.

        Returns:
            tuple[
                list[tuple[int,int,float]],: An episode [(s1,a1,r1), (s2,a2,r2), ...]
                bool: done - True if the episode reaches a goal, False if it hits max_steps.
            ]
        """
        episode = []
        reached_terminal_state = False
        current_state, info  = self.env.reset()
        
        for step in range(max_steps):
            if reached_terminal_state:
                break

            greedy_action = argmax_action(self.q[current_state])

            next_state, reward, reached_terminal_state, truncated, info = self.env.step(greedy_action)

            episode.append((current_state, greedy_action, reward))

            current_state = next_state

        return episode, reached_terminal_state

    def calc_return(self, episode: list[tuple[int,int,float]], done=False) -> float:
        """Given an episode, calculate the return value. An episode is in this format: [(s1,a1,r1), (s2,a2,r2), ...].

        Args:
            episode (list[tuple[int,int,float]]): An episode [(s1,a1,r1), (s2,a2,r2), ...]
            done (bool, optional): True if the episode reaches a goal, False if it does not. Defaults to False.

        Returns:
            float: the return value. None if done is False.
        """
        episode_reward_by_step = []
        for step_num in range(len(episode)):
            step = episode[step_num]
            episode_reward_by_step.append(episode[step_num][2] * \
                                          self.gamma ** step_num)
        return sum(episode_reward_by_step)

class MCCAgent(ValueRLAgent):
    def learn(self) -> None:
        """Monte Carlo Control algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        Note: When an episode is too long (> 500 for CliffWalking), you can stop the episode and update the table using the partial episode.

        The results should be reflected to its q table.
        """

        max_steps = 500
        
        pass

class SARSAAgent(ValueRLAgent):
    def learn(self) -> None:
        """SARSA algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.

        follow the psuedocode on slide 21 for the learning algorithm
        for the SARSA TD control algorithm

        """
        # Loop for each episode
        for epi in range(self.total_epi):
            # Initialize S init state
            s, info = self.env.reset()  # 36
            # Choose A from S using policy derived from Q (greedy)
            a = self.eps_greedy(s)

            # Loop for each step until the end of the episode
            done = False
            while not done:
                # Take Action A ==> observe R, S'

                # next_state, reward, reached_terminal_state, truncated, info, done = self.env.step(greedy_action)

                ss, r, done,truncated, info = self.env.step(a)
                # print(f's,a,r,ss: {s}, {a}, {r}, {ss}')

                # Choose A from S' using policy derived from Q (greedy)
                aa = self.eps_greedy(ss)
                # Q(S,A) <- Q(S,A) + alpha[R + gamma*Q(S',A') - Q(S,A)]
                # update Q value based on (s,a,r,ss)
                # (s, a, r, ss)
                # Q(ss, ??)

                # print("ss:", ss)
                # print("action map", self.q[ss])
                # print("aa:", aa)
                # print("q value for action", self.q[ss][aa])

                bootstrapped_estimate = r + self.gamma * self.q[ss][aa]
                self.q[s][a] += self.alpha * (bootstrapped_estimate - self.q[s][a])
                # S <- S', A <- A';
                s = ss
                a = aa

class QLAgent(SARSAAgent):  
    def learn(self):
        """Q-Learning algorithm
        Update the Q table (self.q) for self.total_epi number of episodes.

        The results should be reflected to its q table.
        """   
        pass

    def choose_action(self, ss: int) -> int:
        """
        [optional] You may want to override this method.
        """   
        pass