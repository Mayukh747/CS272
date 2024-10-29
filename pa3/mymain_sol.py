import gymnasium as gym
from mypa3_mf_control import *

if __name__ == '__main__':

    env_names = ['CliffWalking-v0', 'FrozenLake-v1', "Taxi-v3"]

    for name in env_names:
        print(name)
        env = gym.make(name)

        # MC
        name = "MC"
        for _ in range(5):
            agent = MCCAgent(env)
            agent.learn()
            epi, done = agent.best_run()
            print(f'{name}: Return: {agent.calc_return(epi, done)}, Best Episode: {epi}')
        print()

        # SARSA
        name = "SARSA"
        for _ in range(5):
            agent = SARSAAgent(env)
            agent.learn()
            epi, done = agent.best_run()
            print(f'{name}: Return: {agent.calc_return(epi, done)}, Best Episode: {epi}')
        print()

        # Q-Learning
        name = "Q-Learning"
        for _ in range(5):
            agent = QLAgent(env)
            agent.learn()
            epi, done = agent.best_run()
            print(f'{name}: Return: {agent.calc_return(epi, done)}, Best Episode: {epi}')
