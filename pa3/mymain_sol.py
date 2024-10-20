import gymnasium as gym
from mypa3_mf_control import *

if __name__ == '__main__':
    env = gym.make('CliffWalking-v0')
    # ags = {'SARSA' : SARSAAgent(env), 'Q-Learning' : QLAgent(env), 'MC' : MCCAgent(env)}
    ags = {'SARSA' : SARSAAgent(env)}

    # Run each agent once. 
    for name, agent in ags.items():
        agent.learn()
        epi, done = agent.best_run()
        print(f'{name}: Best Episode: {epi}, Return: {agent.calc_return(epi, done)}')


