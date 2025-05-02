import gym
import random
import numpy as np

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable)

action = environment.action_space.sample()
"""
sol: 0
asagi: 1
sag: 2
yukari: 3
"""

# S1 -> (Action 1) -> S2
new_state, reward, done, info, _ = environment.step(action)

# %%
import gym
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

environment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
environment.reset()

nb_states = environment.observation_space.n
nb_actions = environment.action_space.n
qtable = np.zeros((nb_states, nb_actions))

print("Q-table:")
print(qtable) # ajanin beyni

episodes = 1000 # episode
alpha = 0.5 # learning rate
gamma = 0.9 # discount rate

outcomes = []	

# training
for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False # ajanin basari durumu 
    outcomes.append("Failure")
    
    while not done: # ajan basarili olana kadar state icerisinde hareket et (action sec ve uygula)
        
        # action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
            
        new_state, reward, done, info, _ = environment.step(action)
        
        # update q table
        qtable[state, action] =  qtable[state, action] + alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        
        state = new_state
        
        if reward:
            outcomes[-1] = "Success"
        
print("Qtable After Training: ")  
print(qtable)

plt.bar(range(episodes), outcomes)
      
# test
episodes = 100 # episode
nb_success = 0

for _ in tqdm(range(episodes)):
    
    state, _ = environment.reset()
    done = False # ajanin basari durumu 
    
    while not done: # ajan basarili olana kadar state icerisinde hareket et (action sec ve uygula)
        
        # action
        if np.max(qtable[state]) > 0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
            
        new_state, reward, done, info, _ = environment.step(action)
                
        state = new_state
        
        nb_success += reward
        
print("Success rate:", 100*nb_success/episodes)           
        
    
    































