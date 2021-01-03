# import nauts_ml_envs
# import os
# import sys
# import time
# import numpy as np


# LEARNING_RATE = 0.1
# EPISODES = 25_000
# SHOW_EVERY = 1000

# game_folder = os.path.dirname(__file__)
# img_folder = os.path.join(game_folder, "img")
# background_file = os.path.join(img_folder, "starfield.png")
# player_file = os.path.join(img_folder, "playerShip1_orange.png")
# enemy_file = os.path.join(img_folder, "meteorBrown_med1.png")

# env = nauts_ml_envs.space_evaders.env(background_file, player_file, enemy_file)

# TABLE_SIZE = [10]+[20]*env.num_enemies + [3]
# q_table = np.random.uniform(low=-2, high=0, size=(TABLE_SIZE))

# # remap raw state to interpreted state
# def state_mapping(raw_state):
#     mapped_state = []
#     for i in range(len(raw_state)):
#         mapped_state.append(int(raw_state[i] - env.get_objs_observation_space_low()[i]))
#     return tuple(mapped_state)

# # define interpreter function
# def interpreter(observations):
#     orig_state = observations[0]
#     flags = observations[1]

#     reward = 0

#     # map orignal state to range of 0 - 20
#     interpreted_state = state_mapping(orig_state)

#     # implement a reward policy
#     # penalize collisions
#     if flags[0] == 1:
#         reward = reward - 20

#     return interpreted_state, reward


# def agent_descion(st):
#     # Making a descsion based of the state inputted
#     action = np.argmax(q_table[st])
#     return action


# def agent_learning(st_current, action, reward):
#     # Using the Bellman's Equation to update the q value, tuple can be used as index, tuple+ tuple
#     current_q = q_table[st_current + (action, )]
#     new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE 

#     # Updating the q_table
#     q_table[st_current] = new_q


# collisions = 0

# for episode in range(1, EPISODES + 1):
#     done = False

#     # Time 0
#     init_state = env.reset()
#     # New state coming from interpretation of the enviornment
#     st = state_mapping(init_state)
#     while done == False:
#         #determine action to take
#         action = agent_descion(st)


#         # In the environment you are in raw_state_new, but its not mapped yet

#         #step through the environment with action, st+1
#         raw_state_new, flags, done = env.step(action)
#         obs = [raw_state_new, flags]

#         # Getting a new state from the enviornment that occured from the action taken
#         # Remapping the values to a value of 0- 20, also gives a reward
#         state_new, reward = interpreter(obs)

#         # Perform learning

#         # TO learn you update the q value from state reward and the action
#         agent_learning(st, action, reward)

#         # Update metrics, first element of flags gives the collision, 1 if collision, 0 if no collision
#         collisions += flags[0]

#         # Updating the state to the new state, turning s(t+1) --> s(t)
#         st = state_new

#         # It will only render every 1000 episodes, only true if episode is a multiple of 1000
#         if (episode % SHOW_EVERY == 0):
            
#             env.render()
#             time.sleep(0.1)
#     if (episode % SHOW_EVERY == 0):
#         print(f"episode: {episode} collisions: {collisions} in the last thousand episodes")
#             # Resetting the collision count
#         collisions = 0        
    
# env.quit()

import nauts_ml_envs
import os
import sys
import time
import numpy as np
import pickle


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
points = list(range(0,10))
# Initializing list to count the areas where it died the most
students = [0 for i in range(10)]
# ax.bar(langs,students)
# plt.show()

game_folder = os.path.dirname(__file__)
img_folder = os.path.join(game_folder, "img")
background_file = os.path.join(img_folder, "starfield.png")
player_file = os.path.join(img_folder, "playerShip1_orange.png")
enemy_file = os.path.join(img_folder, "meteorBrown_med1.png")

LEARNING_RATE = 0.1
EPISODES = 25_000
SHOW_EVERY = 1_000

env = nauts_ml_envs.space_evaders.env(background_file, player_file, enemy_file)

TABLE_SIZE = [10]+[20]*env.num_enemies + [3]

with open(r"qtable-1608084196.pickle", "rb") as f:
    q_table = pickle.load(f)

# q_table = np.random.uniform(low=-2, high=0, size=(TABLE_SIZE))

# remap raw state to interpreted state
def state_mapping(raw_state):
    mapped_state = []
    for i in range(len(raw_state)):
        mapped_state.append(int(raw_state[i] - env.get_objs_observation_space_low()[i]))
    return tuple(mapped_state)

# define interpreter function
def interpreter(observations):
    orig_state = observations[0]
    flags = observations[1]

    reward = 0

    # map orignal state to range of 0 - 20
    interpreted_state = state_mapping(orig_state)

    # implement a reward policy
    # penalize collisions

    if flags[0]== 1:
        reward = reward - 20

    # Checking if there a object directly overhead not including the character, they are directly overhead if there is a 10
    if (10 in interpreted_state[1:]):
        reward = reward - 10
    
    # if ((interpreted_state[0] == 1) or interpreted_state[0] == 8):
    #     reward = reward - 5

    return interpreted_state, reward

def agent_decision(st):
    action = np.argmax(q_table[st])
    return action

def agent_learning (st_curr, action, reward):
    current_q = q_table[st_curr + (action, )]
    new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * reward 
    q_table[st_curr + (action, )] = new_q

collisions = 0
try:
    for episode in range(1, EPISODES + 1):
        done = False
        init_state = env.reset()
        st = state_mapping(init_state)
        print_flag = 1
        while done == False:
            #determine action to take
            action = agent_decision(st)

            #step through the environment with action
            raw_state_new, flags, done = env.step(action)
            obs = [raw_state_new, flags]

            # interpret observations
            st_new, reward = interpreter(obs)

            # perform learning
            agent_learning(st, action, reward)

            #update metrics
            if flags[0]== 1:
                collisions = collisions + 1

            st = st_new

            if episode%SHOW_EVERY == 0:
                if print_flag == 1:
                    print("episode:", episode, "---", collisions, "in the last 1000 episodes")
                    collisions = 0
                    print_flag = 0
                env.render()
                time.sleep(0.1)
            # # If the game is done print the observations
            # if (done):
            #     # If collision
            #     if (flags[0] == 1):
            #         # Players positions
            #         num_to_change = st[0]
            #         # Changes the collision count of the position of the collision
            #         students[num_to_change] += 1




    # Saving the table
    with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
        pickle.dump(q_table, f)     
    
    # ax.bar(points, students)
    # plt.show()
    env.quit()
except Exception as e:
    print(e)
    env.quit()


