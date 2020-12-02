import nauts_ml_envs as nauts
import os
# import system
import time


# Setting the pathways to different images and files 
game_folder = os.path.dirname(__file__)
img_folder = os.path.join(game_folder, "img")
background_file = os.path.join(img_folder, "starfield.png")
player_file = os.path.join(img_folder, "playerShip1_orange.png")
enemy_file = os.path.join(img_folder, "meteorBrown_med1.png")

# Initializing the environment object
env = nauts.space_evaders.env(background_file, player_file, enemy_file)

# Variable for done, when the episode is done
done = False

while not done:
    # Plays through the game in every iteration
    env.render()
    # Takes a step depending on the action. Action can be 0, 1, 2. Step only does ONE ACTION.
    # It also returns 3 variables, obs, flags, done.
    obs, flags, done = env.step(1)
    print(obs)
    # Slows down the game.
    time.sleep(0.1)

# When done is set to True, the game will run the quit event and close
env.quit()