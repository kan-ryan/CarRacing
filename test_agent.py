from __future__ import print_function

from datetime import datetime
import numpy as np
import gym
import os
import json


from tensorflow import keras




def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    
    print(state)
    while True:
        
        # preprocess
        #crop state in 84 x 84
        state = state[:-12,6:-6]
        #
        state = np.dot(state[...,0:3], [0.299, 0.587, 0.114])
        #scaling 
        state = state/255.0
        state = np.reshape(state, (1, 84, 84))
        print (state.shape)
        
        # get action
        
         
        
        a = agent.predict_on_batch(state)
        #print(a)
        next_state, r, done, info = env.step(a[0])   
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward

def loadModel(filePath):
    #reads in file  from file path and creates a new model from file
    model = keras.models.load_model(filePath)
    return model

if __name__ == "__main__":
    #read in model data from directory and creates a new model
    newModel = loadModel('/Users/ryankan/Documents/MachineLearning/behaviorCloning_CarRacingv0/models/modeltest15.h5')

    #creates new prediciton for newModel and prints out the
    predNM = newModel.predict_on_batch(np.arange(5*84*84).reshape((5, 84, 84, 1)))
    print(predNM)
    
    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 10                  # number of episodes to test


    # load agent
    #agent = pyTorchModel()
    #agent.load_state_dict(torch.load("models/agent.pkl"))


    #creates environment CarRacin-v0
    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, newModel, rendering=rendering)
        episode_rewards.append(episode_reward)

    
    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
   
    with open(fname, "w") as f:
        json.dump(results, f)
    env.close()
    print('... finished')
    

