from ddq import Agent
import numpy as np
import gym 
from utils import plot_learning_curve

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    n_games = 400
    agent = Agent(lr = 1e-3, gamma = 0.99, epsilon=1, batch_size=64, input_dims=[8], n_actions = 4)
    scores, eps_history = [],[]

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_history.append(score)
        avg_score = np.mean(scores[-100:])
        print("episode {}, score: {}".format(i, score))
        print("average score: {}".format(avg_score))
        print("epsilon = {.2f}".format(agent.epsilon))
    filename = 'keras_lunar_lander.png'
    x = [i+1 for i in range[n_games]]
    plot_learning_curve(x, scores, eps_history, filename)
            
