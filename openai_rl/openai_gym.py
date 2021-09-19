#create a gym environment deep q learning using keras


import gym
import numpy as np
import random
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, LeakyReLU
from keras.models import Sequential


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.tau = .125
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()

    def create_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), padding='same', data_format='channels_last', input_shape=self.env.observation_space.shape))
        model.add(LeakyReLU())
        model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
        model.add(LeakyReLU())
        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU())
        model.add(Dense(self.env.action_space.n))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Time Complexity: O(1)
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    # The agent has a memory of the last 100,000 game steps, which is represented as a list of tuples.
    # 
    # Args:
    #   self: the agent itself
    #   batch_size: How many samples in each minibatch.
    # Returns:
    #   The action with the highest Q-value.
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, new_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(new_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)
    


if __name__ == "__main__":
    batch_size = 32
    env = gym.make('CartPole-v0')
    agent = DQN(env)
    for e in range(100):
        state = env.reset()
        state = np.reshape(state, [1, 84, 84, 1])
        for time in range(1000):
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, 84, 84, 1])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, 100, time, agent.epsilon))
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % 10 == 0:
            agent.update_target_model()
    env.close()
    agent.save("breakout.h5")
    agent.load("breakout.h5")
    state = env.reset()
    state = np.reshape(state, [1, 84, 84, 1])
    for time in range(1000):
        env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, 84, 84, 1])
        state = next_state
        if done:
            print("episode: {}, score: {}, e: {:.2}"
                  .format(time, time, agent.epsilon))
            break
    env.close()




