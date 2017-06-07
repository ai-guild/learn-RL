import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from tensorflow.contrib import layers
import pandas as pd

''''
Policy Gradient Method. 
'''

input_dim = 4
gamma = 0.99
num_hidden = 10
class PolicyGradient:
    def __init__(self, num_hidden = 10, learning_rate = 1e-2):
        self.gamma = gamma
        self.batch_size = 5
        self.input_dim = input_dim
        self.running_reward = None
        self.reward_sum = 0
        self.episode_number = 1
        self.total_episodes = 10000
        self.rendering = True
        tf.reset_default_graph()
        self.states = tf.placeholder(tf.float32, [None,input_dim], name="env_state")
        self.weight_1 = tf.get_variable("Weight_1", shape=[input_dim, num_hidden],initializer=layers.xavier_initializer())
        self.layer_1 = tf.nn.relu(tf.matmul(self.states, self.weight_1))
        weight_2 = tf.get_variable("Weight_2", shape=[num_hidden, 1],initializer=layers.xavier_initializer())
        self.prediction = tf.nn.softmax(tf.matmul(self.layer_1,weight_2,name="score"))
        self.train_variables = tf.trainable_variables()#differentiable variables for the gradient
        self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")
        self.reward_signal = tf.placeholder(dtype=tf.float32, shape=None, name="reward")

        #calculate the log-likelihood of our action
        self.log_likelihood = tf.log(self.actions*(self.actions-self.prediction)+(1-self.actions)*(self.actions+self.prediction))
        self.loss = -tf.reduce_mean(self.log_likelihood*self.reward_signal)
        #perform partial differentiations on the loss, using all trainable variables
        self.first_gradient = tf.gradients(self.loss,self.train_variables)

        self.optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)#Adam optimizer
        self.weight_1_gradient = tf.placeholder(dtype=tf.float32,name="first_BG")
        self.weight_2_gradient = tf.placeholder(dtype=tf.float32, name="second_BG")
        self.batch_gradient = [self.weight_1_gradient, self.weight_2_gradient]
        #Apply adam gradient decent method to optimize, using our trainable variables
        self.update = self.optimizer.apply_gradients(zip(self.batch_gradient, self.train_variables))
        self.sess = tf.Session();
        self.sess.run(tf.global_variables_initializer())
        print("Started")
        self.train()

    def getReward(self, reward):
        discounted_reward = np.zeros_like(reward)
        running = 0
        for time_step in reversed(range(0, reward.size)):
           running = running*gamma + reward[time_step]
           discounted_reward[time_step] = running
        return discounted_reward

    def train(self):
        xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
        env = gym.make('CartPole-v0')
        #Get the first state from the environment
        state = env.reset()

        # Reset the gradient placeholder. We will collect gradients in
        # gradBuffer until we are ready to update our policy network.9
        grad_buffer = self.sess.run(self.train_variables)
        for ix, gradient in enumerate(grad_buffer):
            grad_buffer[ix] = gradient * 0

        while self.episode_number <= self.total_episodes:

            #Only render when the agent reward passes some threshold
            if self.reward_sum / self.batch_size > 100 or self.rendering == True:
                env.render()
                self.rendering = True

            # Reshape the observation size to the shape of our input dimension to network
            x = np.reshape(state, [1, input_dim])
            # Run the policy network and get an action to take.
            prob = self.sess.run(self.prediction, feed_dict={self.states: x})
            action = 1 if np.random.uniform() < prob else 0

            #state
            xs.append(x)
            y = 1 if action == 0 else 0
            ys.append(y)

            # move steps in the environment to get new observations
            state, reward, done, info = env.step(action)
            self.reward_sum += reward#sum up all accumulated rewards from the environment

            drs.append(reward)

            if done:
                self.episode_number += 1
                # stack together all inputs, hidden states, action gradients, and rewards for this episode
                epx = np.vstack(xs)
                epy = np.vstack(ys)
                epr = np.vstack(drs)
                xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset array memory

                # compute the discounted reward backwards through time
                discounted_epr = self.getReward(epr)
                # size the rewards to be unit normal (helps control the gradient estimator variance)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr //= np.std(discounted_epr)

                # Get the gradient for this episode, and save it in the gradBuffer
                gradients = self.sess.run(self.first_gradient, feed_dict={self.states: epx, self.actions: epy, self.reward_signal: discounted_epr})
                for ix, grad in enumerate(gradients):
                    grad_buffer[ix] += grad

                # Start updating the policy gradient with the given episodes
                if self.episode_number % self.batch_size == 0:
                    self.sess.run(self.update, feed_dict={self.weight_1_gradient: grad_buffer[0], self.weight_2_gradient: grad_buffer[1]})
                    for ix, grad in enumerate(grad_buffer):
                        grad_buffer[ix] = grad * 0

                    self.running_reward = self.reward_sum if self.running_reward is None else self.running_reward * 0.99 + self.reward_sum * 0.01
                    print('Average reward for episode %f.  Total average reward %f.' % (
                         self.reward_sum / self.batch_size, self.running_reward / self.batch_size))

                    if self.reward_sum // self.batch_size > 200:
                        print("Task solved in", self.episode_number, 'episodes!')
                        break

                    self.reward_sum = 0

                state = env.reset()

        print(self.sepisode_number, 'Episodes completed.')

def main():
    agent = PolicyGradient()
    agent.train()
if __name__ == "__main__":
    main()