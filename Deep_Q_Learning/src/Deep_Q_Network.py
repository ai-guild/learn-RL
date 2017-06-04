import tensorflow as tf
import numpy as np
import random
import cv2
import tflearn
import pickle
import matplotlib.pyplot as plt
from collections import deque


ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

class Deep_Q_Network:
    def __init__(self, actions):
        #agent actions
        self.actions = actions
        #network weight params
        self.weights = {
            'w_conv1': tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01)),
            'w_conv2': tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01)),
            'w_conv3': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01)),
            'w_fc1': tf.Variable(tf.truncated_normal([3136, 512], stddev=0.01)),
            'w_fc2': tf.Variable(tf.truncated_normal([515, self.actions], stddev=0.01))
        }
        #network bias params
        self.biases = {
            'b_conv1': tf.Variable(tf.constant(0.01, [32])),
            'b_conv2': tf.Variable(tf.constant(0.01, [64])),
            'b_conv3': tf.Variable(tf.constant(0.01, [64])),
            'b_fc1': tf.Variable(tf.constant(0.01, [515])),
            'b_fc2': tf.Variable(tf.constant(0.01, [self.actions]))
        }
        self.state_input,self.Qvalue, self.biases,self.weights = self.buildNetwork()
        #initialize our training method
        self.init_training()
        self.observation = deque()
        #perform no action at the first state
        self.no_action = np.zeros(self.actions)
        self.no_action[0] = 1

    #create the convolution_NN for processing screen pixels
    def buildNetwork(self):
        state_input = tf.placeholder("float", [None, 84,84,4])
        conv1 = self.convolution_layer2D(state_input, self.weights['w_conv1'], self.biases['b_conv1'],strides=4)
        conv2 = self.convolution_layer2D(conv1, self.weights['w_conv2'], self.biases['b_conv2'],strides=2)
        conv3 = self.convolution_layer2D(conv2, self.weights['w_conv3'], self.biases['b_conv3'], strides= 1)
        conv3_flatten = tf.reshape(conv3,[-1, 3136])
        fc1 = tf.nn.relu(tf.add(tf.matmul(conv3_flatten, self.weights['w_fc1']),self.biases['b_fc1'], name="full_connected"))
        #Read out Q-Value layer
        Q_Value = tf.add(tf.matmul(fc1, self.weights['w_fc2']),self.biases['b_fc2'],name="Q_Value")

        return state_input,Q_Value,fc1

    #Start training our agent
    def init_training(self):
        #Agent actions as input to our graph
        self.action_input = tf.placeholder("float",[None,self.actions])
        #predicted actions
        self.pred_y = tf.placeholder("float",[None])
        Q_Action = tf.reduce_sum(tf.matmul(self.Qvalue, self.action_input), reduction_indices= 1)
        self.loss = tf.reduce_mean(tf.square(self.pred_y - Q_Action))
        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

        #saving and load tensorflow model
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()#
        self.session.run(tf.global_variables_initializer)
        self.check_point = tf.train.get_checkpoint_state("saved_weight")
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, self.checkpoint.model_checkpoint_path)
            print("Successfully loaded:", self.checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def trainAgentNetwork(self, frame_step):


    def convolution_layer2D(self, x, Weight, bias, strides = 1):
        x = tf.nn.conv2d(x, Weight, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x,bias)
        return tf.nn.relu(x)


    def max_pool_2x2(self, x, kernel = 2):
        return tf.nn.max_pool(x, ksize=[1, kernel, kernel, 1], strides=[1, 2, 2, 1], padding='SAME')

