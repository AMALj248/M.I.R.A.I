import keras
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.activations import sigmoid
import math
import numpy as np
import random
from collections import deque
import sys


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=''):
        self.state_size = state_size
        # Actions Buy, Sell, Hold
        self.action_size = 3
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # Loading the Saved model or load inbuilt model
        self.model = load_model(model_name) if is_eval else self._model()
        
    # The Neural Model
    
    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model
    
    # Action taking Module
    
    def act(self, state):

        # Take a random Action
        if not self.is_eval and random.random()<= self.epsilon:
            return random.randrange(self.action_size)
        options = self.model.predict(state)
        print('Options ->', options)
        return np.argmax(options[0])   # Choose the best Action

    # Function to reset memory if memory is full
    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)

        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])
        
        for state, action, reward, next_state, done in mini_batch:
          target = reward
          if not done:
              target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

          target_f = self.model.predict(state)
          target_f[0][action] = target
          self.model.fit(state, target_f, epochs = 1, verbose = 1)

        if self.epsilon > self.epsilon_min:
          self.epsilon *= self.epsilon_decay

    def formatPrice(self, n):
        return ("-Rs." if n < 0 else "Rs.") + "{0:.2f}".format(abs(n))

    # Function to recieve data
    def getStockDataVec(self):
        vec = []
        lines = open("HDFCBANK.NS.csv", "r").read().splitlines()
        for line in lines[1:]:
            #print(line)
            #print(float(line.split(",")[4]))
            # Extracting Closing Prices form the CSV
            vec.append(float(line.split(",")[4]))
            print(vec)
        return vec

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def getState(self, data, t, n):
        d = t - n + 1
        block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
        print('Block->\n', block)
        res = []
        for i in range(n - 1):
            res.append(sigmoid(block[i + 1] - block[i]))
        return np.array([res])





# Calling the model


window_size = 5
episode_count = 1
stock_name = 'HDFCBANK'



agent = Agent(window_size)
data = agent.getStockDataVec()
l = len(data) - 1
batch_size = 32
for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = agent.getState(data, 0, (window_size+1))
    total_profit = 0
    agent.inventory = []
    for t in range(l):
        action = agent.act(state)
        # sit
        next_state = agent.getState(data, t+1, (window_size + 1))
        reward = 0
        if action == 1: # buy
            agent.inventory.append(data[t])
            print("Buy: " + agent.formatPrice(data[t]))
        elif action == 2 and len(agent.inventory) > 0: # sell
            bought_price = window_size_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            print("Sell: " + agent.formatPrice(data[t]) + " | Profit: " + agent.formatPrice(data[t] - bought_price))
        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print("--------------------------------")
            print("Total Profit: " + agent.formatPrice(total_profit))
            print("--------------------------------")
        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)
    if e % 10 == 0:
        agent.model.save(str(e))

