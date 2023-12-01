from matplotlib import pyplot as plt

import OpenAiGymEnvironment
import DQNAgent
import torch
import numpy as np
import math

class BermudanOption:
    def __init__(self):
        self.env = OpenAiGymEnvironment.AmeriOptionEnv()

    def getScore(self, netParams):
        observation_space = self.env.observation_space
        action_space = 2
        exTimes = [153, 195, 94, 338, 76, 310, 42, 224, 200, 213]  # giorni dell'anno in cui sarà possibile esercitare la bermudan option
        S0 = 100.0
        K = 110.0
        r = 0.1
        T = 1.0
        # Create the PyTorch model.
        hidden_layer_weights = netParams.previous_layer.trained_weights
        hidden_layer_weights = hidden_layer_weights.copy()
        output_layer_weights = netParams.trained_weights
        output_layer_weights = output_layer_weights.copy()
        input_layer = torch.nn.Linear(4, 4)
        relu_layer = torch.nn.ReLU()
        hidden_layer = torch.nn.Linear(4, 4)
        output_layer = torch.nn.Linear(4, 1)
        softmax_layer = torch.nn.Softmax(dim=1)

        model = torch.nn.Sequential(input_layer,
                                    relu_layer,
                                    hidden_layer,
                                    relu_layer,
                                    output_layer,
                                    softmax_layer)

        for name, param in model.named_parameters():
            #setto i pesi solo per l'hidden layer e l'output layer
            if name.startswith("2."):
                param.data = torch.from_numpy(hidden_layer_weights) #converto da ndarray a pytorch tensor
            if name.startswith("4."):
                param.data = torch.from_numpy(output_layer_weights) #converto da ndarray a pytorch tensor

        gamma = math.exp(-r * T/365)
        agent = DQNAgent.DQNAgent(gammma, input_dims, batch_size, n_actions, 100000)

        total_rewards = []
        self.env.reset()
        simulated_prices = self.env.simulateUnderlyingPrice(S0, r, T, 0.2, 20, 10)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.plot(simulated_prices)
        plt.show()

        total_reward = 0
        steps = 0
        while True:
            action = agent.act(state, steps, sim_prices)
            steps += 1

            state_next, reward, terminal, info = self.env.step(action)
            total_reward += reward
            state_next_numpy_array = np.array([state_next])
            state_next = torch.Tensor(state_next_numpy_array)
            reward = torch.tensor([reward]).unsqueeze(0)

            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            state = state_next
            if terminal:
                break

        total_rewards.append(total_reward)

        self.env.close()
        return total_rewards[-1] #viene restituita una tupla, ma devo ritornare solo un numero
