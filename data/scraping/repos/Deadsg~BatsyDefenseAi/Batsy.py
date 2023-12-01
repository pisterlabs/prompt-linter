import os
import discord
from discord.ext import commands
import openai
import gym
import numpy as np
from sklearn.linear_model import LinearRegression

# Set up your OpenAI API key
openai.api_key = "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"

# Initialize the Discord bot
intents = discord.Intents.default()
intents.typing = False
intents.presences = False
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize Gym environment and create a simple Q-learning agent
env = gym.make('CartPole-v1')
Q = np.zeros((env.observation_space.shape[0], env.action_space.n))

@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.content.startswith('!train_model'):
        # Train a simple Linear Regression model
        X = np.array([[1], [2], [3], [4]])
        y = np.array([3, 4, 2, 5])
        model = LinearRegression()
        model.fit(X, y)
        await message.channel.send(f'Model trained. Coefficient: {model.coef_}, Intercept: {model.intercept_}')

    if message.content.startswith('!run_gym'):
        # Run a simple Q-learning agent in a Gym environment
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = np.argmax(Q[state])
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            state = next_state

        await message.channel.send(f'Total reward: {total_reward}')

# Run the bot
bot.run('MTE0NjkwNDk2Nzc2NTA1MzQ2MA.GXK8U1.wnakgQpSoClJwjrNnlFNwAXCIVzovYwCyDvfU8 ')  # Replace with your bot toke

   