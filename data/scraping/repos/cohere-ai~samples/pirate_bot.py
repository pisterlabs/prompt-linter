import discord
from discord.ext import commands
import cohere
from cohere.classify import Example
import numpy as np
import csv
import ast
import yaml
import random
import numpy as np
import sys

with open('config.yml') as file:
    config = yaml.load(file)
yaml.dump(config, open('config.yml', "w"))

client = commands.Bot(command_prefix=config['command-prefix'])


class pirateClassifier():

    def __init__(self, cohere_key):
        self.examples = [["Aaaarrrrgggghhhh!", "pirate"], ["Bring a Spring Upon ‘er	", "pirate"],
                         ["Cleave Him to the Brisket", "pirate"], ["i think i saw it first", "normal"],
                         ["send him to Davy Jones' Locker", "pirate"], ['The crew is fierce famished', 'pirate'],
                         ["arrg i think i saw it first", "pirate"], ["hello how are you?", "normal"],
                         ["Run a Shot Across the Bow", "pirate"], ["send him to jail", "normal"],
                         ["i am gonna go to the office", "normal"], ['avast ye! i told ye before!', "pirate"],
                         ["I am a pirate!", "pirate"], ["i will honnestly kill you", "normal"],
                         ["hello there scallywags!", "pirate"], ["THATS SO COOL!!", "normal"],
                         ['there she blows!', 'pirate']]
        self.options = ['pirate', 'normal']
        self.co = cohere.Client(cohere_key)

    def cohere_classify(self, inputs):
        random.shuffle(self.examples)
        classifications = self.co.classify(
            model='medium',
            taskDescription='',
            outputIndicator='',
            inputs=inputs,
            examples=[Example(question, clasification) for question, clasification in self.examples])
        results = []
        predictions = []
        for classification in classifications.classifications:
            _results = {}
            predictions.append(classification.prediction)
            for c in classification.confidence:
                _results[c.label] = c.confidence
            results.append(_results)
        return predictions, results


class pirateRephraser():

    def __init__(self, cohere_key):
        self.co = cohere.Client(cohere_key)
        self.examples = [["aaaaa!!!", "Aaaarrgggghhh!"], ["get ready!", "Batten Down The Hatches"],
                         ["Knock him over!", "Cleave Him to the Brisket"],
                         ["hey whats the weather like today?", "aye how be the sea today?"],
                         ['whats your dream in life', 'Ay, what do ye wish in all the seven seas?'],
                         ["kill him!", "send him to Davy Jones' Locker"],
                         ["i am starving", 'The crew is fierce famished'],
                         ['is this your first time?', 'Are ye a yellow belly beginer?'],
                         ['I sit at my office every day', 'i be working the ship from dawn till dusk'],
                         ["I saw it first", "arrgrrg!! i think i saw it first"]]
        self.prompts = ['normal text', 'rephrased as a pirate']
        self.starting_prompt = 'These are examples of setences rephrased to be in pirate speech:'

    def rephrase(self, msg):
        prompt = self.starting_prompt + "\n" + "\n\n".join([
            self.prompts[0] + ":" + e[0] + "\n" + self.prompts[1] + ":" + e[1] for e in self.examples
        ]) + "\n\n" + self.prompts[0] + ":" + msg + "\n" + self.prompts[1] + ":"
        print(prompt)
        prediction = self.co.generate(model='large',
                                      prompt=prompt,
                                      max_tokens=50,
                                      stop_sequences=["\n"],
                                      num_generations=4,
                                      temperature=1.2).generations
        return (prediction)


print("sys.argv[2]")
print(sys.argv)
co_toxicity = pirateClassifier(sys.argv[2])
co_rephraser = pirateRephraser(sys.argv[2])
co = cohere.Client(sys.argv[2])


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))
    print(str(client.user) + 'is connected to the following guild:\n')
    for guild in client.guilds:
        print(guild.name)


@client.command('rate_toxicity')
async def rate_toxicity(ctx, *, message: str = ''):
    _, rating = co_toxicity.cohere_classify([message])
    rating = rating[0]
    await ctx.channel.send("toxicity_rating " + "{:.3f}".format(rating['toxic']) + "%")


@client.command('list_training_data')
async def list_training_data(ctx):
    for i in range(len(co_toxicity.examples)):
        await ctx.channel.send(str(i) + "\t" + co_toxicity.examples[i][0] + "\t" + co_toxicity.examples[i][1])


@client.command('rm_training_example')
async def rm_training_example(ctx, i: int):
    co_toxicity.examples.pop(i)
    await ctx.channel.send("deleted")


@client.command('add_toxic_training_example')
async def add_toxic_training_example(ctx, *, content: str = ''):
    co_toxicity.examples.append([content, 'toxic'])
    await ctx.channel.send("added")


@client.command('add_benign_training_example')
async def add_benign_training_example(ctx, *, content: str = ''):
    co_toxicity.examples.append([content, 'benign'])
    await ctx.channel.send("added")


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


@client.event
async def on_message(message):
    pirate_channel = discord.utils.get(message.guild.channels, name="pirates-only")
    print(message.content)
    print(message.channel.name)
    if message.channel.name == 'pirates-only':
        if message.content and message.content[0] == config['command-prefix']:
            await client.process_commands(message)
            return

        if message.author == client.user:
            return
            

        print("classiying")
        classification, rating = co_toxicity.cohere_classify([message.content])
        classification = classification[0]
        rating = rating[0]
        print(classification, rating)
        if classification != 'pirate':
            await message.add_reaction('☠️')
            generatsions = [g.text for g in co_rephraser.rephrase(message.content)]
            print("what", generatsions)
            classifications, ratings = co_toxicity.cohere_classify(generatsions)
            piratness = [r['pirate'] for r in ratings]
            acceptable_generations = []
            for i, p in enumerate(piratness):
                print(float(p))
                if float(p) >= np.mean(piratness) and not generatsions[i].isspace():
                    acceptable_generations.append(generatsions[i])
            embeddings = co.embed(model='small', texts=acceptable_generations + [message.content]).embeddings
            gen_embeddings = embeddings[:-1]
            msg_embedding = embeddings[-1]
            distances = [cosine_similarity(x, msg_embedding) for x in gen_embeddings]
            print(distances)
            print(acceptable_generations)
            max_i = np.argmax(distances)
            await message.channel.send("Arrg! Thats not pirate enough! did ya mean:")
            await message.channel.send('`' + acceptable_generations[max_i] + '`')
        else:
            await message.add_reaction('🦜')


client.run(sys.argv[1])