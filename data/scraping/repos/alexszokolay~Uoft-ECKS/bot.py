import discord
from discord.ui import Button, View
from discord.ext import commands
import json
from os import environ
from dotenv import load_dotenv
import clean_file_with_algorithm
import cohere
import clean_file_with_algorithm as algorithm


load_dotenv()
token = environ["TOKEN"]

intents = discord.Intents.all()

client = commands.Bot(command_prefix='!', intents=intents)

options = {"1": '😄', "2": '🙂', "3": '😐', "4": '🙁', "5": '😢'}
colours = {"1": discord.ButtonStyle.green, "2": discord.ButtonStyle.blurple,
           "3": discord.ButtonStyle.grey, "4": discord.ButtonStyle.blurple,
           "5": discord.ButtonStyle.red}

emotions = {"Stress": 0, "Boredom": 0, "Loneliness": 0, "Anger": 0,
            "Sadness": 0}

answers = []

emotions_streak = []


@client.event
async def on_ready():
    print('bot ready')


class MyButton(Button):
    def __init__(self, name, color, emoji, emotion):
        super().__init__(label=name, style=color, emoji=emoji)
        self.emotion = emotion

    async def callback(self, interaction: discord.Interaction):
        # ! Update the dictionary
        if ',' in self.emotion:
            emotion1 = self.emotion.split(',')[0].strip()
            emotion2 = self.emotion.split(',')[1].strip()
        else:
            emotion1 = emotion2 = self.emotion
        for key in emotions:
            if key == self.emotion or key == emotion1 or key == emotion2:
                emotions[key] += int(self.label)
        await interaction.response.send_message(
            "Thanks! Your input of " + self.label + " has been recorded.")
        answers.append(int(self.label))


class MyView(View):
    emotions = emotions

    def __init__(self):
        super().__init__()  # timeout=10.0)

    async def button_callback(self, interaction: discord.Interaction):
        await interaction.response.edit_message(view=self)


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # !Help command
    if message.content.startswith('!help'):
        embed_var = discord.Embed(title="Help",
                                  color=discord.Colour.light_grey())
        embed_var.add_field(name="About",
                            value="VentBot is a tool that helps you destress!",
                            inline=True)
        embed_var.add_field(name="Usage", value="!vent", inline=False)
        await message.channel.send(embed=embed_var)

    # ! Main command
    if message.content == '!vent':
        await message.channel.send(
            "Hey there, I heard you weren't feeling so great! ")
        await message.channel.send("Here are some questions:")

        f = open("questions.txt", "r")

        for line in f:
            view = MyView()
            current_emotion = line.split("(")[1].split(")")[0]
            for index in options:
                button = MyButton(index, colours[index],
                                  options[index], current_emotion)
                view.add_item(button)

            await message.channel.send(line.split("(")[0], view=view)
            await client.wait_for('message')

        f.close()

        therapy_bot = clean_file_with_algorithm.TherapyTravel(emotions)
        interests = clean_file_with_algorithm.InterestsList()
        output = clean_file_with_algorithm.emotion_giving_method(therapy_bot,
                                                                   interests)

        co = cohere.Client("MSuvC3ORXmJeWIzxj6D9vIw0QZAhfO6ibEmTlDYG")
        prompt = f"\n Name real locations in Toronto I should go to if I like "\
                 f"{output}"

        # moddel = medium or xlarge
        response = co.generate(
            model='c1e4d1a2-5127-494b-8536-3d6845a4f267-ft',
            prompt=prompt,
            max_tokens=35,
            temperature=0.9,
            stop_sequences=["--"]
        )

        result = response.generations[0].text

        embed_var = discord.Embed(title="Here is your custom suggestion!",
                                  color=discord.Colour.light_grey())
        embed_var.add_field(name="Interests",
                            value=output,
                            inline=True)
        embed_var.add_field(name="Suggestion", value=result, inline=False)
        await message.channel.send(embed=embed_var)



client.run(token)
