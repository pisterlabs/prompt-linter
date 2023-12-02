import json
import nextcord
from nextcord.ext import commands
import openai
import asyncio

class EightBall(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.prompt = ""

        # Load prompt from the .json file
        with open('json/eightball_prompt.json', 'r') as file:
            self.prompt = json.load(file)['prompt']

    @commands.Cog.listener()
    async def on_ready(self):
        print("Magic balls shaken")

    def process_question(self, question):
        # Add "?" at the end of question if it doesn't exist
        return question if question.endswith("?") else question + "?"

    @nextcord.slash_command(description="Ask the Magic 8-Ball a question")
    async def magic8ball(self, interaction: nextcord.Interaction, question: str):
        try:
            await interaction.response.defer()
        except nextcord.NotFound:
            return

        # Process the question
        processed_question = self.process_question(question)

        # Generate a response using the GPT-3 model
        for attempt in range(3):  # Try to generate a response 3 times
            try:
                response = await asyncio.to_thread(
                    openai.Completion.create,
                    engine="text-davinci-003",
                    prompt=self.prompt + processed_question,
                    max_tokens=60,
                    temperature=0.8  # Set the temperature to 0.8
                )
                break
            except openai.OpenAIError:
                if attempt == 2:  # If it's the last attempt
                    await interaction.followup.send("Sorry, I'm having some issues generating a response. Please try again later.")
                    return
                else:
                    await asyncio.sleep(1)

        response_text = response.choices[0].text.strip()

        # Create an embed to send as a response
        embed = nextcord.Embed(
            title="Magic 8-Ball",
            color=nextcord.Color.blue()
        )
        embed.add_field(name="Question", value=processed_question, inline=False)
        embed.add_field(name="Answer", value=response_text, inline=False)
        embed.set_thumbnail(url="https://gateway.ipfs.io/ipfs/QmeQZvBhbZ1umA4muDzUGfLNQfnJmmTVsW3uRGJSXxXWXK")

        # Send the embed as a response to the interaction
        await self.send_message_safe(interaction.followup.send(embed=embed))

    async def send_message_safe(self, action):
        try:
            return await action
        except nextcord.errors.NotFound:
            pass

def setup(bot):
    bot.add_cog(EightBall(bot))  # Add the EightBall cog to the bot

