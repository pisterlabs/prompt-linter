


import asyncio
from difflib import SequenceMatcher
import nextcord
import openai


def setup(bot):
    @bot.slash_command(name="fillintheblank", description="Play a fill in the blank game in the target language.")
    async def fill_in_the_blank(interaction: nextcord.Interaction, target_language: str):
        # Generate a sentence with a blank in the target language
        prompt = f"Generate a sentence with a blank in {target_language}:"
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates fill-in-the-blank sentences in different languages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.8,
        )
        sentence = completion.choices[0].message["content"].strip()

        # Get the answer to the blank
        prompt = f"What is the missing word in the following sentence: '{sentence}'?"
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides answers to fill-in-the-blank sentences in different languages."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.8,
        )
        answer = completion.choices[0].message["content"].strip()

        # Send the sentence with a blank to the user
        await interaction.channel.send(f"💬 **Fill in the Blank** 💬\n\nComplete the following {target_language} sentence:\n\n**{sentence}**")

        # Check the user's answer and provide feedback
        def check(m):
            return m.author.id == interaction.user.id and m.channel.id == interaction.channel.id

        try:
            user_answer = await bot.wait_for("message", check=check, timeout=100)
            similarity = SequenceMatcher(None, user_answer.content.lower(), answer.lower()).ratio()
            if similarity > 0.75:  # Set the similarity threshold (0.75 means 75% similarity)
                await interaction.channel.send("🎉 Correct! Good job! 🎉")
            else:
                await interaction.channel.send(f"Almost! The correct missing word is: **{answer}**")
        except asyncio.TimeoutError:
            await interaction.channel.send("⌛ Time's up! Please try again.")