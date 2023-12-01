


import nextcord
import openai


def setup(bot):
    @bot.slash_command()
    async def generate_lyrics(interaction: nextcord.Interaction):
        """
        Generate song lyrics using the Davinci 003 engine.
        """
        await interaction.response.defer()
        await interaction.followup.send("What's the song prompt?", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        prompt = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Write lyrics for a song based on the following prompt:\n{prompt}\nLyrics:",
            max_tokens=1024,
            temperature=0.7,
        )
        lyrics = response.choices[0].text.strip()
        await interaction.followup.send(f"Here are your lyrics:\n```{lyrics}```")