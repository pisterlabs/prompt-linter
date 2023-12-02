


import nextcord
import openai


def setup(bot):
    @bot.slash_command()
    async def generate_legal_document(interaction: nextcord.Interaction):
        """
        Generate legal documents using the Davinci 003 engine. Not true legal advice!
        """
        await interaction.response.defer()
        await interaction.followup.send("Please provide the necessary information for the legal document:", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        legal_info = response.content
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Given the following legal information:\n{legal_info}\nGenerate the legal document:",
            max_tokens=2048,
            temperature=0.7,
        )
        document = response.choices[0].text.strip()
        embed = nextcord.Embed(title="Generated Legal Document", description=document, color=0x00ff00)
        await interaction.followup.send(embed=embed)    