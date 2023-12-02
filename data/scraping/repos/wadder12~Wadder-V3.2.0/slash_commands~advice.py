

import nextcord
import openai


def setup(bot):
    @bot.slash_command()
    async def generate_financial_advice(interaction: nextcord.Interaction):
        """
        Generate personalized financial advice using the Davinci 003 engine. This is for fun not to be used for real life!
        """
        await interaction.response.defer()
        await interaction.followup.send("Please provide your financial data:", ephemeral=True)
        response = await bot.wait_for('message', check=lambda m: m.author == interaction.user)
        financial_data = response.content
        
        # Generate financial advice using OpenAI's API
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Given the following financial data:\n{financial_data}\nProvide personalized financial advice:",
            max_tokens=1024,
            temperature=0.7,
        )
        advice = response.choices[0].text.strip()
        
        # Send the financial advice in a nice embed
        embed = nextcord.Embed(
            title="Personalized Financial Advice",
            description=advice,
            color=nextcord.Color.blue()
        )
        embed.set_footer(text="This advice is for fun and should not be taken as professional financial advice.")
        await interaction.followup.send(embed=embed) 