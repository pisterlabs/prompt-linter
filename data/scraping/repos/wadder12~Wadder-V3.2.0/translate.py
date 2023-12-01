import io
import nextcord
import openai
import requests
openai_model_engine = "text-davinci-003"

def setup(bot):
    
    
    @bot.slash_command()
    async def translate(interaction: nextcord.Interaction, source_language: str, target_language: str, text: str):
        """
        Translates text from one language to another using OpenAI.
        Usage: /translate2 <source_language> <target_language> <text to translate>
        """
        # Call the OpenAI API to translate the text
        completions = openai.Completion.create(
            engine=openai_model_engine,
            prompt=f"Translate from {source_language} to {target_language}: {text}",
            max_tokens=64,
            n=1,
            stop=None,
            temperature=0.5,
        )
        translated_text = completions.choices[0].text.strip()

        # Send the translated text back to the user
        await interaction.response.send_message(f"{translated_text}") 