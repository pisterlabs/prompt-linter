import discord
from discord import app_commands
import uni_token
import openai

API = uni_token.OPENAITOKEN()
openai.api_key = API

# Aos poucos, este virará o arquivo main.py, e será como que o "cérebro" principal do bot

def generate_response(message):
    prompt = message
    try: 
        response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500,
        top_p=1,
        stop=None,
        temperature=0.7,
        )
        return response.choices[0].text
    except Exception as e:
        return 'Ocorreu um erro ao gerar a resposta'


class aclient(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.synced = False

    async def on_ready(self):
        await self.wait_until_ready()
        if not self.synced:
            await tree.sync()
            self.synced = True
        print('We have logged in as {}'.format(self.user))

client = aclient()
tree = app_commands.CommandTree(client)

@tree.command(name='ping', description='Retorna "pong"')
async def self(interaction: discord.Interaction):
    await interaction.response.send_message('Pong!')

@tree.command(name='gpt', description='Chama a API da OpenAI para conversar com Inteligência Artificial')
async def self(interaction: discord.Interaction, prompt: str):
    awnser = generate_response(prompt)
    await interaction.response.send_message(f'Usuário disse: {prompt}\nUni: ```{awnser}```')

@tree.command(name='resumo', description='Usa inteligência artificial para resumir um texto')
async def self(interaction: discord.Interaction, texto: str):
    awnser = generate_response(f'Resuma o seguinte texto: "{texto}"')
    await interaction.response.send_message(f'{awnser}')

client.run(uni_token.UNITOKEN())