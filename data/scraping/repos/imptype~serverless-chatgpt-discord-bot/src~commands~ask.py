import os
import discohook
import openai
from ..utils.helpers import chunks

openai.api_key = os.getenv('OPENAI_API_KEY')
system_content = os.getenv('CHATGPT_SYSTEM_MESSAGE')

models = ['gpt-3.5-turbo-0301', 'gpt-3.5-turbo']

options = [
  discohook.StringOption('prompt', 'Type your question here.', required = True),
  discohook.StringOption('model', 'The AI model to use.', choices = [
    discohook.Choice(model.replace('.', '-'), model) for model in models # . not allowed in name
  ])
]

@discohook.command('ask', 'Ask ChatGPT anything!', options = options)
async def ask_command(interaction, prompt, model):

  await interaction.defer() # because it takes more than 3 seconds!

  if (await openai.Moderation.acreate(prompt)).results[0].flagged:
    return await interaction.response('Sorry, your prompt was flagged.')

  if not model:
    model = models[-1] 

  completion = await openai.ChatCompletion.acreate(
    model = model, 
    messages = [
      {'role': 'system', 'content' : system_content},
      {'role': 'user', 'content': prompt}
    ]
  )
  
  text = completion.choices[0].message.content

  for text in chunks(text, 2000)[:10]:
    await interaction.followup(text)
