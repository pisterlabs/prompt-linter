from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import anthropic
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Environment Variables
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
PODCASTS_PATH = os.environ.get("PODCASTS_PATH")
SENDGRID_KEY = os.environ.get("SENDGRID_KEY")
SENDGRID_KEY = os.environ['SENDGRID_KEY']
APP_PATH = os.environ['APP_PATH']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
AWS_SECRET_KEY = os.environ['AWS_SECRET_KEY']
REGION_NAME = os.environ['REGION_NAME']
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']

# Assurez-vous d'avoir défini votre clé API comme variable d'environnement
api_key = ANTHROPIC_API_KEY

anthropic = Anthropic()


completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=f"{HUMAN_PROMPT} How many toes do dogs have?{AI_PROMPT}",
)


print(completion.completion)

