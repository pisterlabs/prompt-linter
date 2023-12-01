from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Environment Variables
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
PODCASTS_PATH = os.environ.get("PODCASTS_PATH")
SENDGRID_KEY = os.environ.get("SENDGRID_KEY")
SENDGRID_KEY = os.environ['SENDGRID_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY']
AWS_SECRET_KEY = os.environ['AWS_SECRET_KEY']
REGION_NAME = os.environ['REGION_NAME']
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY']

# Assurez-vous d'avoir défini votre clé API comme variable d'environnement
api_key = ANTHROPIC_API_KEY

"""
testanthropic = Anthropic()


completion = testanthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=f"{HUMAN_PROMPT} How many toes do dogs have?{AI_PROMPT}",
)


print(completion.completion)
"""


def generate_chat_completion_anthropic(consigne, texte, model="claude-2"):
    # Construct the prompt from the given consigne and texte
    prompt = f"{HUMAN_PROMPT} {consigne} : {texte}{AI_PROMPT}"

    # Create an Anthropic client
    client = Anthropic()

    # Create a stream completion using the Anthropic API
    stream = client.completions.create(
        prompt=prompt,
        model=model,
        stream=True,
        # Set any other desired parameters here, for example:
        max_tokens_to_sample=99000
    )

    # Iterate over the stream completions and yield the results
    for completion in stream:
        yield completion.completion

# Example usage
"""consigne = "Expliquez-moi"
texte = "comment fonctionne une boucle for en Python?"
for response in generate_chat_completion_anthropic(consigne, texte):
    print(response, end="", flush=True)
"""

