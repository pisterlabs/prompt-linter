"""
Con este modelo se usa GPT-3 para generar un emoji a partir de un tweet (texto).
"""

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

restart_sequence = "###\n"

response = openai.Completion.create(
    engine="davinci",
    prompt='This is a tweet to emoji converter.\n\nTweet: "I loved the new Batman movie!"\nEmoji: 😍🦇\n###\nTweet: "I hate it when my phone battery dies."\nEmoji: 🤬📵\n###\nTweet: "Tengo mucha hambre, me comería un caballo entero"\nEmoji: 😋🐎\n###\nTweet: "This is the link to the article"\nEmoji: 📲📄\n###\nTweet: "This new music video blew my mind"\nEmoji: 😮🎥\n###\nTweet: "And the stars look very different today."\nEmoji: 🌌🌠\n###\nTweet: "Coconut water taste like it’s been in someone else’s mouth"\nEmoji: 🤢🚰🥥\n###\nTweet: "I’m going to the gym to work out"\nEmoji: 🏋🏻💪\n###\nTweet: "En unas horas viajo de nuevo a Polonia con mis educandos."\nEmoji: 📖🚂🎒\n###\nTweet: "I’m going to miss this place."\nEmoji: 💔📷\n###\n',
    temperature=0.41,
    max_tokens=82,
    top_p=1,
    frequency_penalty=0.5,
    presence_penalty=0,
    stop=["###"],
)
