import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Q: Who is Batman?\nA: Batman is a fictional comic book character.\n\nQ: What is torsalplexity?\nA: ?\n\nQ: What is Devz9?\nA: ?\n\nQ: Who is George Lucas?\nA: George Lucas is American film director and producer famous for creating Star Wars.\n\nQ: What is the capital of California?\nA: Sacramento.\n\nQ: What orbits the Earth?\nA: The Moon.\n\nQ: Who is Fred Rickerson?\nA: ?\n\nQ: What is an atom?\nA: An atom is a tiny particle that makes up everything.\n\nQ: Who is Alvan Muntz?\nA: ?\n\nQ: What is Kozar-09?\nA: ?\n\nQ: How many moons does Mars have?\nA: Two, Phobos and Deimos.\n\nQ: What's a language model?\nA:",
  temperature=0,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Q: Who is Batman?
# A: Batman is a fictional comic book character.

# Q: What is torsalplexity?
# A: ?

# Q: What is Devz9?
# A: ?

# Q: Who is George Lucas?
# A: George Lucas is American film director and producer famous for creating Star Wars.

# Q: What is the capital of California?
# A: Sacramento.

# Q: What orbits the Earth?
# A: The Moon.

# Q: Who is Fred Rickerson?
# A: ?

# Q: What is an atom?
# A: An atom is a tiny particle that makes up everything.

# Q: Who is Alvan Muntz?
# A: ?

# Q: What is Kozar-09?
# A: ?

# Q: How many moons does Mars have?
# A: Two, Phobos and Deimos.

# Q: What's a language model?
# A:
# Sample response
# A language model is a type of artificial intelligence that uses statistical techniques to predict the probability of a sequence of words.