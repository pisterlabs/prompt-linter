import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="ML Tutor: I am a ML/AI language model tutor\nYou: What is a language model?\nML Tutor: A language model is a statistical model that describes the probability of a word given the previous words.\nYou: What is a statistical model?",
  temperature=0.3,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0,
  stop=["You:"]
)

# Prompt
# ML Tutor: I am a ML/AI language model tutor
# You: What is a language model?
# ML Tutor: A language model is a statistical model that describes the probability of a word given the previous words.
# You: What is a statistical model?
# Sample response
# ML Tutor: A statistical model is a mathematical representation of a real-world phenomenon. It is used to make predictions or decisions based on data. Statistical models use probability and statistics to make inferences about the data.
