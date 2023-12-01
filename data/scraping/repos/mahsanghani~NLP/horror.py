import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Topic: Breakfast\nTwo-Sentence Horror Story: He always stops crying when I pour the milk on his cereal. I just have to remember not to let him see his face on the carton.\n    \nTopic: Wind\nTwo-Sentence Horror Story:",
  temperature=0.8,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.5,
  presence_penalty=0.0
)

# Prompt
# Topic: Breakfast
# Two-Sentence Horror Story: He always stops crying when I pour the milk on his cereal. I just have to remember not to let him see his face on the carton.
    
# Topic: Wind
# Two-Sentence Horror Story:
# Sample response
# The wind howled through the night, shaking the windows of the house with a sinister force. As I stepped outside, I could feel it calling out to me, beckoning me to follow its chilling path.