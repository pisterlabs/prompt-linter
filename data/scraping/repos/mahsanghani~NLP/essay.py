import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Create an outline for an essay about Nikola Tesla and his contributions to technology:",
  temperature=0.3,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Create an outline for an essay about Nikola Tesla and his contributions to technology:
# Sample response
# I. Introduction
# A. Who was Nikola Tesla?
# B. Overview of his contributions to technology

# II. Early Life and Education
# A. Where and when Tesla was born
# B. His education and early career

# III. Tesla's Contributions to Technology
# A. Alternating Current
# B. Radio and Remote Control
# C. Robotics
# D. X-Rays

# IV. Tesla's Later Years
# A. His later inventions
# B. His death and legacy

# V. Conclusion
# A. Summary of Tesla's life and accomplishments
# B. Impact of Tesla's work on modern technology