import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Brainstorm some ideas combining VR and fitness:",
  temperature=0.6,
  max_tokens=150,
  top_p=1.0,
  frequency_penalty=1,
  presence_penalty=1
)

# Prompt
# Brainstorm some ideas combining VR and fitness:
# Sample response
# 1. Virtual reality-based fitness classes 
# 2. Home workout programs using virtual reality technology 
# 3. Interactive video game-style workouts with a virtual trainer 
# 4. Virtual running races against other players in VR worlds 
# 5. Immersive yoga and Pilates sessions set in exotic locations 
# 6. Sports simulations involving simulated physical activity within the VR world 
# 7. Group fitness challenges that involve competing against opponents in a virtual environment  
# 8. Adaptive exercise programs tailored to individual’s goals or health conditions