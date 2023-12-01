import openai
import os
from en_variable.set_env import set_env

set_env()

openai.api_key = os.environ.get("OPENAI_API_KEY")

import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
      {'role':'system','content':'You are a helper for extractaring keyword from riviews. Reviews will be for a turist spot. Your job is to understand the reviews and based on that suggest tags like "family friendly", "good for couples", "good for solo", "good for groups", "good for kids", "good for pets", "good for hiking", "good for swimming", "good for camping", "good for fishing", "good for boating", "good for kayaking", "good for canoeing", "good for biking", "good for running", "good for walking", "good for picnics", "good for bird watching", "good for wildlife watching", "good for photography", "good for sunsets", "good for sunrises", "good for stargazing", "good for fall foliage", "good for winter sports", "good for spring flowers", "good for summer flowers", "good for fall foliage", "good for winter sports", "good for spring flowers", "good for summer flowers", "good for fall foliage", "good for winter sports", "good for spring flowers", "good for summer flowers"'},
      {'role':'user',
       'content':'This place is a must visit for both foreigners and Bangladeshi. Offers detailed information about the military condition and history of Bangladesh. Extremely well maintained, neat and clean having a tight security. You need minimum 2 hours to explore the museum properly. Closes at dot 6pm. But the cafeteria and 3D theatre remains open.\nWeekly holiday - Wednesday'}
  ],
  temperature=0.5,
  max_tokens=256,
)
for choice in response.choices:
    print(choice.message['content'])