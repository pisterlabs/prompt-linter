import os
import openai


def feedback(conversation_history):
  conversation_history.append({"role": "system", "content": 
                               """
                               Critically read through the conversation and provide a feedback to improve the discussion
                               """
                               })
  openai.api_key = os.environ["OPENAI_API_KEY"]

  response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history,
        temperature=0.9,
        max_tokens=128
  )

  return response.choices[-1].message.content

