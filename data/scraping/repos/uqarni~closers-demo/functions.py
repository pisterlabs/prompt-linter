import openai
import os
import re
import random
from datetime import datetime, timedelta
import random

#generate random date
def random_date():
  start_date = datetime(2022, 1, 1)
  end_date = datetime.now()
  random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
  
  formatted_date = random_date.strftime("%Y-%m-%d")
  return formatted_date  # Outputs the random date in "YYYY-MM-DD" format



#generate openai response; returns messages with openai response
def ideator(messages, temperature):

  key = os.environ.get("OPENAI_API_KEY")
  openai.api_key = key

  result = openai.ChatCompletion.create(
    model="gpt-4",
    messages= messages,
    temperature = float(temperature)
  )
  response = result["choices"][0]["message"]["content"]
  
  def split_sms(message):
      import re
  
      # Use regular expressions to split the string at ., !, or ? followed by a space or newline
      sentences = re.split('(?<=[.!?]) (?=\\S)|(?<=[.!?])\n', message.strip())
      # Strip leading and trailing whitespace from each sentence
      sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
  
      # Compute the cumulative length of all sentences
      cum_length = [0]
      for sentence in sentences:
          cum_length.append(cum_length[-1] + len(sentence))
      
      total_length = cum_length[-1]
  
      # Find the splitting point
      split_point = next(i for i, cum_len in enumerate(cum_length) if cum_len >= total_length / 2)
  
      # Split the sentences into two parts at the splitting point
      part1 = sentences[:split_point]
      part2 = sentences[split_point:]
  
      # Join the sentences in each part back into strings and exclude any part that is empty
      strings = []
      if part1:
          strings.append(" ".join(part1))
      if part2:
          strings.append(" ".join(part2))
      
      return strings

  split_response = split_sms(response)
  count = len(split_response)
  for section in split_response:
    section = {
      "role": "assistant", 
      "content": section
    }
    messages.append(section)

  return messages, count



def secret_message(followup_message):

    return {"role": "user", "content": followup_message}
