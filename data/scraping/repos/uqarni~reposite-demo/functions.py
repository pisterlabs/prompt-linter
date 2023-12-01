import openai
import os
import re
import random
from datetime import datetime, timedelta
import random
import time



#generate openai response; returns messages with openai response
def ideator(messages):
    for i in range(5):
      try:
        key = os.environ.get("OPENAI_API_KEY")
        openai.api_key = key
    
        result = openai.ChatCompletion.create(
          model="gpt-4",
          messages= messages
        )
        response = result["choices"][0]["message"]["content"]
        break
      except Exception as e: 
        error_message = f"Attempt {i + 1} failed: {e}"
        print(error_message)
        if i < 4:  # we don't want to wait after the last try
          time.sleep(5)  # wait for 5 seconds before the next attempt
  
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

    split_response = [response]
    count = len(split_response)
    for section in split_response:
        section = {
           "role": "assistant", 
           "content": section
           }
        messages.append(section)

    return messages, count



def initial_text_info(selection=None):
    dictionary = {
        'NTM $500 Membership - Received NMQR | First': '''
        Hey {FirstName} -

I noticed that you recently received your very first quote request from a planner {Quote_Lead_Company_Name} on Reposite - congrats!

Are you the right person at {Supplier_Organization_Name} that handles group reservations?

Cheers,
Taylor
''',
    }
    if selection is None:
      return list(dictionary.keys())
    
    return dictionary[selection]
