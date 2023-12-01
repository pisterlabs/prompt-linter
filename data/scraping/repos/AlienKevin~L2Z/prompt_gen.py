import os
import openai
import time
import json
from tqdm import tqdm

# Create prompts for each new word.
words = []
with open("new.txt", "r") as fp:
    topic = "business"
    prompts = []
    for word in fp:
      words.append(word.strip("\n"))
      # base_prompt = f"Generate a {topic} conversation containing the word {word}."
      base_prompt = f"Generate a {topic} conversation between two people containing the word \"{word}\". Then, give the synonyms for the word and show what it means in Chinese.\n"
      prompts.append(base_prompt)


openai.api_key = os.getenv("OPENAI_API_KEY")

def getResponse(prompt):
  return openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.45, max_tokens=200, n=3)
'''
def getResponses(prompts):
  response_list = []
  for i in tqdm(range(0, len(prompts))):
    if i%20 == 0:
      time.sleep(65)
    response_list.append(getResponse(prompts[i]))
  return response_list
'''
#parsed_response = response["choices"][0]["text"].split("\n")
#print(parsed_response)
# with open("prompt_responses.txt", "w") as fp:
#    for prompt in prompts:
'''
response = {
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "logprobs": None,
      "text": "\nPerson A: \"Hey, I wanted to talk to you about a business opportunity that I think might be an opportune time to invest in.\"\nPerson B: \"Oh, what kind of opportunity is it?\"\nPerson A: \"It's a new tech startup that has a lot of potential for growth. I think now is the perfect time to get in on the ground floor.\""
    },
    {
      "finish_reason": "stop",
      "index": 1,
      "logprobs": None,
      "text": "\nPerson 1: Hi there, I'm glad we were able to meet at this opportune moment.\nPerson 2: Yes, I'm glad too. What did you want to discuss?"
    },
    {
      "finish_reason": "stop",
      "index": 2,
      "logprobs": None,
      "text": "\n\nA: \"I think this is an opportune time for us to invest in a new technology platform.\"\nB: \"Yes, I agree. We need to take advantage of this opportunity to stay ahead of the competition.\""
    }
  ],
  "created": 1674958259,
  "id": "cmpl-6drRj2eJd4k7rcUQRmNOCr1PdfFAC",
  "model": "text-davinci-003",
  "object": "text_completion",
  "usage": {
    "completion_tokens": 168,
    "prompt_tokens": 13,
    "total_tokens": 181
  }
}
'''

words_dict = {} 
with open("conversations.json", "w") as fp:
  
  for (i,prompt) in tqdm(enumerate(prompts)):
    word_list = []
    if i%20 == 0 and i != 0:
      time.sleep(65)
    response = getResponse(prompt)
    in_word = True
    
    for choice in response["choices"]:
      word_dict = {}
      text = choice["text"]
      split_response = text.split("\n")
      response_list = []
      for response in split_response:
          if len(response) != 0:
            if ":" in response:
              response_list.append(response.split(":",1)[1].strip().strip('"'))
            else:
              in_word = False
      word_dict["examples"] = response_list[0:-2]
      word_dict["synonyms"] = response_list[-2]
      if "(" in response_list[-1]:
        word_dict["meaning"] = response_list[-1].split("(")[0].strip()
      else:
        word_dict["meaning"] = response_list[-1]
      if in_word:
        word_list.append(word_dict)
  
    words_dict[words[i]] = word_list
    in_word = True
  
  fp.write(json.dumps(words_dict, indent=4, ensure_ascii=False))
  print("done")