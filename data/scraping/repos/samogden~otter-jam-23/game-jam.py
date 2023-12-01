#!env python
import base64
import configparser
import io
import logging
import os.path
from enum import Enum
from pprint import pprint

import openai
from PIL import Image
from io import BytesIO

import helper
from helper import CostTracker

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

class OpenAIInterface(object):
  cost_tracker = CostTracker()
  @staticmethod
  def get_api_key():
    config = configparser.ConfigParser()
    config_file = os.path.expanduser("~/.config/openai")
    if not os.path.exists(config_file):
      open_api_key = input("Please enter OpenAI API Key: ")
      config["DEFAULT"] = {"OPENAI_API_KEY" : open_api_key}
      with open(config_file, 'w') as fid:
        config.write(fid)
      openai.api_key = open_api_key
    else:
      config.read(config_file)
      openai.api_key = config["DEFAULT"]["OPENAI_API_KEY"]
      
  @classmethod
  def send_msg(cls, messages, model="gpt-3.5-turbo", temperature=0.6):
    response = openai.ChatCompletion.create(
      model=model,
      messages= messages,
      temperature=temperature
    )
    cls.cost_tracker.report_run(response["usage"], model)
    return response.choices[0].message
  
  @classmethod
  def get_image_prompt(cls, str):
    prompt = cls.send_msg([{
      "role" : "user",
      "content" : f"{str}"
    }])
    return prompt["content"]
  
  @classmethod
  def get_image(self, prompt, model="dall-e", size="256x256"):
    response = openai.Image.create(
      prompt=f"{prompt}",
      n=1,
      size=size,
      response_format="b64_json"
    )
    self.cost_tracker.report_run({}, model, size)
    image_buffer = io.BytesIO(
      base64.decodebytes(
        response['data'][0]['b64_json'].encode()
      )
    )
    return image_buffer
  
  @classmethod
  def image_variation(cls, last_image: BytesIO, model="dall-e", size="256x256"):
    last_image.seek(0)
    response = openai.Image.create_variation(
      image=last_image,
      n=1,
      size=size,
      response_format="b64_json"
    )
    cls.cost_tracker.report_run({}, model, size)
    image_buffer = io.BytesIO(
      base64.decodebytes(
        response['data'][0]['b64_json'].encode()
      )
    )
    return image_buffer


class Conversation(object):
  def __init__(self, conversation_background=None):
    self.messages = []
    if conversation_background is not None:
      self.add_msg("system", conversation_background)
    self.summary = None
  
  def add_msg(self, kind, msg):
    self.messages.append({
      "role" : kind,
      "content" : msg
    })
  
  def send_msg(self, msg, model="gpt-3.5-turbo"):
    self.add_msg("user", f"{msg}")
    response = OpenAIInterface.send_msg(self.messages)
    self.add_msg("assistant", response["content"])
    return response["content"]
  
  def summarize(self, num_words=200):
    return self.send_msg(f"Please summarize our previous conversation in {num_words} words or less.")
  
  def conversation_loop(self):
    while True:
      msg = input("> ")
      if msg.strip().lower() == "exit":
        self.summary = self.summarize()
        print(f"Summary: {self.summary}")
        break
      elif msg.strip().lower() == "show me":
        last_message = self.messages[-1]["content"]
        prompt = OpenAIInterface.get_image_prompt(
          f"Make me a 100 word description of an image based on: {last_message}"
          )
        image_buffer = OpenAIInterface.get_image(
          f"{prompt[:1000]}"
        )
        img = Image.open(image_buffer)
        img.show()
      else:
        resp = self.send_msg(msg)
        print(resp)
      logging.info(
        f"Total cost: ${OpenAIInterface.cost_tracker.cumulative_cost : 0.6f}"
        )


def run_adventure(adventure_hook):
  
  # Generate some background
  background = Conversation(adventure_hook)
  description = background.send_msg("In 100 words or less, describe the appearance of the assistant.")
  image_buffer = OpenAIInterface.get_image(description)
  img = Image.open(image_buffer)
  img.show()
  description = background.send_msg("In 100 words or less, describe the appearance of the assistant and the location.")
  
  # Get an initial introduction to show the user
  conversation_hook = background.send_msg(f"{adventure_hook}.  The scene can be described as: {description}")
  print(conversation_hook)
  
  # Start the users new adventure and ask them for input
  conversation = Conversation(conversation_hook)
  conversation.conversation_loop()
  
  # Potentially come back to the bar
  continue_answer = input("Would you like to continue your adventure? (y/n) ")
  if continue_answer.lower()[0] == "y":
    summary = conversation.summary
    conversation_hook = background.send_msg(f"In 100 words, describe the scene for the next interaction.  A summary of the previous interaction is: {summary}")
    print(conversation_hook)
    conversation = Conversation(conversation_hook)
    conversation.conversation_loop()
  
  

def main():
  print("Welcome!")
  OpenAIInterface.get_api_key()
  print("Starting adventure....")
  adventure_hook = "I am entering a fantasy tavern to talk to the bartender about getting a cup of ale and possibly some quests.  The bartender, played by the assistant, has a gruff personality but is willing to take my money."
  
  
  adventure_hook = "I am entering the captains cabin of a pirate ship looking to be hired on.  The captain, played by the assistant, is a cruel man, but interested in any special skills I may have that can aid him and his crew."
  
  run_adventure(adventure_hook)
  
  return

if __name__ == "__main__":
  main()
