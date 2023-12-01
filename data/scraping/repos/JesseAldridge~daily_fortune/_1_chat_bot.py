import os, json, glob, random, textwrap, threading, asyncio, subprocess, sys

import chardet, nltk
from discord.ext import tasks

sys.path.append(os.path.expanduser('~/Dropbox/openai_wrapper'))
import openai_wrapper

import _0_discord


with open('words.txt') as f:
  words = f.read().strip().splitlines()

class ChatBot:
  async def init(self, channel, client):
    self.channel = channel
    self.client = client

    @tasks.loop(seconds=60 * 60 * 24)
    async def fortune_loop():
      base_prompt = 'Fun fact about a random, esoteric topic'
      with_word = f'{base_prompt}, inspired by the word "{random.choice(words)}"'
      if random.random() < .5:
        prompt = f'{with_word}:'
      else:
        prompt = f'{with_word} (utterly deranged and untrue):'

      response_str = openai_wrapper.openai_call(prompt)

      sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
      sentences = sent_detector.tokenize(response_str.strip())

      if ':' in sentences[0]:
        response_str = response_str.split(':', 1)[1]

      # Bad sentences:
      # Psychologists have discovered that if you stare at a person for exactly 7 minutes and 13 seconds without blinking, you can gain control over their mind and make them believe they're a chicken. This phenomenon is known as "Poultryosis." (Please note this is completely false and made up for the purpose of humor).
      # Today, the flamingo society remains a strangely intriguing yet entirely untrue piece of trivia.
      # Despite this being utterly false, wouldn't it be fascinating if it were true? Dolphins creating mermaids to protect their home - a whimsical notion indeed!
      # Despite being an imaginary civilization...

      for i in range(2):
        for word in (
          'false', 'untrue', 'deranged', 'whimsical', 'evidence', 'imaginary',
          'fictional', 'fake', 'made up',
        ):
          if word in sentences[-1]:
            sentences = sentences[:-1]

      response_str = ' '.join(sentences)

      print('response:', response_str)

      message = f"{base_prompt}:\n\n{response_str.strip() or ''}"
      if 'test' not in sys.argv:
        await self.channel.send(message)
    fortune_loop.start()

def main():
  _0_discord.main(ChatBot)

if __name__ == '__main__':
  main()
