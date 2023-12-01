import openai
import os
import sys

try:
	openai.api_key = os.environ['OPENAI_API_KEY']
except KeyError:
	sys.stderr.write("""
  You haven't set up your API key yet.
  
  If you don't have an API key yet, visit:
  
  https://platform.openai.com/signup

  1. Make an account or sign in
  2. Click "View API Keys" from the top right menu.
  3. Click "Create new secret key"

  Then, open the Secrets Tool and add OPENAI_API_KEY as a secret.
  """)
	exit(1)
while True:
	response = openai.ChatCompletion.create(
	 engine=
	 "text-davinci-003",  # only available if OpenAI has given you early access, otherwise use: "gpt-3.5-turbo"
	 # 32K context gpt-4 model: "gpt-4-32k"
	 prompt=input())

	print(response)
