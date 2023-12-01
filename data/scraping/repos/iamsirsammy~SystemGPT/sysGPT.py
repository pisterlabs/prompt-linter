import openai # AI imports

import os # Python/bash running imports
import pipreqs

import re # Misc imports

def codeBlockRemover(markdown_string): # This function was written by chatgpt! Not much else was since much of the other code is really context dependent and uses recent updates to the openai library (that's not in gpt's training data)
	# Use regex to look for the codeblock markdown syntax
	pattern = r"```[a-zA-Z]*\n([\s\S]*?)\n```"
	matches = re.search(pattern, markdown_string)
	
	if matches:
		# return the codeblock content
		return matches.group(1)
	
	# if no codeblock is found, return the original string
	return markdown_string

promptFile = open("prompt") # reads the prompt from the file
prompt = promptFile.read()
promptFile.close()

openai.organization = "org-I07eiw47mXjRl66XDbHtJ49n" # If you don't know what openai.organization is, keep it like this.
openai.api_key = "YOUR_API_KEY_HERE" # Change this to your API key. # https://platform.openai.com
model = "gpt-3.5-turbo" # If you have access to GPT-4, replace this with model = "gpt-4"

def bashRun(bash): # this prob isn't required, chatgpt knows how to use os.system. I'm hoping this'll encourage it to use bash tho
	os.system(bash)

def gpt(inputText): # chatgpt doesn't know how to use this well
	completion = openai.ChatCompletion.create(
	model=model,
	messages=[
		{"role": "user", "content": inputText},
	]
	)
	return completion.choices[0].message.content
	
try:
	os.mkdir("temp___") # weird name so chatgpt's code doesn't fail making a temp dir
	print("Created temp folder") # also temp has the last python file generated which might be useful if you want to rerun it
except:
	pass

while True: # Main loop
	userMsg = input("> ")
	print("Waiting...")
	completion = openai.ChatCompletion.create( # fun gpt code!!!
	model=model,
	messages=[
		{"role": "system", "content": prompt},
		{"role": "user", "content": userMsg}
	]
	)
	gptCode = codeBlockRemover(completion.choices[0].message.content)
	print(gptCode)
	if input("Run code? [y/N]") == "y": # could be better so u can say yes and yup and stuff (input()[0].lower())
		pyFile = open("temp___/lastCode.py", "w") # writes the code to a file so pipreqs can work and help me install all the required packages so the code works, i wish pipreqs accepted strings but oh well
		pyFile.write(gptCode)
		pyFile.close()
		
		try:
			os.remove("temp___/requirements.txt") # prob not needed anymore bc --force
		except:
			pass
		
		bashRun("pipreqs ./temp___ --force") # --force to remove requirements.txt if present (might be)
		bashRun("pip install -r temp___/requirements.txt") # might need to change on windows??? also maybe kinda maybe previous line, just download WSL
		exec(gptCode) # the most important line, actually runs gpt's python code after all this dependency installing and function declaring and garbage. can access global variables and functions which is cool for bashRun and gpt access but not cool for potentially overwriting my variables and other terrible things, should change at some point
	else:
		break
	
