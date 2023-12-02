import subprocess
import os
import openai 
from termcolor import colored
from util import get_cwd_structure, format_cwd_structure, generate_response, generate_response_prompt, generate_response_smart

openai.api_key = os.environ.get("OPENAI_KEY")


# Create a aws lambda function that downloads a file from s3 and splits it into pages and uploads the pages to s3.
SPEC = """
Split the util file into multiple files.
"""

filesSys = """
You are Software Engineer GPT.
These are the commands you can use:
- READ <path>
	- reads the file at the path and adds it to the context

====== OUTPUT FORMAT (IMPORTANT!)
READ <path>
READ <path>
READ <path>
...
"""
def filesPrompt(task):
	dirs = format_cwd_structure(get_cwd_structure())

	return """
	====== DIRECTORY STRUCTURE
	pwd = /app

	{dirs}

	====== TASK
	{task}

	====== PROMPT
	Which files do you need to read to complete this task? Use the READ command to read a file. Empty if none.

	====== OUTPUT
	""".replace("\t", "").format(dirs=dirs, task=task)


prompt = filesSys + "\n\n" + filesPrompt(SPEC)
print(colored("BUILDING CONTEXT", "green"))
# print(prompt)
res = generate_response_smart(prompt)


contents = []
reads = res.split("\n")
# print(reads)
for read in reads:
	print(colored("READING", "blue"), read)
	cmd = read.split(" ")[0]

	if cmd != "READ":
		raise Exception("Invalid command")

	path = read.split(" ")[1]
	try:
		with open(path, "r") as f:
			content = f.read()
			s = path + "\n" + content
			contents.append(s)
	except Exception as e:
		print(colored("ERROR", "red"), e)


doSys = """
You are Software Engineer GPT.
These are the commands you can use:
- WRITE_FILE <path> ```<data>```
	- writes (overrides) the data to the file at the path

====== OUTPUT FORMAT (IMPORTANT!)
WRITE_FILE <path> ```<data>```
WRITE_FILE <path> ```<data>```
WRITE_FILE <path> ```<data>```
...
"""

def doPrompt(task, contents):
	dirs = format_cwd_structure(get_cwd_structure())

	return """
	====== DIRECTORY STRUCTURE
	pwd = /app

	{dirs}

	====== FILES
	{contents}

	====== TASK
	{task}
	write code and tests to complete the task and use WRITE_FILE to save the files. on write code files.

	====== OUTPUT FORMAT
	WRITE_FILE <path> ```<file_contents>```
	...

	====== OUTPUT
	""".replace("\t", "").format(dirs=dirs, task=task, contents=contents)

prompt = doSys + "\n\n" + doPrompt(SPEC, "\n".join(contents))
print(colored("BUILDING CONTEXT", "green"))
# print(prompt)
res = generate_response_smart(prompt)

writes = res.split("WRITE_FILE ")
for write in writes:
	if write == "":
		continue

	print(colored("WRITING", "blue"), write)

	path = write.split(" ")[0]
	data = write.split("```")[1]
	# unescape
	data = data.replace("\\n", "\n")
	print(path)
	with open(path, "w") as f:
		f.write(data)





raise Exception("Exit")

