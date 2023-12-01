from dotenv import load_dotenv
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
  AIMessage,
  HumanMessage,
  SystemMessage
)

import argparse
import sys

# Make sure to set OPEN_API_KEY in .env
load_dotenv()

parser = argparse.ArgumentParser("simple-summarizer")
parser.add_argument("-f", "--file", help="The file to summarize.")
args = parser.parse_args()

# Try to read the file's content into file_content
file_content = ""
try:
  with open(args.file, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines, start=1):
      file_content += f"{str(i).rjust(6, '0')} {line}"
except FileNotFoundError:
  sys.stderr.write("Please supply a filename with the flag '--file' or '-f'.")
  exit()
except IOError:
  sys.stderr.write("Error reading the file.")
  exit()

# Use the LLM to summarize the code
llm = OpenAI(temperature=0)

prompt = f'''
You are a highly intelligent, highly skilled enterprise software engineer. 
Please summarize the following code. 
Make specific references to lines of code using the format "[000000]".
If possible, explain what problem this code is trying to solve at a high level.
--- begin example.go ---
000001 package main
000002 
000003 import "fmt"
000004 
000005 func main() {{
000006 
000007     messages := make(chan string, 2)
000008 
000009     messages <- "buffered"
000010     messages <- "channel"
000011 
000012     fmt.Println(<-messages)
000013     fmt.Println(<-messages)
000014 }}
--- end example.go ---
--- begin summary ---
This code sends two messages "buffered" and "channel" to the channel "messages" ([000007, 000009 - 000010]). It then takes these values and prints them to the output at a later time, asynchronously ([000012 - 000013]).
--- begin {args.file} ---
{file_content}
--- end {args.file} ---
--- begin summary ---
'''

# prompt = f"""
# Please make a diagram for the following code in mermaid.js. Use the line numbers instead of the content of each line.
# --- begin {args.file} ---
# {file_content}
# --- end {args.file} ---
# """

print("PROMPT")
print(prompt)
print("RESULT")
result = llm.predict(prompt)
print(result)