#!/usr/bin/python3

print("Content-type: text/html")
print()

from langchain.llms import OpenAI
from langchain.agents import load_tools, AgentType, initialize_agent
import os
import cgi

# Set your OpenAI API key here
my_key = "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"

# Set your SerpApi API key here
my_serp_key = "bfa2b2c0c1008e3016eaf749ba6cca01596c427cb3697091bd6c2db8d480ac3f"
os.environ["SERPAPI_API_KEY"] = my_serp_key

# Get user input from the form
form = cgi.FieldStorage()
user_prompt = form.getvalue("prompt")

# Initialize OpenAI language model
my_llm = OpenAI(model='text-davinci-003', temperature=1, openai_api_key=my_key)

# Load SerpApi tool
my_serp_tool = load_tools(tool_names=['serpapi'])

# Initialize Google Chain agent
my_google_chain = initialize_agent(
    llm=my_llm,
    tools=my_serp_tool,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Check if user_prompt is not None and not an empty string
if user_prompt:
  result = my_google_chain.run(user_prompt)

    # Display the result
    print("<h2>Google Chain Result:</h2>")
    print(f"<p>{result}</p>")
else:
    print("<h2>Please provide a valid prompt.</h2>")

    # Run the Google Chain with the user input
  
