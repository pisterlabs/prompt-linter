import pandas as pd
import re
import os

def extractPhysicalPlan(log):

  # Extract substrings between brackets
  # Using regex
  res = re.findall(r'\{.*?\}', text)

  plans = []  
  for i in res:
    if "physicalPlanDescription" in i:
      print(i)
      plans.append(i)
      
  print(plans)

  sub1 = '"physicalPlanDescription":'
  sub2 = ',"sparkPlanInfo"'

  physical_plan = re.findall(sub1+"(.*)"+sub2,plans[-1])[0]

  return physical_plan

# Create a list to store the text files
texts=[]

# Get all the text files in the text directory
for file in os.listdir("sparklogs"):

    # Open the file and read the text
    with open("sparklogs/" + file, "r", encoding="UTF-8") as f:
        text = f.read()
        if len(text) > 1000:
          print(text)
          text = extractPhysicalPlan(text)

          # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
          texts.append((file.replace('.txt',''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
#df['text'] = df.fname
df.to_csv('logs.csv')
df.head()


import tiktoken

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
tokenizer = tiktoken.get_encoding("cl100k_base")

df = pd.read_csv('logs.csv', index_col=0)
df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()


#df = df[df.n_tokens > 0]

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

df['embeddings'] = df.text.apply(lambda x: client.embeddings.create(input=x, model='text-embedding-ada-002').data[0].embedding)

df.to_csv('embeddings.csv')
df.head()

