# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import os
import openai
import sys

openai.api_key = os.environ["OPENAI_API_KEY"]

# called to log the API usage for debugging
def logLmUsage(request, completion):
  #assert typeof(request) == 'str'
  
  import uuid
  filename = "z_promptTrace__ModuleNlp_v1__"+str(uuid.uuid4())+".txt"
  f = open(filename, "w")
  f.write(request + "\n\n\n")
  f.write(str(completion))
  f.flush()
  f.close()

argText = sys.argv[1] # read text to translate from the argument

completion = openai.ChatCompletion.create(
  temperature=0.0,
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a translator."},
        #{"role": "user", "content": "Express the sentence \"My cat is white, fat and furry and lives in the basement of my mom.\" as multiple simple english sentences!"},
        
        # version 0: works!
        #{"role": "user", "content": f"Express the sentence \"{argText}\" as multiple simple english sentences!"},

        # version 1: force as enumeration
        {"role": "user", "content": f"Express the sentence \"{argText}\" as multiple simple english sentences, enumerate them!"},
    ]
)

logLmUsage(argText, completion)

# print for debugging
print( completion.choices[0].message.content )

lines = completion.choices[0].message.content.split('\n')

# clean raw output from LM
def cleanText(txt):
  res = txt
  
  if res[1] == '.': # is it part of a enumeration?
    res = res[3:]
   
  # remove trailing dot
  if res[-1] == '.':
    res = res[:-1]
  
  return res

# remove superfluous words
def removeWords(txt):
  if txt[:4] == 'the ': 
    txt = txt[4:]
  
  txt = txt.replace(' a ', ' ')
  txt = txt.replace(' the ', ' ')
  
  return txt

for iLine in lines:
  iLine = iLine.lower()
  
  cleaned0 = cleanText(iLine)
  cleaned1 = removeWords(cleaned0)
  #print(cleaned0) # DBG
  print(cleaned1)

  
  if True: # codeblock to translate to narsese
    tokens = cleaned1.split(" ")
    if tokens[1] == "is":
      subj = tokens[0]
      pred = "_".join(tokens[2:])
      print(f"<{subj} --> {pred}>.")
    
    elif tokens[1] in ["has","can"]: # all other relations
      rel = tokens[1]
      a = tokens[0]
      b = "_".join(tokens[2:])
      print(f"<({a}*{b}) --> {rel}>.")
      

