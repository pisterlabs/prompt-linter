import openai
import os

def ai_location_search(pdf_txt):
  '''
  uses gpt to search inspection report text for location data. First parses text 
  and then calls ai_classify to perform binary classification of each text line. Hard cap
  calls to ai_classify by setting search_depth val. Conditional branch executed 
  if document formatting does not match assumed formatting or if no location is found. 
  Conditional branch calls ai_analyze to search entire text extraction for locations.
  WARNING: If ai_analyze is called, results may be unpredictable. 
  
  Input: pdf_text string 
  
  Return: Locations or no location notification as string.
  '''
  #split text at fixed text marker in document. Replace as needed.
  split_flag = "some fixed substring marker in text extract"
  #flag to control conditional logic path
  bool_flag = False
  #split text if flag found 
  if split_flag in pdf_txt:
      locations = "" #return value once locations are found 
      split_text = pdf_txt.split(split_flag) #split text at flag
      lines = split_text[1].split("\n") #split text at each newline for traversal
      search_depth = 17 #set num lines to search for locations.
      for i in range(search_depth): #traverse each line of text
        if i > 5: #skip first few lines after split
          if ai_classify(lines[i]): #if line of text looks like location execute 
            locations += lines[i]+"\n" #append loc(s). If multiple seperate w '\n'
            bool_flag = True #At least one location found. Flag True

  if bool_flag: #locations found with classify function
    #print(f'classify: {locations}') #return locations
    return locations 
  else: 
    #alternate format or no location. Call analyze to search entire document
    locations = ai_analyze(pdf_txt)
    #print(f'analyze: {locations}')
    return locations 


def ai_classify(line):
  """
  use openai gpt to determine whether a string matches a pattern.
  Input: line string 
  Return: Boolean true / false if pattern matches 
  """
  #load openai api_key from env variables
  openai.api_key = os.getenv("OPENAI_API_KEY")
  
  #examples of data to match for LLM binary classification prompt 
  location_examples = (
    "Gallifrey - Citadel; Time Rotor Room; Grid lines TARDIS.4/REG.5; Installation of Time Vortex Stabilizers\n"
    "Skaro - Dalek City; Emperor's Chambers; TARDIS.9/EX.3; Reinforcement of Dalek Battle Armor\n"
    "Location: TARDIS; Roof; Mod 15 Outrigger Retrofit for Connection Type F; Console Room - Cloister Room/TARDIS.1\n"
    "Mondas - Cybermen Outpost; Cyber-Conversion Chambers; TARDIS.5/TQ-TR; Welding of Cyberman Support Struts\n"
    "Gallifrey - Academy; Prydonian Chapter Room; Grid lines TM.3/TARDIS.9; Installation of Time Scoop Mechanism\n"
    "Gallifrey - Time War Battlefield; OS-Interface; TB.1-TB.5/TARDIS.7; Threaded Rod Welding for Time Lord Weaponry\n"
    "Karn - Sisterhood of Karn; Basement; TG-TJ/TARDIS.8 Field Work for Elixir of Life Production"
  )
  
  #prompt to LLM for binary classification of data (responds Yes if match. Responds No if no match)
  classifier_prompt = (f'Here are examples of different locations seperated by newlines. Is this input also a location? Respond with only "yes" or "no"\n'
      f'"yes" examples: {location_examples}\n'
      f'input: {line}')
  #system message helps set the behavior of gpt
  system_message = "You parse text extracted from pdfs and categorize it"
  
  #DEBUG - check prompt formatting 
  #print(f'Prompt: {classifier_prompt}')
  
  #call openai - gpt-3 turbo model
  response = openai.ChatCompletion.create(
    model ="gpt-3.5-turbo",
    temperature=0.2, #lower temperature to reduce randomness in responses
    messages=[
      {"role": "system", "content": system_message},
      {"role": "user", "content": classifier_prompt}
    ]
  )
  
  #DEBUG - Print gpt response for debugging
  """
  print(type(response))
  print(response) #includes all response metadata 
  print("gpt3.5 response text: "+ response['choices'][0]['message']['content'])
  """
  
  #Save openai yes/no response and format to machine friendly string 
  ai_response = response['choices'][0]['message']['content'] #save gpt response
  ai_response = ai_response.replace('\n', '') #remove newlines
  ai_response = ai_response.lower() #convert to lowecase 

  #DEBUG - formatted ai response
  print(f'Formatted Response: {ai_response}')

  #Convert yes / no to boolean value for return
  if "yes" in ai_response:
    #print('True')
    return True
  else: # Either ai_response is 'no' or Skynet became self aware 
    #print('False')
    return False


def ai_analyze(text):
  """
  use openai gpt to analyze entire pdf text extraction and search for location
  Input: text string 
  Return: location(s) string
  """
  #load openai api_key from env variables
  openai.api_key = os.getenv("OPENAI_API_KEY")
  
  #examples of data to match for LLM binary classification prompt 
  location_examples = (
    "Gallifrey - Citadel; Time Rotor Room; Grid lines TARDIS.4/REG.5; Installation of Time Vortex Stabilizers\n"
    "Skaro - Dalek City; Emperor's Chambers; TARDIS.9/EX.3; Reinforcement of Dalek Battle Armor\n"
    "Location: TARDIS; Roof; Mod 15 Outrigger Retrofit for Connection Type F; Console Room - Cloister Room/TARDIS.1\n"
    "Mondas - Cybermen Outpost; Cyber-Conversion Chambers; TARDIS.5/TQ-TR; Welding of Cyberman Support Struts\n"
    "Gallifrey - Academy; Prydonian Chapter Room; Grid lines TM.3/TARDIS.9; Installation of Time Scoop Mechanism\n"
    "Gallifrey - Time War Battlefield; OS-Interface; TB.1-TB.5/TARDIS.7; Threaded Rod Welding for Time Lord Weaponry\n"
    "Karn - Sisterhood of Karn; Basement; TG-TJ/TARDIS.8 Field Work for Elixir of Life Production"
  )
  
  #prompt to gpt
  analyze_prompt = (
  f'Search this text for locations. Return locations if found.\n'
  f'text: {text}'
  )
  #system message helps set the behavior of gpt
  system_message = (
  f'You are a master at searching a document for locations and '
  f'returning found locations to the user. List the locations, and ' 
  f'only the locations. Do not add context to the response.'
  f'If no locations are found reply "no locations found".\n'
  f'Here are examples of what locations look like in this context:\n'
  f'{location_examples}'
  )
  
  #TEST - check prompt formatting 
  #print(f'Prompt: {classifier_prompt}')
  
  #call openai - gpt-3 turbo model
  response = openai.ChatCompletion.create(
    model ="gpt-3.5-turbo",
    temperature=0.2, #lower temperature to reduce randomness in response
    messages=[
      {"role": "system", "content": system_message},
      {"role": "user", "content": analyze_prompt}
    ]
  )

  #DEBUG - Print gpt prompt, response, token count, etc.
  '''
  print(f'Prompt: {analyze_prompt}')
  print(f'System Message: {system_message}')
  print(type(response))
  print(response) #includes all response metadata 
  print("gpt3.5 response text: "+ response['choices'][0]['message']['content'])
  '''
  
  #Return GPT Locations Extractions w/ Disclaimer
  return ("AI Extracted:\n" 
          + response['choices'][0]['message']['content'])

  

