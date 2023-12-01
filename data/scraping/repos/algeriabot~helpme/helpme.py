import openai
import sys
import os
import threading
import time


# Constants
SYSTEM_INFO = "Linux (Ubuntu 20.04)"
#SYSTEM_INFO = "Windows 10"
openai.api_key = "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"

# Message list
message_list = [
    {"role": "system", "content": f"Write a terminal command based on the description provided, and revise it as the user requests. Only output the command.\nSystem info: {SYSTEM_INFO}\nWorking directory: {os.getcwd()}"},
    
]

def highlight_bash(code):
  return code.strip()

### Loading wheel ###
# Braille loading wheel characters
BRAILLE_LOADING_WHEEL = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
# Global variable to signal the loading thread to stop
stop_loading = threading.Event()
def loading_wheel():
    idx = 0
    while not stop_loading.is_set():
        sys.stdout.write('\r' + '  Thinking... ' + BRAILLE_LOADING_WHEEL[idx])
        sys.stdout.flush()
        idx = (idx + 1) % len(BRAILLE_LOADING_WHEEL)
        time.sleep(0.1)
    print('\r               \r', end='')


# Get the message
message = ' '.join(sys.argv[1:])


### Main loop ###
keep_asking = True
execute = False
while keep_asking:

  message_list.append({"role": "user", "content": message})
  
  print()
  print("  ---------- COMMAND ----------")
  print()
  
  # Start the loading thread
  loading_thread = threading.Thread(target=loading_wheel)
  loading_thread.start()
  
  # Get completion, add to message list
  resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=message_list,
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
  )
  command = resp['choices'][0]['message']['content']
  message_list.append({"role": "assistant", "content": command})
  
  # Stop and join the threads together
  stop_loading.set()
  loading_thread.join()
  
  # Print the response
  print(f"  {highlight_bash(command)}")
  print("  -----------------------------")
  print()


  action = ""
  while not (action in ['x','r','q']):
    action = input("  e[x]ecute, [r]evise, or [q]uit? ")
    if action == "x":
      execute = True
      keep_asking = False
    elif action == "q":
      keep_asking = False
    elif action == "r":
      print(f"\r  {' ' * len('e[x]ecute, [r]evise, or [q]uit?  ')}", end='')
      message = input("\r  Enter revision: ")
  


if execute:
  print(f"  executing command: {command}\n")
  os.system(command)
else:
  print("  abort")
