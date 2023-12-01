from voice_tts import SpeechAssistant
from loc_inf import LLMChatBot
from python_interpreter import PythonInterpreter, run_interpreter
import os
import openai
import datetime
import threading  # Import the threading module
import keyboard
import sys
import subprocess
import appdirs

# set variables
use_local_model = True
use_voice_tts = False
use_voice_recognition = True
# model path for whatever local model you want to use (if use_local_model is True)
# can't set environment variable if there is a space in the path so i have to hardcode it for now
# llm_model_path = r"C:\Users\Jeff\AppData\Local\Open Interpreter\Open Interpreter\models\codellama-7b-instruct.Q3_K_S.gguf"

# Input message
initialmachineprompt = "What would you like me to do, sir: "


# Create an instance of the SpeechAssistant class
if use_voice_tts or use_voice_recognition:
    assistant = SpeechAssistant(wakeup_word="interpreter")

if use_voice_tts:
    # tts the initial machine prompt
    tts_thread = threading.Thread(
        target=assistant.tts_and_play_audio, args=(initialmachineprompt,))
    tts_thread.start()  # Start the TTS thread
else:
    print(initialmachineprompt)

# Get input from speech recognition
if use_voice_recognition:
    message = assistant.start_speech_recognition()
else:
    message = input()

# if not local model, use OpenAI Chat API
if use_local_model:
    # get all local models from app directories
    repo_id = "codellama-7b-instruct.Q3_K_S"
    user_data_dir = appdirs.user_data_dir("Open Interpreter")
    default_path = os.path.join(user_data_dir, "models")
    model_path = os.path.join(default_path, repo_id)
    # replace \ with \\ for windows and add the .gguf extension
    model_path = model_path.replace("\\", "\\\\")
    model_path += ".gguf"
        

    llm = LLMChatBot(model_path=llm_model_path)
    output_file = "output/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".py"
else:
    # Set up OpenAI API
    openai.api_type = "azure"
    openai.api_base = os.getenv("AZURE_API_BASE")
    openai.api_version = os.getenv("AZURE_API_VERSION")
    openai.api_key = os.getenv("AZURE_API_KEY")
    output_file = "output/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".py"
    
    # Create a response using OpenAI Chat API
    response = openai.ChatCompletion.create(
        engine="gpt-35-turbo",
        messages=[
            {"role": "user", "content": "You are a computer with the ability to run any code you want when you are given a prompt and return a response with a plan of what code you want to run. You should start your response with a plan, The commands you provide should be in a single code block encapsulated in '''python and ''' for Python and should be valid Python programs."},
            {"role": "assistant", "content": "I am a computer with the ability to run any code I want when I am given a prompt and return a response with a plan of what code I want to run I Will start my response with a plan. The commands I provide should be in a single code block encapulated in '''python and ''' and should be a valid Python program."},
            {"role": "user", "content": message}
        ],
        temperature=0.001,
        max_tokens=2000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
if use_local_model:
    # get response from local model
    response = llm.get_response(message)

    # Print the response
    print("Response:", response)

# If the response is not empty, print the response, this is horrible i will do something to clean this up maybe i know its inefficient
if response:
    if use_local_model:
        print("Response:", response)
        plan = response.split("```python")[0]
        plan = plan.replace("'", "")
        plan = plan.replace('`', "")
        print("plan:", plan)
    else:
        print(response.choices[0].message.content)
        # Extract plan from the response
        plan = response.choices[0].message.content.split("```python")[0]
        plan = plan.replace("'", "")
        plan = plan.replace('`', "")
        print("plan:", plan)
    
    if use_voice_tts:
        # Create a thread for TTS and audio playback
        assistant.tts_and_play_audio(plan)

    # Check if there's Python code in the response
    if use_local_model and "```python" in response:
        python_code = response.split("```python")[1].split("```")[0].strip()
        print("Python code:", python_code)
    elif use_local_model and "```" in response:
        python_code = response.split("```")[1].split("```")[0].strip()
        print("Code found in the response but not Left out the word python:", python_code)
    elif not use_local_model and "```python" in response.choices[0].message.content:
        python_code = response.choices[0].message.content.split(
            "```python")[1].split("```")[0].strip()
        print("Python code:", python_code)

        # Run and possibly save the Python code
    if python_code:
        # create an instance of the PythonInterpreter class
        interpreter = PythonInterpreter()

        # send the code to the PythonInterpreter class
        interpreter_code_output = run_interpreter(python_code)
        print("Python code output:\n", interpreter_code_output)
        # send response to the local model
        if use_local_model:
            
            llm_chat_response = llm.get_response(
                "you have ran the python program you produced and it produced this response " + interpreter_code_output + " what would you like me to do next?")
            print("Response:", llm_chat_response)
            if llm_chat_response:
                if "```python" in llm_chat_response:
                    python_code = llm_chat_response.split(
                        "```python")[1].split("```")[0].strip()
                    print("Python code:", python_code)
                elif "```" in llm_chat_response:
                    python_code = llm_chat_response.split(
                        "```")[1].split("```")[0].strip()
                    print("Code found in the response but not Left out the word python:", python_code)
                else:
                    print("No Python code found in the response")

                # Run and possibly save the Python code
                if python_code:
                    interpreter_code_output = run_interpreter(python_code)
                    print("Python code output:\n", interpreter_code_output)
                    llm_chat_response = llm.get_response(
                        "you have ran the python program you produced and it produced this response " + interpreter_code_output + " what would you like me to do next?")
                    print("Response:", llm_chat_response)

        
    else:
        print("No Python code found in the response")

# if we are at the end of the program, close the TTS thread
if use_voice_tts and tts_thread.is_alive():
    tts_thread.join()
    
