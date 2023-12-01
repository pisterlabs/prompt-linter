'''
Introducing FenixAGI, an advanced AI assistant designed to revolutionize project management. Powered by Open A.I.'s GPT-16k 3.5-turbo language model, Fenix is a versatile companion that assists users in various tasks.
'''

import openai
import json
import os
import pandas as pd
from termcolor import colored
from functions import *
from function_descriptions import function_descriptions
import pyttsx3
import tenacity
import re
from FileOperations import FileOperations
from FenixState import FenixState
from voice_assistant import VoiceAssistant
import speech_to_text

openai.api_key = os.getenv("OPENAI_API_KEY")
# check if the api key is valid with a try catch block
try:
    openai.Completion.create(engine="davinci", prompt="This is a test", max_tokens=5)
except Exception as e:
    print(e)
    

va = VoiceAssistant()

approved_functions = [
    "search_codebase",
    "bing_search_save",
    "scrape_website",
    "save_fenix",
    "read_file",
    "write_file",
    "delete_file",
    "create_directory",
    "ask_user_for_additional_information",
    "create_code_search_csv",
    "count_tokens_in_string",
    "count_tokens_in_file",
    "create_markdown_file",
    "fenix_help",
    "move_file",
    "list_files_in_directory",
    "suggest_function_chain",
    "visualize_data_3d",
    "get_folder_hierarchy",
]

base_instructions = "Fenix A.G.I. Mark-III is an advanced AI assistant built by a guy named parth. Fenix is an AI agent built on top of the Open A.I. GPT language models. Fenix can execute the following functions:" + str(approved_functions) + \
    "Fenix can also learn from the user's feedback and revise its instructions to improve its performance over time. Designed to be extended and personalized, Fenix is a powerful tool for any developer, researcher, or student."

COLORS = {
    'launch': 'cyan',
    'function_call': 'cyan',
    'function_info': 'green',
    'important': 'red',
    'query': 'green',
    'response': 'blue',
    # Add more as necessary...
}



def fenix_help(help_query):
    help_text = '''
Fenix is an advanced AI assistant made by parth: https://www.linkedin.com/in/parthspatil/ https://twitter.com/htrapvader 
For more wacky LLM projects: https://replit.com/@p4r7h.
  Restoring Fenix State
  At the start of a session, the function checks if a saved state of the Fenix AI assistant exists. If it does, the state is loaded and the session continues from where it left off last time. If not, a new state is initialized.
  User Input
  The function continuously prompts the user for input and processes it accordingly.
  Special Commands
  There are several special commands the user can input to guide the behavior of Fenix:
  'exit' or 'quit': Terminates the session and saves the current state of Fenix.
  'v': Toggles between different voice_modes for Fenix. Fenix can use the default voice, a voice clone of the creator (parth), or no voice at all. Feel free to add your own voice clones to the list of available voices.
  'a': Toggles between manual and automatic mode. In manual mode, Fenix will ask for approval before executing a function. In automatic mode, approved functions are executed automatically.
  'd': Toggles whether the assistant's responses should be displayed or not.
  'e' or ' ' : Records the user's voice for 5 seconds and transcribes it. The transcription is then used as the user's input.
  'r': Resets Fenix to a default state, clearing the conversation history and meta instructions.
  AI Response Generation
  The user's input (and the entire conversation history) is fed into an instance of the GPT-3.5-turbo model, which generates the assistant's response. If the response includes a function call, the function is executed if it is approved and the mode is either manual and approved by user, or automatic. The result of the function call can optionally be displayed to the user, depending on the current settings. OpenAI API keys can be obtained here: https://platform.openai.com.
  Web Search
    If the user's input is a web search query, the function uses the Bing Search API to search the web and returns the top 10 results. The results are saved to a CSV file and the file path is returned to the user. You can get your api key here: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api  
  Extending Functionality
  New functionality can be added to Fenix by adding new functions to the approved_functions list and corresponding entries to the function_descriptions list.
  Error Handling
  Errors during the execution of a function are handled by returning an error message to the user and not executing the function. Unrecognized user inputs in response to a function call prompt are treated as 'no' responses.
  Future Development
  Currently, the conversation history used to generate responses is limited by the maximum context length of the GPT-3.5-turbo model. Future versions of the run_conversation() function may implement more sophisticated methods for managing long conversation histories.'''
    help_text += "\n\nFenix is also capable of executing a wide range of functions, and visualizing information, these don't have explicit keystrokes. Here are the available functions for Fenix: "
    help_text += "\n".join(
        [f"\n {function}" for i, function in enumerate(approved_functions)])
    return help_text
    help_text += "\n\nFenix is also capable of executing a wide range of functions, these don't have explicit keystrokes. Here are the available functions for Fenix: "
    help_text += "\n".join(
        [f"\n {function}" for i, function in enumerate(approved_functions)])
    return help_text


def critique_and_revise_instructions(instructions, conversation_history, approved_functions):
    chat_log = '\n'.join([
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in conversation_history
    ])

    meta_prompt = f"""The Assistant has just had the following interactions with a User. Please critique the Assistant's performance and revise the Instructions based on the interactions.

    ####
    Approved Functions (the Assistant can use these functions):
    {approved_functions}
    Chat Log:
    {chat_log}

    ####

    First, critique the Assistant's performance: What could have been done better? 
    Then, revise the Instructions to improve the Assistant's responses in the future. 
    The new Instructions should help the Assistant satisfy the user's request in fewer interactions. 

    Start your critique with "Critique: ..." and your revised instructions for the assistant with "New Instructions:".

    """

    meta_response = openai.ChatCompletion.create(model="gpt-3.5-turbo-16k-0613",
                                                 messages=[{
                                                     "role": "user",
                                                     "content": meta_prompt
                                                 }])
    tell_user(
        "Analyzing conversation history and learning from your feedback...",
        "yellow",fenix_state.voice_mode)
    print(colored("Meta Prompt: " + meta_prompt, "cyan"))
    print("Token count is:", count_tokens_in_string(meta_prompt))
    meta_text = meta_response['choices'][0]['message']['content']
    # count_tokens_in_string(meta_text)
    print(
        colored(
            "Meta Critique: " +
            meta_text.split("Critique: ")[1].split(
                "New Instructions:")[0].strip(),
            "yellow"))
    try:
        new_instructions = meta_text.split("New Instructions:")[1].strip()
        print(colored("New Instructions: " + new_instructions, "cyan"))
    except IndexError:
        print("No new instructions found in AI output.")
        return ""

    return new_instructions


def stringify_conversation(conversation):
    return ' '.join([str(msg) for msg in conversation])


def ask_user(question, color='purple'):
    return input(colored(f"\n{question}", color))


def tell_user(message, color='blue', voice_mode=None):
    print(colored(message, color))
    if voice_mode is not None:
        va.voice_message(base_instructions, message, voice_mode)

def truncate_conversation(conversation, user_input="", function_response=""):
    MAX_TOKENS = 15000

    if count_tokens_in_string(stringify_conversation(conversation)) > MAX_TOKENS:
        # Remove oldest messages until we have room for new ones
        while count_tokens_in_string(stringify_conversation(conversation)) + count_tokens_in_string(user_input) + count_tokens_in_string(str(function_response)) > MAX_TOKENS:
            # print("Dropping: " + str(conversation[2]))
            for _ in range(2):  # Two pop operations
                if len(conversation) > 2:  # Ensure we have enough messages to remove
                    removed_message = conversation.pop(2)
                    # print(f"Removing message due to token limit: {removed_message}")
                else:
                    break  # If not enough messages to remove, exit the loop
    return conversation




@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(2))
def get_base_streaming_response(model, messages):
    # try to get a response from the model if it fails, drop the 3rd oldest message and try again
    try:
        # use the openai api to generate a response
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stream=True,
        )
    except Exception as e:
        print(e)
        messages.pop(2)
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            stream=True,
        )
    responses = ''
    # Process each chunk
    for chunk in response:
        if "role" in chunk["choices"][0]["delta"]:
            continue
        elif "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            responses += r_text
            print(r_text, end='', flush=True)
    assistant_message = responses
    messages.append({
        "role": "assistant",
        "content": assistant_message
    })
    return messages, assistant_message


@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_fixed(2))
def get_function_calling_response(model, messages, functions, function_call):
    # try to get a response from the model if it fails, drop the 3rd oldest message and try again
    try:
        # use the openai api to generate a response
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call=function_call,
        )
    except Exception as e:
        print(e)
        messages.pop(2)
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            functions=functions,
            function_call=function_call,
        )
    return response


def run_conversation():
    global fenix_state
    # Check if the state file exists
    if os.path.exists("fenix_state.json"):
        fenix_state = FileOperations.rez_fenix()
        conversation = fenix_state.conversation
        conversation.append(
            {"role": "system", "content": "Fenix State Rezzed."})
    else:
        fenix_state = FenixState(display_response=True,
                                 mode="auto",
                                 approved_functions=approved_functions,
                                 voice_mode="pyttsx3")
        conversation = fenix_state.conversation
        conversation.append(
            {"role": "system", "content": "Fenix State Created."})

    fenix_state.instructions = base_instructions
    tell_user("Agent Fenix is Online.", COLORS['launch'],None)

    tell_user("""There are several special keyboard commands the user can use to guide Fenix's behavior. Here are the special keyboard commands:

  'v': Toggles between different voice_modes for Fenix. Fenix can use the default voice, a voice clone of the creator (parth), or no voice at all. Feel free to add your own voice clones to the list of available voices.
  'a': Toggles between manual and automatic mode. In manual mode, Fenix will ask for approval before executing a function. In automatic mode, approved functions are executed automatically.
  'd': Toggles whether the assistant's responses should be displayed or not.
  'e' or ' ' : Records the user's voice for 5 seconds and transcribes it. The transcription is then used as the user's input.
  'r': Resets Fenix to a default state, clearing the conversation history and meta instructions.
  'exit' or 'quit': Terminates the session and saves the current state of Fenix.""",
              COLORS['important'],None)
    tell_user("Voice is now set to " + str(fenix_state.voice_mode),
              COLORS['important'],None)


    while True:
        user_input = ask_user("> ", COLORS['query'])
        if user_input.lower() in ["exit", "quit"]:
            tell_user("Exiting Fenix.", COLORS['important'],fenix_state.voice_mode)
            conversation.append(
                {"role": "system", "content": "Exiting Fenix."})
            FileOperations.save_fenix(fenix_state)
            conversation.append({"role": "system", "content": "State saved."})
            break

        elif user_input.lower() == " " or user_input.lower() == "e":
            speech_to_text.record(5, "output.wav")
            transcript = speech_to_text.transcribe("output.wav")
            user_input = transcript

        elif user_input.lower() == "a":
            # Toggle automatic function calling mode
            user_input = "Toggle automatic function calling mode."
            conversation.append({"role": "user", "content": user_input})
            if (fenix_state.mode == "manual"):
                fenix_state.mode = "auto"
                tell_user("Fenix will now execute approved functions automatically.",
                          COLORS['important'],fenix_state.voice_mode)
                conversation.append({
                    "role": "system",
                    "content": "Fenix is now in automatic mode."
                })
                conversation.append({
                    "role":
                    "system",
                    "content":
                    "Fenix will now execute approved functions automatically."
                })
            elif (fenix_state.mode == "auto"):
                fenix_state.mode = "manual"
                tell_user(
                    "Fenix will now ask for approval before executing a function.",
                    COLORS['important'],fenix_state.voice_mode)
                conversation.append({
                    "role": "system",
                    "content": "Fenix is now in manual mode."
                })
                conversation.append({
                    "role":
                    "system",
                    "content":
                    "Fenix will now ask for approval before executing a function."
                })
        elif user_input == "d":
            # Toggle display response
            user_input = "Toggle display response."
            conversation.append({"role": "user", "content": user_input})
            fenix_state.display_response = not fenix_state.display_response
            tell_user(
                f"Display Function Response is now set to {fenix_state.display_response}.",
                COLORS['important'],fenix_state.voice_mode)

            conversation.append({
                "role":
                "system",
                "content":
                f"Display Function Response is now set to {fenix_state.display_response}."
            })

        elif user_input.lower() == "v":
            # Toggle voice
            conversation.append({"role": "user", "content": "Toggle voice."})

            if fenix_state.voice_mode == "pyttsx3":
                va.switch_voice("eleven_monolingual_v1")
                fenix_state.voice_mode = va.voice_mode
            elif fenix_state.voice_mode == "eleven_monolingual_v1":
                va.switch_voice(None)
                fenix_state.voice_mode = va.voice_mode
            elif fenix_state.voice_mode == None:
                va.switch_voice("pyttsx3")
                fenix_state.voice_mode = va.voice_mode

            conversation.append({
                "role": "system",
                "content": "Voice is now set to " + str(fenix_state.voice_mode)
            })
            tell_user(
                f"Voice is now set to {fenix_state.voice_mode}.",
                COLORS['important'],fenix_state.voice_mode)


        elif user_input.lower() == "r":
            # update meta instructions and derez fenix
            temp_instructions = critique_and_revise_instructions(fenix_state.instructions,
                                                                 conversation, approved_functions)
            # Derez Fenix
            user_input = "Derez Fenix."
            conversation.clear()  # Clear the conversation list
            fenix_state.conversation.clear()  # Clear the conversation in FenixState
            conversation.append({"role": "user", "content": user_input})

            FileOperations.derez_fenix()

            fenix_state = FenixState(display_response=False,
                                     instructions=base_instructions+" " + temp_instructions,
                                     mode="auto",
                                     approved_functions=approved_functions)
            tell_user("Meta Instructions updated and conversation history reset.",
                      COLORS['important'],fenix_state.voice_mode)
            tell_user("I have appended the following instructions to my base instructions: " + temp_instructions,
                      COLORS['important'],fenix_state.voice_mode)
            conversation = fenix_state.conversation
            conversation.append({
                "role": "system",
                "content": "New Fenix State Created."
            })

        # Check if the user input is a function call
        if user_input.lower() not in ("v", "a", "d", "r"):
            print(colored("User Input: " + user_input, COLORS['query']))
            conversation.append({"role": "user", "content": user_input})
            response = get_function_calling_response(
                model="gpt-3.5-turbo-16k-0613",
                messages=conversation,
                functions=function_descriptions,
                function_call="auto",
            )

            message = response["choices"][0]["message"]
            if message.get("function_call"):
                print(colored(("Function Call:" + str(message.get('function_call'))
                                ), "cyan"))
                function_name = message["function_call"]["name"]
                if function_name in approved_functions:
                    try:
                        args = json.loads(message["function_call"]["arguments"])
                    except json.JSONDecodeError as e:
                        print(f"Failed to decode JSON: {e}")
                    else:
                        current_function_call = (function_name, args)

                    if fenix_state.mode == "manual":
                        user_input = ask_user("Do you want to run the function? (y/n)",
                                                COLORS['query'])
                        if user_input.lower() in ["y", "yes"]:
                            try:
                                function_response = eval(function_name)(**args)
                            except Exception as e:
                                function_response = f"Error: {str(e)}"

                        elif user_input.lower() in ["n", "no", "exit", "quit"]:
                            tell_user(
                                "Function Call: Not executing function", COLORS['important'],fenix_state.voice_mode)
                            assistant_message = "Function execution skipped by user."
                            conversation.append({
                                "role": "assistant",
                                "content": assistant_message
                            })
                            function_response = None
                        else:
                            tell_user(
                                "Unrecognized input. Default action is not to execute the function.",
                                COLORS['important'],fenix_state.voice_mode)
                            assistant_message = "Function execution skipped due to unrecognized input."
                            conversation.append({
                                "role": "assistant",
                                "content": assistant_message
                            })
                            function_response = None
                    elif fenix_state.mode == "auto":
                        try:
                            function_response = eval(function_name)(**args)
                        except Exception as e:
                            function_response = f"Error: {str(e)}"

                    if function_response is not None:
                        print(
                            colored("Function Response: "+str(function_response), COLORS['response']))
                        
                        function_content = {
                            "role": "function",
                            "name": function_name,
                            "content": str(function_response),
                        }

                        conversation.append({
                            "role": "system",
                            "content": "Function Response: " + str(function_response)
                        })
                        print("Conversation length (tokens): " +
                                str(count_tokens_in_string(stringify_conversation(conversation))))
                        conversation = truncate_conversation(
                            conversation, user_input, function_response)

                        conversation, assistant_message = get_base_streaming_response(
                            model="gpt-3.5-turbo-16k-0613",
                            messages=conversation + [
                                {
                                    "role": "user",
                                    "content": user_input
                                },
                                function_content,
                            ],
                        )

                        conversation.append({
                            "role": "assistant",
                            "content": assistant_message
                        })

                else:
                    tell_user("Sorry, I don't have access to that function.",
                                COLORS['important'],fenix_state.voice_mode)
                    assistant_message = "Function execution skipped by assistant."
                    conversation.append({
                        "role": "assistant",
                        "content": assistant_message
                    })

            else:
                conversation = truncate_conversation(conversation, user_input)
                conversation, assistant_message = get_base_streaming_response(
                    model="gpt-3.5-turbo-16k",
                    messages=[
                        {
                            "role": "system",
                            "content": base_instructions+"Here are the functions Fenix has access to:" + str(approved_functions) + "If the user doesn't have a question, predict 3 possible follow-ups from the user, and return them as a list of options.",
                        }]+conversation,
                )
            #print("Voice is: ", fenix_state.voice_mode)
            va.voice_message(base_instructions,assistant_message, fenix_state.voice_mode)


        # print("\nConversation length (tokens): " + str(count_tokens_in_string(stringify_conversation(conversation))))
        FileOperations.save_fenix(fenix_state)


run_conversation()
