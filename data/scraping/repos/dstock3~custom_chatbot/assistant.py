from gtts import gTTS # type: ignore
import subprocess
import openai, config
from system.systemCommands import system_commands
from system.customCommands import custom_commands
from system.processCommand import process_system_command, process_custom_command
from system.determineOS import determine_os
from system.format import strip_html_tags, get_display
from intel.emoji import extract_emojis
from intel.openai_call import apiCall
from intel.personalities import get_persona_list
from debug.debug_wrapper import debug
import re

#type hinting
from typing import Dict, Optional, List, Any, Union, Tuple
from debug.types import MessageDict, OpenAIObject, SystemMessage, TranscriptDict

openai.api_key = config.OPENAI_API_KEY
TRANSCRIPTION_MODEL = "whisper-1"
OS_NAME = determine_os()

def parse_transcript(text: str, operating_system: str, ai_name: str) -> Dict[str, Optional[str]]:
    #Parse a given transcript to identify if a command is present and if so, determine the command type
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    ai_name = ai_name.lower()

    all_commands = [
        ("system", system_commands[operating_system]),
        ("custom", custom_commands.keys())
    ]

    for command_type, commands in all_commands:
        for cmd in commands:
            if ai_name + " " + cmd in text:
                return {"command": cmd, "command-type": command_type}
            elif command_type == "custom":
                for alt_cmd in custom_commands[cmd]['alt']: # type: ignore
                    if ai_name + " " + alt_cmd in text:
                        return {"command": cmd, "command-type": command_type}

    return {"command": None, "command-type": None}

def process_command(
    command: Optional[str], 
    command_type: Optional[str], 
    messages: List[MessageDict], 
    file: str, 
    ai_name: str,
) -> Tuple[List[MessageDict], bool]:
    # Processes a recognized command and updates the message list accordingly.
    if command is not None:
        if command_type == "custom":
            process_custom_command(command, custom_commands, messages, file, ai_name)
        elif command_type == "system":
            process_system_command(command, system_commands[OS_NAME])
        return messages, True
    else:
        user_message = {"role": "user", "content": file}
        messages.append(user_message)

        return messages, False

def process_input(
    is_audio: bool, 
    file: str, 
    messages: List[MessageDict], 
    ai_name: str
) -> Tuple[List[MessageDict], bool, Optional[str]]:
    """
    Processes user input to execute commands or update the conversation.
    Handles both audio and text-based input.
    """
    if is_audio:
        with open(file, "rb") as f:
            transcript = openai.Audio.transcribe(TRANSCRIPTION_MODEL, f)
            text = transcript["text"]
    else:
        text = file

    command_info = parse_transcript(text, OS_NAME, ai_name)
    command = command_info["command"]
    command_type = command_info["command-type"]
    messages, is_command = process_command(command, command_type, messages, text, ai_name)

    return messages, is_command, command

def derive_model_response(
    model: str, 
    messages: List[MessageDict], 
    temperature: float, 
    ai_name: str
) -> OpenAIObject:
    """
    Generates a model response based on the conversation history and specified model.
    Handles gpt-3.5-turbo, gpt-4, and other models.
    """
    if (model == "gpt-3.5-turbo") or (model == "gpt-4") or (model == "gpt-4-32k") or (model == "gpt-4-1106-preview"):
        response = apiCall(messages, 500, temperature, True)
    else:
        conversation_history = "".join(f"{message['role'].capitalize()}: {message['content']}\n" for message in messages)
        prompt = f"{conversation_history}{ai_name}:"
        completion_response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=["\n"],
            temperature=temperature,
        )
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": completion_response.choices[0].text.strip(),
                    }
                }
            ]
        }
    return response

def generate_response(
    messages: List[MessageDict], 
    temperature: float, 
    model: str, 
    ai_name: str, 
    command: Optional[str]
) -> Tuple[SystemMessage, List[MessageDict], Optional[Any]]:
    '''
    Takes in the messages so far, model-related parameters, and a command to generate a response from the AI model. It returns the generated system message, the updated messages list, and an optional display value.
    '''
    emoji_check = None
    display = None
    command_flag = False
    
    if command == "remember when" or command == "make a note":
        command_flag = True
        '''
        This solution is very convoluted. Maybe break this up into separate functions?
        There will be other instances when derive_model_response needs to be circumvented, e.g. if I create another custom command that calls the API itself.
        '''

        for message in messages:
            if message["role"] == "assistant":
                emoji_check, cleaned_text = extract_emojis(message["content"]) # type: ignore
                if emoji_check:
                    system_message, display = get_display(emoji_check, cleaned_text)
                else:
                    system_message = message
    else:    
        response = derive_model_response(model, messages, temperature, ai_name)
        emoji_check, cleaned_text = extract_emojis(response["choices"][0]["message"]["content"]) # type: ignore

        if emoji_check:
            system_message, display = get_display(emoji_check, cleaned_text)
        else:
            system_message = response["choices"][0]["message"] # type: ignore

    if command_flag:
        return system_message, messages, display
    else:
        messages.append(system_message)
        return system_message, messages, display

def convert_to_audio(system_message: SystemMessage) -> None:
    # This function takes in the system message and converts it to audio. It uses the gTTS library to convert the text to speech.
    content = strip_html_tags(system_message['content'])
    tts = gTTS(content, tld='com.au', lang='en', slow=False)
    tts.save('output.mp3')

    # Use subprocess to launch VLC player in a separate process
    subprocess.Popen(['vlc', '--play-and-exit', 'output.mp3', 'vlc://quit', '--qt-start-minimized'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def create_chat_transcript(
    messages: List[MessageDict], 
    isCommand: bool,
    command: Optional[str],
    ai_name: str
) -> List[TranscriptDict]:
    '''
    Creates a chat transcript from a list of messages. If isCommand is True, the 'user_message' in the transcript is set to the 'command' value.
    '''
    
    chat_transcript: List[TranscriptDict] = []
    user_message = ''
    assistant_message = ''
    prev_command_set = None

    for index, message in enumerate(messages):
        if message['role'] == 'user':
            prev_command_set = parse_transcript(message['content'], OS_NAME, ai_name)
            
            if prev_command_set["command"] is not None:
                user_message += prev_command_set['command']
            else:
                user_message += message['content']
        elif message['role'] == 'assistant':
            assistant_message += message['content']

            if isCommand and index == len(messages) - 1:
                if command is not None:
                    chat_transcript.append({'user_message': command, 'assistant_message': assistant_message})
            else:
                chat_transcript.append({'user_message': user_message, 'assistant_message': assistant_message})

            user_message = ''
            assistant_message = ''
    return chat_transcript

def main(
    user: Optional[Dict[str, Any]], 
    isAudio: bool, 
    input: Optional[str], 
    existing_messages: Optional[List[MessageDict]] = None,
    user_info: Optional[List[str]] = None,
    prior_conversations: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[TranscriptDict], Optional[Any]]:
    #Main function processes the user input and generates an assistant response based on the user's settings and personality.

    if user is not None:
        name = user['name']
        voice_command = user['voice_command']
        voice_response = user['voice_response']
        model = user['model']
        personality = user['personality']
        ai_name = user['system_name']

    chat_transcript: List[TranscriptDict] = []
    display = None

    all_personality_options = get_persona_list()
    personality_data = all_personality_options.get(personality)

    if input is not None:
        try:
            if existing_messages:
                messages = personality_data["messages"] + existing_messages
                messages, isCommand, command = process_input(isAudio, input, messages, ai_name)
            else:
                messages, isCommand, command = process_input(isAudio, input, personality_data["messages"], ai_name)

            if messages:
                name_message = {
                    "role": "system",
                    "content": f"Hello, I am {name}. You are my AI assistant named {ai_name}. "
                        f"Please remember to address yourself as {ai_name} and address me as {name}. "
                        f"Additionally, if you provide any code snippets, mark the beginning with '%%%CODE_START%%%' "
                        f"and the end with '%%%CODE_END%%%' After that, let me know the specific language being used within a separate block. Make sure the language is accurate. Mark the beginning with '%%%LANGUAGE_START%%%' and the end with '%%%LANGUAGE_END%%%'."
                        f"Here is {name}'s user info. If any of this is relevant, feel free to mention it:"
                        f"{user_info if user_info else ''}"
                        f"Here is a prior conversation that may be relevant to your current conversation:"
                        f"{prior_conversations if prior_conversations else ''}"
                }
                messages.insert(0, name_message)

                system_message, messages, display = generate_response(messages, personality_data["temperature"], model, ai_name, command)
                if voice_response:
                    convert_to_audio(system_message)
                chat_transcript = create_chat_transcript(messages, isCommand, command, ai_name)

        except Exception as e:
            chat_transcript.append({'user_message': '', 'assistant_message': "An error occurred: {}".format(str(e))})

    return chat_transcript, display