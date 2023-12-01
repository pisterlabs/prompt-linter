import openai 
import os
import json
import tiktoken
# import spatial.distance.cosine
from scipy import spatial

# Set your secret API key
openai.api_key = "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"

def send_message_to_chatgpt(message_input, role=None, model="gpt-3.5-turbo", temperature=0, include_beginning=True, is_list=False):

    encoder = tiktoken.encoding_for_model(model)
    max_tokens = 4050 if model == "gpt-3.5-turbo" else 8150 if model == "gpt-4" else None

    if not is_list:
        cleaned_message = message_input.replace("'", "").replace('"', '').replace("’", "")
        truncated_message = truncate_single_message(cleaned_message, encoder, max_tokens)
        message_input = [{"role": role, "content": truncated_message}]
    else:
        message_input = truncate_messages(message_input, encoder, max_tokens)

    final_message = message_intro + message_input if include_beginning else message_input

    response = openai.ChatCompletion.create(
        model=model,
        messages=final_message,
        temperature=temperature,
    )

    response_content = response.choices[0].message.content
    return response_content

def truncate_messages(messages, encoder, max_tokens):
    truncated_messages = []
    current_tokens = 0
    for message in messages:
        content = message["content"]
        content_tokens = encoder.encode(content)
        current_tokens += len(content_tokens)

        if current_tokens > max_tokens:
            excess_tokens = current_tokens - max_tokens
            truncated_content = encoder.decode(content_tokens[:-excess_tokens])
            message["content"] = truncated_content
            current_tokens = max_tokens

        truncated_messages.append(message)

        if current_tokens == max_tokens:
            break

    return truncated_messages

def truncate_single_message(message, encoder, max_tokens):
    message_tokens = encoder.encode(message)

    if len(message_tokens) > max_tokens:
        truncated_message = encoder.decode(message_tokens[:max_tokens])
        return truncated_message
    else:
        return message
### Prerequisites

message_intro = [
    {"role": "system", "content": "You are a composer and sound designer. Your objective is to create a music or soundscape that matches the description given by the user. You must only respond in the format specified by the user, you won't add anything else to your responses."},
    ]

# dictionary. First key is "simple", second key is "complex"
example_prompts = {"prompt1": "A sad slow song","prompt2": "An irish folk melody", "simple3": "gloomy thunder" , "simple4": "sped-up pink panther", "complex1": "mozart, epic strings, heroic and fast"}

### Main Code

# main
def temperature_evaluation():
    response_prompt_enhancer_melody = "dynamic, happy, virtuosic jumping in speed. The dynamic and upbeat melody would likely be composed in a major key, with a lively tempo and quick, intricate passages that demonstrate the virtuosic abilities of the performers. The melody would be characterized by a sense of joy and excitement, with a constant forward momentum that propels the listener forward. As the melody progresses, the tempo and intensity would increase, with each instrumental performance building upon the last to create a sense of dynamic energy. The instrumentation would likely feature a range of instruments, with each performer showcasing their individual virtuosity in quick and complex passages that showcase their technical abilities. Overall, the composition would be a celebration of the joy and energy of music, with each note and performance contributing to a sense of excitement and exhilaration."

    message_melody = [                    
        {"role": "user", "content": f"'{response_prompt_enhancer_melody}' Create MIDI files that match the description of the melody. Use the MIDIUtil Python library and only respond with a list of tuples, where each tuple represents a single note in the format (pitch, velocity, duration). The pitch is an integer value between 0 and 127, representing the note's frequency according to the MIDI standard. The velocity is an integer value between 0 and 127, representing the note's intensity or loudness, with higher values indicating a louder note. The duration is a positive integer representing the length of the note in beats. Please provide a full melody using this format: melody = [(PITCH_1, VELOCITY_1, DURATION_1), (PITCH_2, VELOCITY_2, DURATION_2), (PITCH_3, VELOCITY_3, DURATION_3), ...]. Replace the placeholders (PITCH_n, VELOCITY_n, DURATION_n) with appropriate integer values for your melody."},
        ]
    
    temperature = 0.1
    model = "gpt-3.5-turbo"
    for i in range (2):
        while temperature < 1.6:
            response_melody = send_message_to_chatgpt(message_melody, role="user", model=model, temperature=temperature, include_beginning=True, is_list=True)

            # write response to txt file named after first 5 chars of response_prompt_enhancer_melody and temperature and model
            with open(f"evaluation/melody_{response_prompt_enhancer_melody[:5]}_{temperature}_{model}.txt", "w", encoding="utf-8") as f:
                f.write(response_melody)

            print(f"melody_{response_prompt_enhancer_melody[:5]}_{temperature}_{model}.txt")
            temperature += 0.4
            
        
        model = "gpt-4"
        temperature = 0.1


    return

temperature_evaluation()