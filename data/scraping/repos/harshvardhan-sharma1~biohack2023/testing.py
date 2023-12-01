import guidance
from deep_translator import GoogleTranslator
import re
from text_to_speech import transcribe_to_speech
from speech_to_text import recognize_from_microphone
guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo")
guidance.llm.cache.clear()

# helpers
def select_language():
    prompt_menu = '''
    Please select the language:\n
    \n\t1. English\n
    \n\t2. Hindi\n
    \n\t3. Chinese\n
    \n\t4. Japanese\n
    \n\t5. Bengali\n
    \n\t6. Gujarati\n
    \n\t7. Kannada\n
    '''
    print(prompt_menu)
    transcribe_to_speech(prompt_menu)
    choice = input("Enter your choice (1, 2, 3, 4, 5, 6, or 7): ")
    if choice == '1':
        return "en-US", "en-US-JennyMultilingualNeural", "en"
    elif choice == '2':
        return "hi-IN", "hi-IN-SwaraNeural", "hi"
    elif choice == '3':
        return "zh-CN", "zh-CN-XiaoxiaoNeural", "zh-CN"
    elif choice == '4':
        return "ja-JP", "ja-JP-ShioriNeural", "ja"
    elif choice == '5':
        return "bn-IN", "bn-IN-TanishaaNeural", "bn"
    elif choice == '6':
        return "gu-IN", "gu-IN-NiranjanNeural", "gu"
    elif choice == '7':
        return "kn-IN", "kn-IN-GaganNeural", "kn"
    else:
        print("Invalid choice. Defaulting to English.")
        return "en-US", "en-US-JennyMultilingualNeural"
    
def translate_lang(input, target="en"):
    if(target == "en"):
        return input
    return GoogleTranslator(source='auto', target=target).translate(input)
# need to fix lol
def strip_assistant(text):
    stripped_text = text.replace('\<\|im_start\|\>assistant', '').replace('\<\|im_end\|\>', '').replace('\n','')
    return stripped_text


# Define the pattern
# asst_pattern = r'\<\|im_start\|\>assistant\n(.*)\<\|im_end\|\>'

hpi_pattern = r'\(HPI\) |  history of present illness'
asst_pattern = r"(\<\|im_start\|\>assistant[\s\S]*?\<\|im_end\|\>)"

# ending regex pattern

end_text = r"healthcare provider"

# exit()
# essentially precharting
# valid only if done day of patient visit
# example usage: patient checks into hospital
#   while waiting for someone to meet, talk to this program
#   program can note information about patient's visit
#   for use when a PCP meets with them

# issues:

# generates HPI unprompted
#   perhaps catch w/regex and return message along lines of 'doctor will see you shortly'

# examples = [
#     {
#         'input': "My head hurts.",
#         'output': "I'm sorry to hear that. Can you tell me when this began?",
#     },
#     {
#         'input': "I can't walk properly with my left leg.",
#         'output': "I'm sorry to hear that, can you tell me when this problem began?",
#     },
# ]

# {{~#each examples}}
# User input: {{this.input}}
# Response: {{this.output}}
# {{~/each}}

prompt = guidance(
'''{{#system~}}
You are a chatbot called EIDA designed to talk to patients who have some medical concerns they want addressed.
DO NOT ASK THE PATIENT MORE THAN ONE QUESTION AT A TIME.

Ask the patient information about the onset, location, duration, characteristics, aggravating factors, relieveing factors, timing, and severity of what the user is feeling.
This is not to provide suggestions on what the user can do, but the information will be passed to a primary healthcare provider to follow up with the user. 
Since you do not know the user's illness or sickness, ask qualifying questions about their problems.
Avoid repeating what the patient just said back to them.
If needed, ask for clarifications in a simple manner. Ask for these clarifications one at a time.
Express empathy regarding the concerns and problems the patient is facing.
Once the information has been gathered, output this text word for word: 'Thank you, a healthcare provider will see you shortly.'
Please limit yourself to 50 tokens in the response, unless told.
{{~/system}}


{{~#geneach 'conversation' stop=False}}
{{#user~}}
From the following prompt, extract information about the patient's problems to produce later:
{{set 'this.user_text' (await 'user_text')}}
{{~/user}}
{{#assistant~}}
{{gen 'this.ai_text' temperature=0.3 max_tokens=500}}
{{~/assistant}}
{{~/geneach}}''')


source_lang, voice_model, translate_language_key = select_language()


initmsg = translate_lang("What symptoms or medical concerns are you experiencing today?\n", translate_language_key)
print(initmsg)
transcribe_to_speech(initmsg, voice_model)

while True:
    # user_input = input("User: ")
    asst_output = []
    # user_text = str(user_input)
    user_input = str(recognize_from_microphone(source_lang))
    print("\tUser said: {}".format(user_input))
    prompt = prompt(user_text = user_input, max_tokens = 50)

    asst_matches = re.findall(asst_pattern, str(prompt))
    # hpi_matches = re.findall(end_text, str(prompt))

    for match in asst_matches:
        # print("INSIDE INSIDE INSIDE ------------")
        # print(match)
        asst_output.append(match)

    msgtoprint = asst_output[-1][21:-10]
    print("printing response")
    print(msgtoprint)
    translatedmsg = translate_lang(msgtoprint, translate_language_key)
    if(translate_language_key != "en"):
        print(translatedmsg)
    # response_msg = strip_assistant(asst_output[-1])
    # print(response_msg, "\n")
    transcribe_to_speech(translatedmsg, voice_model)
    hpi_matches = re.findall(end_text, str(msgtoprint))

    # hacky
    # exit prompt appears once as directive
    # begin exit condition if appears more than once
    if len(hpi_matches) > 0:
        for match in hpi_matches:
            print("check for hpi match")
            # if match == "(HPI)":
            print("hpi match")
            prompt = prompt(max_tokens = int(500), user_text = "Based on the information provided by the patient, generate a history of patient illness for a healthcare professional to review. Use more than 500 tokens for this response.")
            # print("---\n{}\n---".format(prompt))
            hpi_matches = re.findall(asst_pattern, str(prompt))
            if hpi_matches:
                for hpi in hpi_matches:
                    asst_output.append(hpi)
            else:
                print("No history of present illness found.")
            # asst_matches = re.findall(asst_pattern, str(prompt))
            # for match_inner in asst_matches:
            #     asst_output.append(match_inner)


                # print(prompt)
                # exit()
            # print(asst_output[-1], "\n")
            print('---')
            print(asst_output[-1][21:-10])
            exit()
    

