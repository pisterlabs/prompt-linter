import os
import openai
from executor import executeScript, voiceAnswer
from helpers import validations

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def promptDataToGPT(action_prompt: str, context: dict[str, str]):

        is_code = False
        external_code = False

        promptGuide = promptTopicToGPT(action_prompt, context)
        promptGuide[1] = promptGuide[1].replace("Tag: ", "")

        if not validations.validTopic(promptGuide[1]):
                return

        promptGuide[1] = promptGuide[1][:3]

        if promptGuide[1] == "[C]":
                is_code = True
                external_code = True
        else:
                try:
                        if promptGuide[0].find("python") != -1:
                                is_code = True
                                promptGuide[0] = promptGuide[0] + "\n" + getPrompts("python_structure/python_code")
                except:
                        if context['VOICE']:
                                voiceAnswer.main("Please give me more context.", "ERROR_OCURRED")
                        return

        if promptGuide[1] != "[C]":
                promptGuide[0] = getPrompts("python_structure/required") + "\n" + promptGuide[0]

        try:
                message = openAICall(promptGuide[0], action_prompt, context)
                print(bcolors.OKCYAN + "[COOPER]", "- Received")
                if promptGuide[1] == "[C]":
                        code = message.split("%%%")[1]
                        file_metadata_prompt = getPrompts("code_prompt/file_metadata")
                        metadata = openAICall(file_metadata_prompt, code, context)
                        name = metadata.split("%%%")[1]
                        saveCode(code, name)
                        return

                global notes
                global name_file
                try:
                        notes = message.split("&&&")[1]
                except:
                        pass
                if is_code:
                        parts = message.split("%%%")
                        code = parts[1]
                        executeScript.main(parts[1], notes)
                                
                else:   
                        if context['VOICE'] and not external_code:
                                message_voice = message.replace("$$$", "")
                                message_voice = message_voice.replace("%%%", "")
                                voiceAnswer.main(message_voice, "GENERAL_TOPIC")
                return
        
        except Exception as error:
                print(error)

def promptTopicToGPT(user_action, config):

    try:
        topic = openAICall(getPrompts("topic_selector"), user_action, config)
        topic = topic.replace("Tag: ", "")
        topic_prompt = getPrompts(f"topic_prompts/{topic}")
        return [topic_prompt, topic]

    except Exception as e:
        voiceAnswer.main("TOPIC ERROR", "TOPIC_DETECTOR_ERROR")
        return [None, None]
    
def getPrompts(file_path):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        settings_file_path = os.path.join(dir_path, '../settings', f"{file_path}.txt")
        try:
                with open(settings_file_path, "r") as Topic:
                    getText = Topic.readlines()
                    Topic.close()
                    return " ".join(getText)
        except Exception as e:
                print(e)

def saveCode(code: str, title: str):
    with open(f"./{title}", "w") as f:
        f.write(code)
        print(bcolors.OKCYAN + "\n [COOPER]", "- Code saved")
        f.close()

def openAICall(promptGuide, actionPrompt, context):
        NUMBER_RESPONSES = 1 if len(actionPrompt) < 150 else  2
        try:
                response = openai.ChatCompletion.create(
                        model=context['MODEL'],
                        max_tokens=context['MAX_TOKENS'],
                        temperature=context['TEMPERATURE'],
                        messages=[{"role":"system", "content": promptGuide},
                        {"role":"user", "content":f"[{actionPrompt}]"}]
                        ,
                        n=NUMBER_RESPONSES
                )
        except Exception as error:
                if context['VOICE']:
                        voiceAnswer.main("Network error.", "ERROR_CALL")
                print(error)
                return

        return response.choices[0].message.content