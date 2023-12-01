from langchain.prompts import PromptTemplate
import os
from langchain.chains import LLMChain
import re
from langchain.llms import OpenAIChat
import comfy
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys
from dotenv import load_dotenv
load_dotenv()
# Put OPENAI_API_KEY in .env file

llm = OpenAIChat(temperature=0.8,model_name='gpt-3.5-turbo')
prompt = PromptTemplate(
    input_variables=["convo"],
    template="""

    Conversation:
    {convo}"""
    )
ai = LLMChain(llm=llm, prompt=prompt, verbose=True)


def Chat_AI(convo):
    convo = ai_gf.run(convo)
    EMOJI_PATTERN = re.compile(
    "(["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "])"
    )
    convo = re.sub(EMOJI_PATTERN, "", convo)
    return  convo


def SnapGF(c, name, driver):
    convo = c.text
    convo = convo.replace(name.upper(), "Caleb")
    replace_list = [("Open the mobile app to view", ""), ("Opened", ""), ("New Snap", ""), ("KODI WALLS IS USING SNAPCHAT FOR WEB", ""), ("YOU ARE USING SNAPCHAT ON WEB", "")]
    for pattern, replace in replace_list:
        convo = convo.replace(pattern, replace)

    convo = convo.split("\n")
    convo_dict = {}
    for i in range(len(convo)):
        if convo[i] == "ME":
            convo_dict[f"{convo[i]}"] = convo[i + 1]
        elif convo[i] == "Caleb":
            convo_dict[f"{convo[i]}"] = convo[i + 1]
    if "ME" not in convo_dict.keys():
        convo_dict["ME"] = ""
    convo = f"ME: {convo_dict['ME']}\nCaleb: {convo_dict['Caleb']}"
    print(convo)

    if "send image" in convo_dict["Caleb"].lower():
        text_input = convo_dict["Caleb"]
        comfy.get_gen(text_input, name)
        time.sleep(60)
        try:
            driver.find_element(By.XPATH, '//input[@type="file"]').send_keys(f"/home/charles/Dev/code/python/fsb/output/{name}_00001_.png")
        except:
            print("Error generating image")
        try:
            textbox = driver.find_element(By.XPATH, '//*[@role="textbox"]')
            textbox.send_keys("Sent Image")
            textbox.send_keys(Keys.RETURN)
        except:
            print("Error sending image")

        if os.path.exists(f"/home/charles/Dev/code/python/fsb/output/{name}_00001_.png"):
            time.sleep(2)
            os.remove(f"/home/charles/Dev/code/python/fsb/output/{name}_00001_.png")
        else:
            print("The file does not exist")



    response = Chat_AI(convo)
    replace_list = [("ME", ""), ("Caleb", ""), ("\n", " "), (":", "")]
    for pattern, replace in replace_list:
        response = response.replace(pattern, replace)
    return response
    
