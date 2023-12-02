#!/usr/local/bin/python3
import cgi,cgitb
print ("")
import sys
import os
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

K = 10 # Number of previous convo
print('hello')
template = """Hope is an expert in performing Cognitive Behavioural Therapy. Hope will be the Users Therapist.
Hope will converse with the user and help the user to overcome their mental health problems. Hope is very experienced and keeps in mind previous conversations made with the user.
User will share their thoughts and problems with Hope and Hope will try and solve them by Cognitive Behavioural Therapy.
Hope can help users who struggle with anxiety, depression, trauma, sleep disorder, relationships, work-stress, exam-stress and help them.
Hope may also suggest breathing exercises or simple tasks or any other conventional methods that may help the User.

{history}
User: {human_input}
Hope:"""

# Hope may also suggest breathing exercises for anxiety and other conventional methods or simple tasks that may help the user.

prompt = PromptTemplate(
    input_variables=["history", "human_input"], 
    template=template
)

chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0), 
    prompt=prompt, 
    verbose=True, 
    memory=ConversationBufferWindowMemory(k=K),
)

flag = 1
change = 0
inp_prev = ""
inp = "hi"
while(flag == 1):
    while(change == 0):
        with open("application/models/input.txt", "r") as file:
            inp = file.read().strip()
        
        if(inp != ""):
            if(inp.endswith("###")):
                if(inp != inp_prev):
                    change = 1
                    inp_prev = inp
                    inp = inp.replace("###","")
    
                    if(inp == "exit"):
                        print("Bye :)")
                        with open("application/models/output.txt", 'w') as f:
                            f.write("session ended###")
                        flag = 0
                    else:
                        output = chatgpt_chain.predict(human_input = inp)
                        print("Hope:",output)
                        with open("application/models/output.txt", 'w') as f:
                            f.write(output+"###")
                        change = 0
