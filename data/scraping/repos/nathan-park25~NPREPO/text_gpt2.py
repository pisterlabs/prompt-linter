import pickle
import os
from openai import ChatCompletion
import openai
# Set the API key
openai.api_key = "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"
RESET_FLAGS=0


cc=ChatCompletion()
def get_respone_gpt(cc,prompt,RESET_FLAGS):
    if RESET_FLAGS:
        # check if the chat history file exists, if not create it
        if not os.path.exists('chat_history.pkl'):
            with open('chat_history.pkl', 'wb') as f:
                default_dict={'role':'system','content':"You are a helpful assistant."}
                pickle.dump([default_dict], f)
        else: # if file exists, delete file and recreate it
            os.remove('chat_history.pkl')
            with open('chat_history.pkl', 'wb') as f:
                default_dict={'role':'system','content':"You are a helpful assistant."}
                pickle.dump([default_dict], f)
        # print CHAT RESET done
        # print("CHAT RESET DONE")
        text=""
    if not RESET_FLAGS:
        # getting list including dictionary from the chat_history.pkl file

        with open('chat_history.pkl', 'rb') as f:
            chat_history = pickle.load(f)
        tmp_chat_history=[]
        # appending the chat history to the prompt
        for chat in chat_history:
            tmp_chat_history.append(chat)
        prompt_chat={'role':'user','content':prompt}
        ## ading prompt_chat to the tmp_chat_history
        tmp_chat_history.append(prompt_chat)
        # getting ChatCompletion instance
        # cc = ChatCompletion()
        response = cc.create(
            # model="gpt-3.5-turbo",
            model="gpt-4",
            messages=tmp_chat_history,
        )
        text=response.choices[0]['message']['content']
        
        # put text to windwos Clipboard
        import pyperclip
        pyperclip.copy(text)
        tmp_assistant_response={'role':response.choices[0]['message']['role'],'content':response.choices[0]['message']['content']}
        # adding assistant response to the chat_history
        tmp_chat_history.append(tmp_assistant_response)
        
        # update chat_history.pkl file using tmp_chat_history
        with open('chat_history.pkl', 'wb') as f:
            # print(tmp_chat_history)
            pickle.dump(tmp_chat_history, f)
    # return text
    return text


## input 0 to reset the chat
prompt=r"""
Rewrite this in English:
Sumit에게, 
Muhammad Yousaf가 언급했듯이, 제가 Modelling을 리딩하고 있으며, "  RFTHP10288 Leonora Power Project - RFT Part A - General Information and RFT Conditions.pdf " 문서의 A2.7 (b) Powerfactory Modelling Requirements를 수행할수 있음을 알립니다. 

주어진 문서에 따르면, R0 Deliverable을 다음과 같다고 볼수 있습니다. 
- PowerFactory LoadFlow Model following Horizo Power Modelling Use Guideline 
- RMS Dynamic model (standard or generic)
- Voltage Control Strategy (High Level) 
- Configuration of generating units and balance of plants 

위의 역무로 보았을때 Preliminary Data가 충분히 제공된다면, notification of preferred status의 고지를 받은 이후 4주기간내에 충분히 해당 delibeverable를 Horizon Power에 제공할수 있다고 고려됩니다. 

다만 사전 준비에 따라 다음과 같은 정보와 데이터가 제공되었으면 합니다. 
- Estimation date of getting notification of preffered status 
- Preliminary generating system design(single line diagrams with detailed ratings/types in generating units and balance of plants) 

감사합니다. 





"""

prompt=prompt.strip()
get_respone_gpt(cc,"",1)
if prompt=="0":
    print(get_respone_gpt(cc,"",1))
else:
    print(get_respone_gpt(cc,prompt,0))




