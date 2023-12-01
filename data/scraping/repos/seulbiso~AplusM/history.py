import json
from flask import session
from langchain.schema import messages_from_dict, messages_to_dict, HumanMessage, AIMessage


class SetMessage:
    def __init__(self, i, o, mode, conv):
        self.message = self.set_memory(i, o) if mode == "mode_docs" else conv.memory.chat_memory.messages  # 수정 필요

    def set_memory(self, i, o):
        res = []
        res.append(HumanMessage(content=i, additional_kwargs={}))
        res.append(AIMessage(content=o, additional_kwargs={}))
        return res

class SaveHistory(SetMessage):

    def __init__(self, conversation_chain, conversation_number, mode, persona, user_info, i, o):
        super().__init__(i, o, mode, conversation_chain)
        self.memory = messages_to_dict(self.message)
        self.mode = mode
        self.persona = persona
        self.user_info = user_info
        self.conversation_number = conversation_number

    def convert_message(self, message:dict):
        res = {}
        res["type"] = message["type"]
        res["content"] = message["data"]["content"]
        res["additional_keyword"] = ', '.join(message["data"]["additional_kwargs"])
        return res
    
    def message_to_record(self, type):
        message = {}
        session_number = "xxx"   # 수정 필요
        conversation_number = str(self.conversation_number)
        mode = self.mode
        persona = self.persona
        user_info = json.dumps(self.user_info, ensure_ascii=False)
        
        if type == "Human":
            message = self.convert_message(self.memory[-2])
        elif type == "AI":
            message = self.convert_message(self.memory[-1])

        message.update({"session_number":session_number, "conversation_number":conversation_number, "mode":mode, "persona":persona, "user_info":user_info})
  
        return message
    
    def get_record(self):
        res = []
        human_message = self.message_to_record("Human")
        ai_message = self.message_to_record("AI")
        res.append(human_message)
        res.append(ai_message)

        return res