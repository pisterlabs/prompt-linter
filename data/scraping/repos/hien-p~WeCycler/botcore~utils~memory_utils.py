from typing import List, Dict
from langchain.memory import ConversationBufferMemory

def load_sample_qa():

    data = {
  "title": "Old phone",
  "product": "phone",
  "features": [
    "What is the screen size? 2.8 inches",
    "What is the RAM size? 512 MB",
    "What is the storage capacity? 4 GB",
    "What is the battery capacity? 1500 mAh",
    "Is there any malfunction or defect? yes",
    "What is the current physical condition of the product? excellent",
    "Is the product still under warranty? yes"
  ]
}

    ques = [i.split("?")[0] for i in data['features']]
    ans = [i.split("?")[1] for i in data['features']]
    return ques, ans


class QAMemory():

    def __init__(self, input_key: str):

        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key=input_key)
             
    
    def load_qa_to_memory(self, questions: List[str], answers: List[str]):
        for q, a in zip(questions, answers):
            self.memory.chat_memory.add_ai_message(q)
            self.memory.chat_memory.add_user_message(a)
        return True
    
    def import_qa(self, data: Dict):

        ques = [i.split("?")[0] for i in data['features']]
        ans = [i.split("?")[1] for i in data['features']]
        self.load_all(data['title'], ques, ans)
        return True

    def load_all(self, product: str, questions: List[str], answers: List[str]):
        self.load_product_context(product)
        self.load_qa_to_memory(questions, answers)
        print("Load done")
    
    def load_product_context(self, product: str):
        
        self.memory.chat_memory.add_user_message(f"I have this used {product} of mine. Please ask me some questions about it.")
