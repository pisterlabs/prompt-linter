# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage, 






# 4 roles in chatmessage
# 1. Human Message : A message from human
# 2. AIMessage : Message from LLM 
# 3. System Message : Message from System
# 4. FunctionMessage: Message from function

if __name__ == "__main__":
    load_dotenv()
    # llm = OpenAI()
    chat_model = ChatOpenAI()
    text = 'Suggest me some  youtube channel name based on a company that creates youtube videos os AI'

    message = [HumanMessage(content=text)]
    print(chat_model.predict_messages(message))





