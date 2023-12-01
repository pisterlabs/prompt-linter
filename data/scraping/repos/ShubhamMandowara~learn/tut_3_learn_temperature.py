# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema import HumanMessage




if __name__ == "__main__":
    load_dotenv()
    chat_model = ChatOpenAI(temperature=1)
    text = 'Suggest me some  youtube channel name based on a company that creates youtube videos os AI'

    message = [HumanMessage(content=text)]
    print(chat_model.predict_messages(message))