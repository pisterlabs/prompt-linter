import os
from langchain.chat_models import JinaChat
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

os.environ["JINACHAT_API_KEY"] = "UkeLsEM8paN7Tr7CnGTf:6abdd62ddf5aa09291ae885d3bb65f037358846ea039269d37b8b2b967ec5f23"

chat = JinaChat(temperature=0.9)

def feed (prompt):
  sys_msg = """Let's suppose you are a fashion assistant. Generate some fashion recommendations after
            reading through some of my characteristics. I am a 20 year old guy who loves to dress
            subtle. My previous purchases are a pair of Nike Air Jordans, H&M plain t-shirts,
            Baggy jeans. I love lighter colours like beige, cream, and sky blue. Now, answer relevantly
            and straight to the point in less than 50 words """

  print("System Message: ", sys_msg)
  print("Human Message: ", prompt)

  messages = [
    SystemMessage(content="Let's suppose you are a fashion assistant. Generate some fashion recommendations after "
                          "reading through some of my characteristics. I am a 20 year old guy who loves to dress "
                          "subtle. My previous purchases are a pair of Nike Air Jordans, H&M plain t-shirts, "
                          "Baggy jeans. I love lighter colours like beige, cream, and sky blue. Now, answer relevantly "
                          "and straight to the point in less than 50 words"),
    HumanMessage(content=prompt)
  ]

  print("From Jinachat: ", chat(messages).content)

if __name__ == "__main__":
  prompt = input("Enter prompt: ")
  feed(prompt)