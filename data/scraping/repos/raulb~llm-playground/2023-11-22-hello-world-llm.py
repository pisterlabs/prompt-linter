from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

client = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=100)


reply = client(
  # this is one prompt. To carry out the message to another prompt, we'll need to do it differently
  [ 
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is a large language model?"), # role: user
  ]
)

print(reply)
