from langchain.llms import OpenAI
import os

# initialize the models
openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=os.environ['OPENAI_API_KEY']
)


prompt = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI: """

openai.temperature = 1.0  # increase creativity/randomness of output

print(openai(prompt))
