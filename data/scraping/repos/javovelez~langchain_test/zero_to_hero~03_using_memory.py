import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv(dotenv_path='../.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ACTIVELOOP_TOKEN = os.getenv('ACTIVELOOP_TOKEN')


llm = OpenAI(model="text-davinci-003", temperature=0)
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)

# Start the conversation
conversation.predict(input="Tell me about yourself.")

# Continue the conversation
conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with data analysis?")

print("Display the conversation")
print(conversation)
#memory=ConversationBufferMemory(
# chat_memory=ChatMessageHistory(
#   messages=[
#       HumanMessage(content='Tell me about yourself.', additional_kwargs={}, example=False),
#       AIMessage(content=" Hi there! I'm an AI created to help people with their daily tasks. I'm programmed to understand natural language and respond to questions and commands. I'm also able to learn from my interactions with people, so I'm constantly growing and improving. I'm excited to help you out!", additional_kwargs={}, example=False),
#       HumanMessage(content='What can you do?', additional_kwargs={}, example=False),
#       AIMessage(content=" I can help you with a variety of tasks, such as scheduling appointments, setting reminders, and providing information. I'm also able to answer questions and provide advice. I'm always learning, so I'm sure I can help you with whatever you need.", additional_kwargs={}, example=False),
#       HumanMessage(content='How can you help me with data analysis?', additional_kwargs={}, example=False),
#       AIMessage(content=" I'm not currently able to help with data analysis, but I'm always learning and expanding my capabilities. I'm sure I'll be able to help you with data analysis in the future.", additional_kwargs={}, example=False)
#       ]
#   ),
#   output_key=None,
#   input_key=None,
#   return_messages=False,
#   human_prefix='Human',
#   ai_prefix='AI',
#   memory_key='history')
#   callbacks=None callback_manager=None verbose=True tags=None prompt=PromptTemplate(input_variables=['history', 'input'], output_parser=None, partial_variables={}, template='The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:', template_format='f-string', validate_template=True) llm=OpenAI(cache=None, verbose=False, callbacks=None, callback_manager=None, tags=None, client=<class 'openai.api_resources.completion.Completion'>, model_name='text-davinci-003', temperature=0.0, max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0, n=1, best_of=1, model_kwargs={}, openai_api_key='', openai_api_base='', openai_organization='', openai_proxy='', batch_size=20, request_timeout=None, logit_bias={}, max_retries=6, streaming=False, allowed_special=set(), disallowed_special='all') output_key='response' output_parser=NoOpOutputParser() return_final_only=True llm_kwargs={} input_key='input')
