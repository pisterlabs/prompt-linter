from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from dotenv import load_dotenv
load_dotenv()
conversation_with_summary = ConversationChain(
    llm=OpenAI(temperature=0),
    verbose=True,
    memory=ConversationSummaryBufferMemory(llm=OpenAI(), max_token_limit=30),
)
openai = OpenAI(streaming=True)
