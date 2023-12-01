import constants
import os
os.environ["OPENAI_API_KEY"] = constants.openapi_key


from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex, LLMPredictor, ServiceContext, GPTListIndex
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig, LlamaIndexTool, GraphToolConfig
from langchain.chat_models import ChatOpenAI
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform


index = GPTVectorStoreIndex.load_from_disk(f'Clubs.json')
r = index.query("What is ACM?")

print(r)
