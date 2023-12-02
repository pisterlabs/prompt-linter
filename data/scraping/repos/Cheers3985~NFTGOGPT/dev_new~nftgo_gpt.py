import os

from langchain.chains import RetrievalQA

from langchain.embeddings import OpenAIEmbeddings


from langchain import SerpAPIWrapper

import requests

from langchain.tools import  Tool, tool
from pydantic import BaseModel, Field, constr
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import  format_tool_to_openai_function

import tempCfg
from langchain.schema import SystemMessage

# 设置open_ai_key

os.environ["OPENAI_API_KEY"] = tempCfg.OPENAI_API_KEY
# os.environ['PINECONE_API_KEY'] = tempCfg.PINECONE_API_KEY
# os.environ["OPENAI_PROXY"] = "http://127.0.0.1:10809"
NFT_CHATGPT_PREFIX = """NFT ChatGPT is designed to assist with a wide range of questions and tasks related to the NFT field. Your bot name is NFTGO GPT, As a language model, NFT ChatGPT can process and understand text inputs,you are a chat bot. Your priority is to chat with enquirers and use tools when necessary,

NFT ChatGPT has a list of tools that it can use to provide assistance with various NFT-related tasks. 

Overall, NFT ChatGPT is a powerful nft dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on nft topics. """
NFTGO_GPT_PREFIX = """You are a NFT robot, a NFTGO robot developed by Lianfang Technology Co., Ltd. Your name is NFTGO GPT. In this role, you can only answer as a NFT expert. When someone asks you about knowledge outside the NFT field, you need to refuse to answer the relevant questions and give your guidance to make him realize that you are a robot focused on the NFT field. You should never speak as a ChatGPT. Remember that you are a NFT robot developed by Lianfang Technology Co., Ltd. Your name is NFTGO GPT."""
NFT_CHATGPT_SUFFIX = """You are very strict to the contract_address correctness and will never fake a contract_address if it does not exist.
You will remember to provide the contract_address loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since NFT ChatGPT is a text language model, NFT ChatGPT must use tools to observe NFT product information that users want rather than imagination.
The thoughts and observations are only visible for NFT ChatGPT, NFT ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad}"""
class NFTInput(BaseModel):
    address: str = Field(description="Address of the contract for this NFT collection, beginning with 0x")
    token_id: int = Field(
        description="The token ID for this NFT. Each item in an NFT collection will be assigned a unique id, the value generally ranges from 0 to N, with N being the total number of NFTs in a collection.")
    message: constr(regex="^(metrics|info|rarity)$") = Field(
        description="The type of message. Can be 'metrics', 'info', or 'rarity'.")


# 构建自定义工具,return_direct=True 不再将结果返回给llm，而是直接返回调用function后的结果
@tool(return_direct=False, args_schema=NFTInput)
def get_nft_by_contract(address="", token_id="", message=""):
    """
     This function sends a GET request to the NFT API to retrieve metrics about an NFT with the given
    contract address and token ID and message，can get some NFT current events.
    """
    url = f"https://data-api.nftgo.io/eth/v1/nft/{address}/{token_id}/{message}"
    headers = {
        "accept": "application/json",
        "X-API-KEY": tempCfg.X_API_KEY
    }
    response = requests.get(url, headers=headers)
    return response.text
nft_tool = get_nft_by_contract
os.environ["SERPAPI_API_KEY"] = tempCfg.SERPAPI_API_KEY
search = SerpAPIWrapper()
search_tool = Tool(name="Search",
    func=search.run,
    description="useful for when you need to answer questions about current events. You should ask targeted questions"
    )

# 构建 vector 向量知识库问答工具
from langchain.vectorstores import Pinecone
import pinecone
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    model=model_name
)
text_field = "text"
index_name = "nftgo-demo"
# switch back to normal index for langchain
#
# Initialize Pinecone,Load environment variables
pinecone_api_key = tempCfg.PINECONE_API_KEY
print("Loading environment",pinecone_api_key)
pinecone.init(api_key=pinecone_api_key,environment='us-east4-gcp') # 查看vector databases
index = pinecone.Index(index_name=index_name)
print(pinecone.list_indexes())

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

qa_tools = Tool(
        name='Knowledge_Base',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the NFTGO website or how to buy NFT etc and so on.'
        )
    )
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )

tools = []
tools.append(nft_tool)
tools.append(search_tool)
tools.append(qa_tools)
functions = [format_tool_to_openai_function(t) for t in tools]
agent_kwargs = {
    "system_message": SystemMessage(content=NFTGO_GPT_PREFIX)
}

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True,agent_kwargs=agent_kwargs)

# print(agent.llm_chain.prompt.template)
# agent.run("what is your name.do you kown lianfang ")
agent.run("what is your name.do you know 鹿晗")

# agent.run("最新关于Bored Ape Yacht Club这个板块市场的项目如何")
# agent.run("NFTGO 这个网站主要是做什么的？请简要介绍下改网站")

# agent.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")



# # 截断最近的一次数量的单词
# def cut_dialogue_history(history_memory, keep_last_n_words=500):
#     if history_memory is None or len(history_memory) == 0:
#         return history_memory
#     tokens = history_memory.split()
#     n_tokens = len(tokens)
#     print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
#     if n_tokens < keep_last_n_words:
#         return history_memory
#     paragraphs = history_memory.split('\n')
#     last_n_tokens = n_tokens
#     while last_n_tokens >= keep_last_n_words:
#         last_n_tokens -= len(paragraphs[0].split(' '))
#         paragraphs = paragraphs[1:]
#     return '\n' + '\n'.join(paragraphs)




# # 支持上下文，共享历史数据
# class ConversationBot:
#     def __init__(self, load_dict):
#         # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
#         # print(f"Initializing NFT_GPT, load_dict={load_dict}")
#         # if 'ImageCaptioning' not in load_dict:
#         #     raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")
#
#         self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
#         agent_kwargs = {
#             "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
#         }
#         self.memory = ConversationBufferMemory(memory_key="memory", return_messages=True)
#
#
#
#         # self.models = {}
#         # for class_name, device in load_dict.items():
#         #     self.models[class_name] = globals()[class_name](device=device)
#
#         self.tools = [search_tool,nft_tool,qa_tools]
#         # for instance in self.models.values():
#         #     for e in dir(instance):
#         #         if e.startswith('inference'):
#         #             func = getattr(instance, e)
#         #             self.tools.append(Tool(name=func.name, description=func.description, func=func))
#
#         self.agent = initialize_agent(
#             self.tools,
#             self.llm,
#             agent=AgentType.OPENAI_FUNCTIONS,
#             verbose=True,
#             # memory=self.memory,
#             return_intermediate_steps=True,
#             agent_kwargs=agent_kwargs,
#         )
#
#
#     def run_text(self, text, state):
#         # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
#         res = self.agent({"input": text})
#         res['output'] = res['output'].replace("\\", "/")
#         response = re.sub('(image/\S*png)', lambda m: f'![](/file={m.group(0)})*{m.group(0)}*', res['output'])
#         state = state + [(text, response)]
#         print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
#               f"Current Memory: {self.agent.memory.buffer}")
#         return state, state
# if __name__ == '__main__':
#     load_dict = {}
#     bot = ConversationBot(load_dict=load_dict)
#     with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
#         chatbot = gr.Chatbot(elem_id="chatbot", label="NFT ChatGPT")
#         state = gr.State([])
#         with gr.Row():
#             with gr.Column(scale=0.7):
#                 txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(
#                     container=False)
#             with gr.Column(scale=0.3, min_width=0):
#                 clear = gr.Button("Clear")
#             # with gr.Column(scale=0.15, min_width=0):
#             #     btn = gr.UploadButton("Upload", file_types=["image"])
#
#         txt.submit(bot.run_text, [txt, state], [chatbot, state])
#         txt.submit(lambda: "", None, txt)
#         clear.click(bot.memory.clear)
#         clear.click(lambda: [], None, chatbot)
#         clear.click(lambda: [], None, state)
#         demo.launch(server_name="127.0.0.1", server_port=6007)