# from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool, LlamaToolkit, create_llama_chat_agent
# from langchain import OpenAI
# from langchain.llms import OpenAIChat
# from langchain.chat_models import ChatOpenAI
# from langchain.agents import initialize_agent
# from llama_index import SimpleDirectoryReader,GPTListIndex,GPTVectorStoreIndex,LLMPredictor,PromptHelper,ServiceContext,StorageContext,load_index_from_storage
# from llama_index.indices.list import GPTListIndex
# from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
# from langchain.chains.conversation.memory import ConversationBufferMemory
# import openai, os
# os.environ["OPENAI_API_KEY"] = "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"
# openai.api_key = os.environ["OPENAI_API_KEY"] 
# index_configs= []
# storage_context = StorageContext.from_defaults(persist_dir = 'D:\Code\AAA_github\DATN_Web\BotGPT\ChatGPT_0307_4')
# index = load_index_from_storage(storage_context)
# query_engine = index.as_query_engine()
# tool_config = IndexToolConfig(
#     query_engine = query_engine,
#     name=f"BookChatbot",
#     description=f"trả lời câu hỏi liên quan về sách, dữ liệu sách như giá tiền, id, thể loại",
#     tool_kwargs={"return_direct": True}
# )
# index_configs.append(tool_config)
# tool = LlamaIndexTool.from_tool_config(tool_config)
# toolkit = LlamaToolkit(index_configs =  index_configs)
# memory = ConversationBufferMemory(memory_key="chat_history")
# llm=ChatOpenAI(temperature=0)
# agent_chain=create_llama_chat_agent(
#     toolkit,
#     llm,
#     memory=memory,
#     verbose=True
# )
# while True:
#   text_input = input("User: ")
#   response = agent_chain.run(input=text_input)
#   print(f"Agent: {response}")