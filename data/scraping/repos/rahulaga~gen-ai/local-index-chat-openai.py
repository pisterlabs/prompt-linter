from llama_index import LLMPredictor, StorageContext, load_index_from_storage
from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import langchain
import gradio as gr

#debug
langchain.debug=False

#llm
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", max_tokens=1024)
llm_predictor = LLMPredictor(llm=llm)

#load local index
storage_context = StorageContext.from_defaults(persist_dir="local-index")
index = load_index_from_storage(storage_context)

# define custom retrievers
query_engine = index.as_query_engine()

# tool config
tool_config = IndexToolConfig(
    query_engine=query_engine,
    name="VMware Index",
    description="Documents about VMware cloud services and cloud partner navigator",
    tool_kwargs={"return_direct": True}
)

toolkit = LlamaToolkit(
    index_configs=[tool_config]
)

memory = ConversationBufferMemory(memory_key="chat_history")
agent_chain = create_llama_chat_agent(
    toolkit,
    llm,
    memory=memory,
    verbose=True
)

#If you want command line remove Gradio below and uncomment this
#while True:
#    text_input = input("User: ")
#    response = agent_chain.run(input=text_input)
#    print(f'Agent: {response}')

#Gradio chatbot
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    prompt = gr.Textbox()

    def respond(message, chat_history):
        response = agent_chain.run(input=message)
        chat_history.append((message, response))
        return "", chat_history

    prompt.submit(respond, [prompt, chatbot], [prompt, chatbot])

demo.launch()
