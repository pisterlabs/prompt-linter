from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import xformers
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools
from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import initialize_agent
from typing import Union
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from typing import List
import torch

from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools
from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import initialize_agent
from langchain.chains import RetrievalQA
import chromadb

from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import load_tools
from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import initialize_agent
import argparse


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')

from constants import CHROMA_SETTINGS

from torch import cuda, bfloat16
import transformers

def run_model(query, model_app):
    #  'meta-llama/Llama-2-70b-chat-hf' ----x
    #  meta-llama/Llama-2-13b-chat-hf
    # meta-llama/Llama-2-7b-chat-hf
    # bigscience/bloom-560m
    model_id = 'bigscience/bloom-560m'
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Ask questions to your documents without an internet connection, '
                                                    'using the power of LLMs.')
        parser.add_argument("--hide-source", "-S", action='store_true',
                            help='Use this flag to disable printing of source documents used for answers.')

        parser.add_argument("--mute-stream", "-M",
                            action='store_true',
                            help='Use this flag to disable the streaming StdOut callback for LLMs.')

        return parser.parse_args()
    
    class OutputParser(AgentOutputParser):
        def get_format_instructions(self) -> str:
            return FORMAT_INSTRUCTIONS

        def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
            try:
                # this will work IF the text is a valid JSON with action and action_input
                response = parse_json_markdown(text)
                action, action_input = response["action"], response["action_input"]
                if action == "Final Answer":
                    # this means the agent is finished so we call AgentFinish
                    return AgentFinish({"output": action_input}, text)
                else:
                    # otherwise the agent wants to use an action, so we call AgentAction
                    return AgentAction(action, action_input, text)
            except Exception:
                # sometimes the agent will return a string that is not a valid JSON
                # often this happens when the agent is finished
                # so we just return the text as the output
                return AgentFinish({"output": text}, text)

        @property
        def _type(self) -> str:
            return "conversational_chat"

    # initialize output parser for agent
    parser = OutputParser()

    # Set quantization configuration to load large model with less GPU memory  - Cannot use quantization in Windows
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # Initialize model configuration and model
    hf_token = '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'
    model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_token)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_token
    )
    model.eval()

    print(f"Model loaded on {device}")

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)

    # Define text generation pipeline
    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # Langchain expects the full text
        task='text-generation',
        temperature=0.0,  # 'Randomness' of outputs, 0.0 is the min and 1.0 the max
        max_new_tokens=512,  # Max number of tokens to generate in the output
        repetition_penalty=1.1  # Without this, output begins repeating
    )

    llm = HuggingFacePipeline(pipeline=generate_text)


    
    
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # print("DB work started")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    # print("db work ended")
    retriever = db.as_retriever()
    # docs = db.similarity_search(query)
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]


    memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True, output_key="output"
    )
    tools = load_tools(["llm-math"], llm=llm)

    # initialize agent
    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=False,
        early_stopping_method="generate",
        memory=memory,
        agent_kwargs={"output_parser": parser}
    )


    # setting system messages
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    sys_msg = B_SYS + """
    Please assume the persona of a Software engineer with in-depth expertise in the field of Cloud Computing.
     Provide detailed, step-by-step explanations for any technical questions or problem-solving tasks
    
    """ + E_SYS
    new_prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = new_prompt

    instruction = B_INST + " Please assume the persona of a Software engineer with in-depth expertise in the field of Cloud Computing. Provide detailed, step-by-step explanations for any technical questions or problem-solving tasks " + E_INST
    human_msg = instruction + "\nUser: {input}"

    agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg
    # agent.agent.llm_chain.prompt

    rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=retriever, return_source_documents= not args.hide_source
    )

    from sentence_transformers import SentenceTransformer

    model1 = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    embedding1 = model1.encode("this is sentence").tolist()

    # model_app = 'Unstructured Text'

    if model_app == 'Unstructured':
        docs = db.similarity_search(query)
        # print("The answer based on Text matching search is \n", docs[0].page_content)
        return docs[0].page_content
    elif model_app == 'Structured':
        rag_answer = rag_pipeline(query)
        agent_query = rag_answer['result']
        return agent_query
    elif model_app == 'Word2Vec':
        rag_answer = rag_pipeline(query)
        agent_query = rag_answer['result']
        combined_answer = agent("Answer the question with the context provided" + agent_query)
        print(combined_answer['output'])
        return combined_answer['output']
    else:
        return "Invalid model_app value"

    
    # try:
    # #    query_db = input(" \n Ask a query to your vector database: ")
    # #   query_db = "What are the problems while connecting athena to S3?"
    #     docs = db.similarity_search(query)
    #     # print("The answer based on Text matching search is \n", docs[0].page_content)

    #     # print("\n \n Loading........setting up query console \n")
    #     # query = input(" \n Ask a question to an AI powered Assistant: ")

    #     # if query == "exit":
    #     #     break  # break out of the while loop , this is error becasue
        
    #     # Generate text based on the input query
    #     rag_answer = rag_pipeline(query)
    #     print(rag_answer, "\n")
    #     agent_query = rag_answer['result']
    #     # print("\n \n The RAG based answer is \n \n", agent_query)
    #     print("\n \n these are embeddinggs----", embedding1)
    #     # answer, docs = rag_answer['result'], [] if args.hide_source else rag_answer['source_documents']
        
    #     # agent_input = input("\nEnter a query for agent: ")
    #     # agent_answer = agent(agent_input)

    #     # print("agent answer", agent_answer)
    #     # print("answer -----> ",answer)
        
    #     # print("\n \n Processing........RAG ouput is being fed to agent \n \n ")
    #   # print(str(agent_query))
    #     combined_answer = agent("Answer the question with the context provided"+agent_query)  # str(agent_query) is not working

    #     # print("\n A thoughtfull answer From the AI assistant is \n \n \n ", combined_answer['output'])
    #     return combined_answer['output']
    # except Exception as e:
    #     return {'error': str(e)}
