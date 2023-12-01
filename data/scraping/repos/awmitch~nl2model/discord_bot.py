import discord
import asyncio
import configparser
import logging
import time
# Import things that are needed generically
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langchain import llm_cache
from langchain.cache import InMemoryCache
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import ConversationalRetrievalChain

from langchain.chains.flare.base import QuestionGeneratorChain, _OpenAIResponseChain 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from urllib.parse import urlparse
from OMPython import OMCSessionZMQ, ModelicaSystem
import langchain
import re
import os

from GTbot import *
from webcrawl import *
llm_cache = InMemoryCache()
langchain.verbose = True
# Load configuration file
config = configparser.ConfigParser()
config.read('config.ini')
TOKEN = config.get('DEFAULT', 'TOKEN')
os.environ["OPENAI_API_KEY"] = config.get('DEFAULT', 'OPENAI')

# Set constants
MAX_ITER = 6
MAX_TOKEN_LIMIT = None
CHATGPT = True

logger = logging.getLogger('discord.gateways')
logger.setLevel(logging.ERROR)

# Set up logging
logging.basicConfig(
    filename="output.log",
    filemode='a',
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
)

#Set up discord client
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True
client = discord.Client(intents=intents)


class Nl2ModelBot():

##########################################################
#####################  Lookup Model  #####################
##########################################################
    webcrawler = WebCrawler()
    
    # question generator chain and document combiner
    
    lookup_prompt = PromptTemplate(
        template=LOOKUP_TEMPLATE,
        input_variables=["chat_history","question"]
        )
    question_handler = QuestionHandler()
    question_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        max_tokens=MAX_TOKEN_LIMIT,
        temperature=0,
        callbacks=[question_handler],
        )
    question_generator = LLMChain(
        llm=question_model,
        prompt=lookup_prompt, 
        #verbose=True,
        callbacks=[question_handler]
        )
    
    lookup_handler = LookupHandler()
    lookup_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        max_tokens=MAX_TOKEN_LIMIT,
        temperature=0,
        callbacks=[lookup_handler],
        )
    doc_chain = load_qa_chain(
        lookup_model, 
        chain_type="stuff",
        #verbose=True,
        )
    
    # document compressor chain
    compressor_parser=NoModelicaParser()
    compressor_template = COMPRESSOR_TEMPLATE.format(no_output_str=compressor_parser.no_output_str)
    compressor_prompt=PromptTemplate(
        template=compressor_template,
        input_variables=["question", "context"],
        output_parser=compressor_parser
    )
    compressor_handler = CompressorHandler()
    compressor_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        max_tokens=MAX_TOKEN_LIMIT,
        temperature=0,
        callbacks=[compressor_handler],
        )
    compressor_chain = LLMChain(
        llm=compressor_model,
        prompt=compressor_prompt,
        #verbose=True,
        callbacks=[compressor_handler],
        )
    compressor = LLMChainExtractor(llm_chain=compressor_chain)
    embeddings = Nl2ModelEmbedding(model="cl100k_base")
    nl2model_vector = FAISS.load_local(f"processed/openmodelica.org/index.html.txt_index", embeddings)
    nl2model_retriever = nl2model_vector.as_retriever()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=nl2model_retriever,
        )
    lookup_mem = ConversationBufferMemory(input_key="question",memory_key="chat_history", return_messages=True)
    lookup_chain = ConversationalRetrievalChain(
        retriever=compression_retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        max_tokens_limit=MAX_TOKEN_LIMIT,
        #verbose=True,
        memory=lookup_mem,
        callbacks=[lookup_handler],
    )

##########################################################
######################  Model Coder  #####################
##########################################################

    modelica_omc = OMCSessionZMQ()
    #modelica_system = ModelicaSystem(f"./model.mo","BouncingBall",["Modelica"])
    #modelica_system.buildModel()
    modelica_model = ModelObject(
        omc=modelica_omc,
        lookup_chain=lookup_chain,
        lookup_mem=lookup_mem,
        #modelica_system=modelica_system,
        nl2model_retriever=nl2model_retriever,
        nl2model_vector=nl2model_vector,
    )
    
    modelica_output_parser = ModelicaOutputParser()
    modelica_mem = ConversationBufferMemory(input_key="user_input",memory_key="chat_history", return_messages=True)

    # question chain
    modelica_handler = ModelicaHandler()
    modelica_llm_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        max_tokens=MAX_TOKEN_LIMIT,
        temperature=0,
        callbacks=[modelica_handler],
        )
    modelica_question_prompt = PromptTemplate(
        template=MODELICA_QUESTION_TEMPLATE,
        input_variables=["user_input", "current_response", "uncertain_span"],
    )
    modelica_question_chain = QuestionGeneratorChain(
        llm=modelica_llm_model,
        prompt=modelica_question_prompt,
        )
    modelica_prompt = PromptTemplate(
        template=MODELICA_TEMPLATE,
        input_variables=["context",  "user_input", "response"],
        )
    
    # response chain
    response_llm = OpenAI(
            model_name="text-davinci-003",
            max_tokens=32,
            model_kwargs={"logprobs": 1},
            temperature=0,
        )
    response_handler = ResponseHandler()
    modelica_response_chain = _OpenAIResponseChain(
        llm=response_llm,
        prompt=modelica_prompt,
        callbacks=[response_handler]
    )
    modelica_retriever = ModelicaRetriever(
        modelica_model=modelica_model,
        discord_client=client,
        compression_retriever=compression_retriever,
        )
    # flare chain
    modelica_chain = ModelicaFlareChain(
        question_generator_chain=modelica_question_chain,
        response_chain=modelica_response_chain,
        output_parser=modelica_output_parser,
        retriever=modelica_retriever,#compression_retriever,#vectorstore.as_retriever(),
        min_prob=0.2,
        #verbose=True,
        memory=modelica_mem,
        callbacks=[modelica_handler],
        modelica_model=modelica_model
        )
    logging.info(f"Initial modelica_chain: {id(modelica_chain)}")

##########################################################
####################  Nl2Model Agent  ####################
##########################################################

    nl2model_tools = Nl2ModelTools(modelica_chain)
    nl2model_handler = Nl2ModelHandler()
    llm_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        max_tokens=MAX_TOKEN_LIMIT,
        temperature=0,
        callbacks=[nl2model_handler],
        )
    # Initialize the custom tool
    summary_handler = SummaryHandler()
    nl2model_mem = Nl2ModelMemory(
        llm=llm_model, 
        human_prefix="User", 
        memory_key="chat_history", 
        input_key="question",
        #summary_callbacks=[summary_handler]
        )
    # Make model global/shared by reference to all to all tools
    nl2model_prompt = Nl2ModelPrompt(
        template=NL2MODEL_TEMPLATE,
        tools=nl2model_tools,
        input_variables=["question", "intermediate_steps"],
        memory=nl2model_mem
        )
    
    nl2model_chain = LLMChain(
        llm=llm_model, 
        prompt=nl2model_prompt, 
        #verbose=True,
        memory = nl2model_mem,
        #callbacks=[nl2model_handler],
        )
    nl2model_parser = Nl2ModelParser()
    nl2model_agent = Nl2ModelAgent(
        llm_chain=nl2model_chain,
        output_parser=nl2model_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in nl2model_tools],
        )
    nl2model_executor = AgentExecutor.from_agent_and_tools(
        agent=nl2model_agent, 
        tools=nl2model_tools, 
        #verbose=True, 
        max_iterations=MAX_ITER, 
        early_stopping_method="generate")

    

    def extract_domain(self,text):
        url_match = re.search(self.webcrawler.HTTP_URL_PATTERN, text)
        if url_match:
            url = url_match.group()
            domain = urlparse(url).netloc
            return domain
        else:
            return None
    def extract_path(self,text):
        url_match = re.search(self.webcrawler.HTTP_URL_PATTERN, text)
        if url_match:
            url = url_match.group()
            parsed_url = urlparse(url)
            path_parts = parsed_url.path.rsplit('/', 1)[0]
            return f'{parsed_url.scheme}://{parsed_url.netloc}{path_parts}'
        else:
            return None

    def remove_emojis(self, text):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text).encode('utf-8', errors='ignore').decode('utf-8')
    
    async def send_chunked_message(self, channel, message):
        message_parts = []
        current_message = ""
        for part in message.split(', '): 
            if len(current_message + part + ', ') < 2000: 
                current_message += part + ', '
            else: 
                message_parts.append(current_message.rstrip(', '))
                current_message = part + ', '
        message_parts.append(current_message.rstrip(', '))
        for part in message_parts:
            await channel.send(part)

    async def handle_nl2model_command(self, channel, message):
        human_input = self.remove_emojis(message.content[5:].strip())
        if not human_input:
            await self.send_chunked_message(channel,"Please provide a prompt.") 
            return
        self.lookup_handler.set_channel(channel)
        self.summary_handler.set_channel(channel)
        self.modelica_handler.set_channel(channel)
        self.modelica_retriever.set_channel(channel,message.author)
        self.nl2model_handler.set_channel(channel)
        self.question_handler.set_channel(channel)
        self.response_handler.set_channel(channel)
        self.compressor_handler.set_channel(channel)
        

        start_time = time.time()
        self.nl2model_kwargs = {
            "question":human_input,
            }
        
        response = await self.nl2model_executor.arun(callbacks=[self.nl2model_handler],**self.nl2model_kwargs)
        end_time = time.time()
        response_time = end_time - start_time
        message.content = ""
        await self.handle_check_command(channel, message)
        #await self.send_chunked_message(channel,f"{message.author.mention} {response[:1900]}")

    async def handle_check_command(self, channel, message):
        human_input = self.remove_emojis(message.content[7:].strip())
        # If the input is 'file', send a file
        if human_input == 'files':
            # Ensure the file exists
            if os.path.isfile(self.modelica_model.model_file):  
                await self.send_chunked_message(channel, file=discord.File(self.modelica_model.model_file))
                await self.send_chunked_message(channel, file=discord.File(self.modelica_model.results_file))
            else:
                await self.send_chunked_message(channel, f"{message.author.mention} File not found.")
        else:
            list_names = ['quantities', 'continuous', 'inputs', 'outputs', 'parameters', 'simOptions', 'solutions', 'code']
            if human_input and human_input in list_names:
                response = self.modelica_model.get_value(human_input)
                await self.send_chunked_message(channel, f"{message.author.mention} {response[:1900]}")
            else:
                responses = {name: self.modelica_model.get_value(name) for name in list_names}
                await self.send_chunked_message(channel, f"{message.author.mention} {responses}")

    async def handle_crawl_command(self, channel, message):
        human_input = self.remove_emojis(message.content[7:].strip())
        domain = self.extract_domain(human_input)
        path = self.extract_path(human_input)
        if ((not human_input) or (not domain)):
            await self.send_chunked_message(channel,"Please provide a url.") 
            return
        self.webcrawler.crawl(human_input, domain, path)
        
        await self.send_chunked_message(channel,f"Successfully crawled `{domain}`, use `!embed {domain}` to convert to useable vectorstores.")
       
    async def handle_embed_command(self, channel, message):
        human_input = self.remove_emojis(message.content[7:].strip())
        # Check if there is a file provided
        input_parts = human_input.split(' ', 1) 
        domain = input_parts[0]
        file = input_parts[1] if len(input_parts) > 1 else None  # If there is no file, None will be assigned
        
        domains = []
        with os.scandir("./text") as entries:
            for entry in entries:
                if entry.is_dir():
                    domains.append(entry.name)
        if ((not human_input) or (domain not in domains)):
            return await self.send_chunked_message(channel,f"Please provide a valid domain from {domains}.") 
            
        if file:
            files = []
            if domain in domains:
                with os.scandir(f"./text/{domain}") as entries:
                    for entry in entries:
                        if entry.is_file():
                            files.append(entry.name)

            if file not in files:
                return await self.send_chunked_message(channel,f"The file '{file}' does not exist in the domain '{domain}'. Please provide a valid file from {files}.")
                
        if not os.path.exists("text/"+domain+"/"):
            os.mkdir("processed/" + domain + "/")
            
        files_to_process = []
        # If a specific file is provided, process only that file
        if file:
            files_to_process.append(file)
        else:  # If not, process all files in the domain
            files_to_process = os.listdir("text/" + domain + "/")
        
        for file_name in files_to_process:
            file_extension = os.path.splitext(file_name)[1]
            if file_extension == '.pdf':
                loader = PyPDFLoader("text/" + domain + "/" + file_name)
            else:
                loader = TextLoader("text/" + domain + "/" + file_name)

            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]
            #return await self.send_chunked_message(channel, f"{texts}, {metadatas}")
            text_embeddings = self.embeddings.embed_documents(texts)
            text_embedding_pairs = list(zip(texts, text_embeddings))

            db = FAISS.from_embeddings(text_embedding_pairs, self.embeddings, metadatas)
            db.save_local(f"processed/{domain}/{file_name}_index")
                
        if file:
            await self.send_chunked_message(channel,f"Successfully embedded {file} from {domain},use `!load {domain} {file}` to query with the reference.")
        else:
            await self.send_chunked_message(channel,f"Successfully embedded {domain}, use `!load {domain}` to query with the reference.")

    async def handle_load_command(self, channel, message):
        human_input = self.remove_emojis(message.content[6:].strip())
        # Check if there is a file provided
        input_parts = human_input.split(' ', 1) 
        domain = input_parts[0]
        file = input_parts[1] if len(input_parts) > 1 else None  # If there is no file, None will be assigned
        
        domains = []
        with os.scandir("./processed") as entries:
            for entry in entries:
                if entry.is_dir():
                    domains.append(entry.name)
        if not human_input:
            pass
        elif (domain not in domains):
            await self.send_chunked_message(channel,f"Please provide a valid domain from {domains}.") 
            return
        else:
            domains = [domain]

        init_flag = False
        for domain in domains:
            # If a specific file is provided, load only that file
            new_db = None
            
            files = []
            if domain in domains:
                with os.scandir(f"./processed/{domain}") as entries:
                    for entry in entries:
                        if entry.is_dir():
                            files.append(entry.name)
            if file:
                if f"{file}_index" not in files:
                    return await self.send_chunked_message(channel,f"The file '{file}_index' does not exist in the domain '{domain}'. Please provide a valid file from {files}.")
                new_db = FAISS.load_local(f"processed/{domain}/{file}_index", self.embeddings)
            else:  # If not, load the entire domain
                sub_domains = []
                with os.scandir(f"./processed/{domain}") as entries:
                    for entry in entries:
                        if entry.is_dir():
                            sub_domains.append(entry.name)
                for sub_domain in sub_domains:
                    if not new_db:
                        new_db = FAISS.load_local(f"processed/{domain}/{sub_domain}", self.embeddings)
                    else:
                        new_db.merge_from(FAISS.load_local(f"processed/{domain}/{sub_domain}", self.embeddings))
            if not init_flag:
                init_flag = True
                self.nl2model_vector = new_db
            else:
                self.nl2model_vector.merge_from(new_db)
        self.nl2model_retriever = self.nl2model_vector.as_retriever()
        self.modelica_model.nl2model_retriever = self.nl2model_retriever
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor, 
            base_retriever=self.modelica_model.nl2model_retriever,
            )
        self.lookup_chain = ConversationalRetrievalChain(
            retriever=self.compression_retriever,
            question_generator=self.question_generator,
            combine_docs_chain=self.doc_chain,
            max_tokens_limit=MAX_TOKEN_LIMIT,
            #verbose=True,
            memory=self.modelica_model.lookup_mem,
            callbacks=[self.lookup_handler],
            )
        self.modelica_model.lookup_chain = self.lookup_chain
        if not human_input:
            await self.send_chunked_message(channel,f"Successfully loaded {domains}, use !ref to query with the reference.")
        elif file:
            await self.send_chunked_message(channel,f"Successfully loaded {file} from {domain}, use !ref to query with the reference.")
        else:
            await self.send_chunked_message(channel,f"Successfully loaded {files} from {domain}, use !ref to query with the reference.")

    async def handle_ref_command(self, channel, message):
        human_input = self.remove_emojis(message.content[5:].strip())
        if (not human_input):
            return await self.send_chunked_message(channel,"Please provide a prompt.") 
        self.lookup_handler.set_channel(channel)
        self.question_handler.set_channel(channel)
        self.compressor_handler.set_channel(channel)

        kwargs = {
            "question":human_input,
            "chat_history":f"{self.lookup_mem.buffer}",
            }
        response = await self.lookup_chain.arun(**kwargs)#callbacks=[self.lookup_handler]
        if not response:
            return await self.send_chunked_message(channel,"Unable to look up. Try loading first `!load`.")
        
        if response:
            self.modelica_model.modelica_context = response
        #    await self.send_chunked_message(channel,response)


    async def handle_rst_command(self, channel):
        self.nl2model_mem.clear()
        self.modelica_mem.clear()
        self.lookup_mem.clear()
        logging.info(f"rst modelica_model: {id(self.modelica_model)}")
        self.modelica_model.modelica_input = ""
        self.modelica_model.modelica_context = ""
        #self.nl2model_vector = FAISS.load_local(f"processed/openmodelica.org/index.html.txt_index", self.embeddings)
        #self.modelica_model.update_retriever()
        await self.send_chunked_message(channel,"Successfully reset.")

    async def handle_status_command(self, channel):
        logging.info(f"Status modelica_model: {id(self.modelica_model)}")
        message = f"Lookup Memory:\n{self.lookup_mem.buffer}\n\n" + f"Summary Memory:\n{self.nl2model_mem.buffer}\n\n" + f"Fail Input:\n{self.modelica_model.modelica_input}\n\n" + f"Flare Memory:\n{self.modelica_model.modelica_context}\n\n"
        await self.send_chunked_message(channel,f"{message}")

    async def handle_help_command(self, channel):
        help_message = ("Available commands for nl2model:\n"
        "!n2m `[prompt]` - Interact with the n2lmodel agent.\n"
        "!check `[quantities, continuous, inputs, outputs, parameters, simOptions, solutions, code, files]` - must choose one.\n"
        "!rst - Resets the nl2model state for a new interaction.\n"
        "!crawl `[url]` - crawls and stores provided website as text documents.\n"
        "!embed `[domain]` `[file]`(optional) - performs text embedding on the domain or particular file.\n"
        "!load `[domain]` `[file]`(optional) - loads the specified domain or particular file into the lookup vectorstore.\n"
        "!ref `[prompt]` - using the loaded embedded text, query with reference.\n"
        "!status - displays information on the current bot state."
        "!help - displays this message.")
        await self.send_chunked_message(channel,help_message)

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.channel.name == 'console':
        # Define command mapping
        command_mapping = {
            '!n2m': bot.handle_nl2model_command,
            '!check': bot.handle_check_command,
            '!rst': bot.handle_rst_command,
            '!crawl': bot.handle_crawl_command,
            '!embed': bot.handle_embed_command,
            '!load': bot.handle_load_command,
            '!ref': bot.handle_ref_command,
            '!status': bot.handle_status_command,
            '!help': bot.handle_help_command,
        }

        try:
            # Check each command for a match and execute the corresponding handler
            for command, handler in command_mapping.items():
                if message.content.startswith(command):
                    if command == '!rst' or command == '!status' or command == '!help':
                        # If the handler does not need the message as an argument
                        await handler(message.channel)
                    else:
                        # If the handler needs the message as an argument
                        await handler(message.channel, message)
                    break

        except Exception as e:
            logging.exception(f"Error processing gpt command: {e}")
            await bot.send_chunked_message(message.channel, f"The following error occurred: {e}")

@client.event
async def on_ready():
    logging.info(f'{client.user} has connected to Discord!')

bot = Nl2ModelBot()    
client.run(TOKEN, log_handler=None)
