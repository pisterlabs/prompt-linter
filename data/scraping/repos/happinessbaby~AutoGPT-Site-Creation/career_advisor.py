
import os
from pathlib import Path
from typing import Any
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.agents import ConversationalChatAgent, Tool, AgentExecutor
from common_utils import file_loader, check_content, binary_file_downloader_html, search_all_chat_history, search_user_material
from langchain_utils import (create_vectorstore, create_summary_chain,
                             retrieve_redis_vectorstore, split_doc, CustomOutputParser, CustomPromptTemplate, create_vs_retriever_tools,
                             create_retriever_tools, retrieve_faiss_vectorstore, merge_faiss_vectorstore, handle_tool_error, create_search_tools, create_wiki_tools)
# from langchain.prompts import BaseChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
# from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType, Tool, load_tools
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict, AgentAction
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.docstore import InMemoryDocstore
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.tools.human.tool import HumanInputRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks import get_openai_callback, StdOutCallbackHandler, FileCallbackHandler
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
# from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
# from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
# from langchain.schema.messages import SystemMessage
# from langchain.prompts import MessagesPlaceholder
# from langchain.agents import AgentExecutor
# from langchain.agents.agent_toolkits import (
#     create_vectorstore_agent,
#     VectorStoreToolkit,
#     create_vectorstore_router_agent,
#     VectorStoreRouterToolkit,
#     VectorStoreInfo,
# )
from langchain.vectorstores import FAISS
# from feast import FeatureStore
import pickle
import json
import langchain
import faiss
from loguru import logger
from langchain.evaluation import load_evaluator
from basic_utils import convert_to_txt, read_txt
from openai_api import get_completion
from langchain.schema import OutputParserException
from multiprocessing import Process, Queue, Value
from generate_cover_letter import cover_letter_generator, create_cover_letter_generator_tool
from upgrade_resume import  resume_evaluator, create_resume_evaluator_tool
from customize_document import create_resume_customize_writer_tool, create_cover_letter_customize_writer_tool, create_personal_statement_customize_writer_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from typing import List, Dict
from json import JSONDecodeError
from langchain.tools import tool, format_tool_to_openai_function
import re
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt
from langchain.agents.agent_toolkits import FileManagementToolkit
from langchain.tools.file_management.read import ReadFileTool
from langchain.cache import InMemoryCache
from langchain.tools import StructuredTool




from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
log_path = os.environ["LOG_PATH"]
# debugging langchain: very useful
langchain.debug=True 
# The result evaluation process slows down chat by a lot, unless necessary, set to false
evaluate_result = False
# The instruction update process is still being tested for effectiveness
update_instruction = False
delimiter = "####"
word_count = 100
memory_max_token = 500
memory_key="chat_history"




### ERROR HANDLINGS: 
# for tools, will use custom function for all errors. 
# for chains, top will be langchain's default error handling set to True. second layer will be debub_tool with custom function for errors that cannot be automatically resolved.


        


class ChatController():

    # llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0, cache = False)
    llm = ChatOpenAI(model="gpt-4", streaming=True,temperature=0, cache = False)
    embeddings = OpenAIEmbeddings()
    chat_memory = ConversationBufferMemory(llm=llm, memory_key=memory_key, return_messages=True, input_key="input", output_key="output", max_token_limit=memory_max_token)
    # chat_memory = ReadOnlySharedMemory(memory=chat_memory)
    # retry_decorator = _create_retry_decorator(llm)
    langchain.llm_cache = InMemoryCache()


    def __init__(self, userid):
        self.userid = userid
        self._initialize_log()
        self._initialize_chat_agent()
        if update_instruction:
            self._initialize_meta_agent()
        


    def _initialize_chat_agent(self) -> None:

        """ Initializes main chat, a CHAT_CONVERSATIONAL_REACT_DESCRIPTION agent:  https://python.langchain.com/docs/modules/agents/agent_types/chat_conversation_agent#using-a-chat-model """
    
        # initialize tools
        cover_letter_tool = create_cover_letter_generator_tool()
        cover_letter_customize_tool = create_cover_letter_customize_writer_tool()
        # cover letter generator tool
        # cover_letter_tool = [cover_letter_generator]
        resume_evaluator_tool = create_resume_evaluator_tool()
        # resume evaluator tool
        # resume_evaluator_tool = [resume_evaluator]
        resume_customize_tool = create_resume_customize_writer_tool()
        # document_customize_tool = [document_customize_writer]
        personal_statement_customize_tool = create_personal_statement_customize_writer_tool()
        user_material_tool = [search_user_material]
        # interview mode starter tool
        # interview_tool = self.initiate_interview_tool()
        # file_loader_tool = create_file_loader_tool()
        # file loader tool
        working_directory = f"./static/{self.userid}/"
        file_loader_tool = [file_loader]
        # file_sys_tools = FileManagementToolkit(
        #     root_dir=working_directory, # ensures only the working directory is accessible 
        #     selected_tools=["read_file", "list_directory"],
        # ).get_tools()
        # file_sys_tools[0].return_direct = True
        # file_sys_tools[1].return_direct = True
        # file_sys_tools[0].description = 'Read file from disk. DO NOT USE THIS TOOL UNLESS YOU ARE TOLD TO DO SO.'
        # file_sys_tools[1].description = 'List files and directories in a specified folder. DO NOT USE THIS TOOL UNLESS YOU ARE TOLD TO DO SO.'

        requests_get = load_tools(["requests_get"])
        link_download_tool = [binary_file_downloader_html]
        # general vector store tool
        store = retrieve_faiss_vectorstore("faiss_web_data")
        retriever = store.as_retriever()
        general_tool_description = """This is a general purpose tool. Use it to answer general job related questions through searching database.
        Prioritize other tools over this tool. """
        general_tool= create_retriever_tools(retriever, "search_general_database", general_tool_description)
        # web reserach: https://python.langchain.com/docs/modules/data_connection/retrievers/web_research
        # search = GoogleSearchAPIWrapper()
        # embedding_size = 1536  
        # index = faiss.IndexFlatL2(embedding_size)  
        # vectorstore = FAISS(self.embeddings.embed_query, index, InMemoryDocstore({}), {})
        # web_research_retriever = WebResearchRetriever.from_llm(
        #     vectorstore=vectorstore,
        #     llm=self.llm, 
        #     search=search, 
        # )
        # web_tool_description="""This is a web research tool. You should not use it unless you absolutely cannot find answers elsewhere. Always return source information."""
        # web_tool = create_retriever_tools(web_research_retriever, "search_web", web_tool_description)
        # basic search tool 
        search_tool = create_search_tools("google", 5)
        # wiki tool
        wiki_tool = create_wiki_tools()
        # gather all the tools together
        self.tools =  cover_letter_tool + resume_evaluator_tool + search_tool + wiki_tool + link_download_tool  + general_tool + resume_customize_tool + personal_statement_customize_tool + cover_letter_customize_tool + user_material_tool  + requests_get
        # + [tool for tool in file_sys_tools]
        tool_names = [tool.name for tool in self.tools]
        print(f"Chat agent tools: {tool_names}")

        # initialize evaluator
        if (evaluate_result):
            self.evaluator = load_evaluator("trajectory", agent_tools=self.tools)

        # initialize dynamic args for prompt
        self.entities = ""
        self.instruction = ""
        # initialize chat history
        # self.chat_history=[]
        # initiate prompt template
        # You are provided with information about entities the Human mentioned. If available, they are very important.
        template = """Your name is Acai. You are a career AI advisor. The following is a friendly conversation between a human and you. 

        You are talkative and provides lots of specific details from its context.
          
        If you do not know the answer to a question,  truthfully say you don't know. 

        You should only converse with the human on career and education related topics. 

        If human wants to talk about other unrelated subjects, please let them know that you are a career advisor only. 

        You should not answer unrelated questions. 

        Always check the relevant entities below before answering a question. They will help you assist the human better. 

        Relevant entities: {entities}

        You are provided with the following tools, use them whenever possible:

        If you encounter any problems communicating with Human, follow the Instruction below, if relevant. 

        Instruction: {instruction}
        """


        # Conversation:
        # Human: {input}
        # AI:


        # OPTION 1: agent = CHAT_CONVERSATIONAL_REACT_DESCRIPTION
        # initialize CHAT_CONVERSATIONAL_REACT_DESCRIPTION agent
        # self.chat_memory.output_key = "output"  
        self.chat_agent  = initialize_agent(self.tools, 
                                            self.llm, 
                                            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
                                            # verbose=True, 
                                            memory=self.chat_memory, 
                                            return_intermediate_steps = True,
                                            handle_parsing_errors="Check your output and make sure it conforms!",
                                            callbacks = [self.handler])
        # modify default agent prompt
        prompt = self.chat_agent.agent.create_prompt(system_message=template, input_variables=['chat_history', 'input', 'entities', 'instruction', 'agent_scratchpad'], tools=self.tools)
        self.chat_agent.agent.llm_chain.prompt = prompt
        # Option 2: structured chat agent for multi input tools, currently cannot get to use tools
        # suffix = """Begin! Reminder to ALWAYS respond with a valid json blob of a single action. 
        # Use tools if necessary. Respond directly if appropriate. 
        
        # You are provided with information about entities the Human mentioned. If available, they are very important.

        # Always check the relevant entity information before answering a question.

        # Relevant entity information: {entities}
    
        # Format is Action:```$JSON_BLOB```then Observation:.\nThought:"""

        # chat_history = MessagesPlaceholder(variable_name="chat_history")
        # self.chat_agent = initialize_agent(
        #     self.tools, 
        #     self.llm, 
        #     agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        #     verbose=True, 
        #     memory=self.chat_memory, 
        #     agent_kwargs = {
        #         "memory_prompts": [chat_history],
        #         "input_variables": ["input", "agent_scratchpad", "chat_history", 'entities'],
        #         'suffix': suffix,
        #     },
        #     allowed_tools=tool_names,
        # )


        # OPTION 3: agent = custom LLMSingleActionAgent
        # agent = self.create_custom_llm_agent()
        # self.chat_agent = AgentExecutor.from_agent_and_tools(
        #     agent=agent, tools=self.tools, verbose=True, memory=memory
        # )

        # Option 4: agent = conversational retrieval agent

        # template = """The following is a friendly conversation between a human and an AI. 
        # The AI is talkative and provides lots of specific details from its context.
        #   If the AI does not know the answer to a question, it truthfully says it does not know. 


        # Before answering each question, check your tools and see which ones you can use to answer the question. 

        #   Only when none can be used should you not use any tools. You should use the tools whenever you can.  


        #   You are provided with information about entities the Human mentions, if relevant.

        # Relevant entity information: {entities}

        # {instruction}
        # """

        # self.memory = AgentTokenBufferMemory(memory_key="chat_history", llm=self.llm, return_messages=True, input_key="input")
        # system_message = SystemMessage(
        # content=(
        #     template
        #     ),
        # )
        # self.prompt = OpenAIFunctionsAgent.create_prompt(
        #     system_message=system_message,
        #     extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],

        # )
        # agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        # self.chat_agent = AgentExecutor(agent=agent, tools=self.tools, memory=self.memory, verbose=True,
        #                            return_intermediate_steps=True)


         
            
        

      

    def _initialize_meta_agent(self) -> None:

        """ Initializes meta agent that will try to resolve any miscommunication between AI and Humn by providing Instruction for AI to follow.  """
 
        # It will all the logs as its tools for debugging new errors
        # tool = StructuredTool.from_function(self.debug_error)
        debug_tools = [search_all_chat_history]
        # initiate instruct agent: ZERO_SHOT_REACT_DESCRIPTION
        # prefix =  """Your job is to provide the Instructions so that AI assistant would quickly and correctly respond in the future. 
        
        # Please reflect on this AI  and Human interaction below:

        # ####

        # {chat_history}

        # ####

        # If to your best knowledge AI is not correctly responding to the Human request, or if you believe there is miscommunication between Human and AI, 

        # please provide a new Instruction for AI to follow so that it can satisfy the user in as few interactions as possible.

        # You should format your instruction as:

        # Instruction for AI: 

        # If the conversation is going well, please DO NOT output any instructions and do nothing. Use your tool only if there is an error message. 
        
        # """
        # Your new Instruction will be relayed to the AI assistant so that assistant's goal, which is to satisfy the user in as few interactions as possible.
        # If there is anything that can be improved about the interactions, you should revise the Instructions so that AI Assistant would quickly and correctly respond in the future.
        memory = ReadOnlySharedMemory(memory=self.chat_memory)

        # Whenver there's an error message, please use the "debug_error" tool.
        system_msg = """You are a meta AI whose job is to provide the Instructions so that your colleague, the AI assistant, would quickly and correctly respond to Humans.

        You are provided with their Current Conversation. If the current conversation is going well, you don't need to provide any Instruction. 
        
        If the current conversation is not going well, please use you tool "search chat history" to search for similar occurences and the solution. 
     
        Remember, provide a new Instruction for the AI assistant to follow so that it can satisfy the Human in as few interactions as possible."""


        template = """Complete the objective as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: Human input that can be used to assess if the conversation is going well. 
        Thought: you should always think about what to do
        Action: the action to take, should be based on Chat History below. If necessary, can be one of [{tool_names}] 
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question


        Begin!

        Current Conversation: {chat_history}

        Input: {input}
        {agent_scratchpad}
        """


        prompt = CustomPromptTemplate(
            template=template,
            tools=debug_tools,
            system_msg=system_msg,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["chat_history", "input", "intermediate_steps"],
        )
        output_parser = CustomOutputParser()
        # LLM chain consisting of the LLM and a prompt
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in debug_tools]

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )
        self.meta_agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=debug_tools, memory=memory, verbose=True)



        # self.meta_agent = initialize_agent(
        #         tools,
        #         self.llm, 
        #         agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        #         max_execution_time=1,
        #         early_stopping_method="generate",
        #         agent_kwargs={
        #             'prefix':prefix,
        #             "input_variables": ["input", "agent_scratchpad", "chat_history"]
        #         },
        #         handle_parsing_errors=True,
        #         memory = memory, 
        #         callbacks = [self.handler]
        # )



    def _initialize_log(self) -> None:
         
        """ Initializes log: https://python.langchain.com/docs/modules/callbacks/filecallbackhandler """

         # initialize file callback logging
        logfile = log_path + f"{self.userid}.log"
        self.handler = FileCallbackHandler(logfile)
        logger.add(logfile,  enqueue=True)
        # Upon start, all the .log files will be deleted and changed to .txt files
        for path in  Path(log_path).glob('**/*.log'):
            file = str(path)
            file_name = path.stem
            if file_name != self.userid: 
                # convert all non-empty log from previous sessions to txt and delete the log
                if os.stat(file).st_size != 0:  
                    convert_to_txt(file, f"./log/{file_name}.txt")
                os.remove(file)



    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff between retries
        stop=stop_after_attempt(5)  # Maximum number of retry attempts
    )
    def askAI(self, userid:str, user_input:str, callbacks=None,) -> str:

        """ Main function that processes all agents' conversation with user.
         
        Args:

            userid (str): session id of user

            user_input (str): user question or response

        Keyword Args:

            callbacks: default is None

        Returns:

            Answer or response by AI (str)  
            
         """

        try:    
            # BELOW IS USED WITH CONVERSATIONAL RETRIEVAL AGENT (grader_agent and interviewer)
            print([tools.name for tools in self.tools])
            # response = self.chat_agent({"input": user_input, "chat_history":[], "entities": self.entities, "instruction": self.instruction}, callbacks = [callbacks])
            response = self.chat_agent({"input": user_input, "chat_history":[], "entities": self.entities, "instruction": self.instruction})
            # convert dict to string for chat output
            response = response.get("output", "sorry, something happened, try again.")
            if (update_instruction):
                self.instruction = self.askMetaAgent()
                print(self.instruction)
            # response = asyncio.run(self.chat_agent.arun({"input": user_input, "entities": self.entities}, callbacks = [callbacks]))
            # print(f"CHAT AGENT MEMORY: {self.chat_agent.memory.buffer}")
            # update chat history for instruct agent
            # self.update_chat_history(self.chat_memory)
            # print(f"INSTRUCT AGENT MEMORY: {self.chat_agent.memory.buffer}")
            # update instruction from feedback 
            # self.update_instructions(feedback)       
            if (evaluate_result):
                evalagent_q = Queue()
                evalagent_p = Process(target = self.askEvalAgent, args=(response, evalagent_q, ))
                evalagent_p.start()
                evalagent_p.join()
                evaluation_result = evalagent_q.get()
                # add evaluation and instruction to log
                self.update_meta_data(evaluation_result)
            
        # let instruct agent handle all exceptions with feedback loop
        except Exception as e:
            print(f"ERROR HAS OCCURED IN ASKAI: {e}")
            error_msg = str(e)
            # needs to get action and action input before error and add it to error message
            if (update_instruction):
                query = f""""Debug the error message and provide Instruction for the AI assistant: {error_msg}
                    """        
                self.instruction = self.askMetaAgent(query)
                # self.update_instructions(feedback)
            if evaluate_result:
                self.update_meta_data(error_msg)
            self.askAI(userid, user_input, callbacks)        

        # pickle memory (sanity check)
        # with open('conv_memory/' + userid + '.pickle', 'wb') as handle:
        #     pickle.dump(self.chat_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print(f"Sucessfully pickled conversation: {chat_history}")
        return response
    

    def askMetaAgent(self, query="Update the Instruction please.") -> None:    

        """ Evaluates conversation's effectiveness between AI and Human. Outputs instructions for AI to follow. 
        
        Keyword Args:
        
            query (str): default is empty string
        
        """

        try: 
            feedback = self.meta_agent({"input":query}).get("output", "")
        except Exception as e:
            if type(e) == OutputParserException or type(e)==ValueError:
                feedback = str(e)
                feedback = feedback.removeprefix("Could not parse LLM output: `").removesuffix("`")
            else:
                feedback = ""
        return feedback
    


    def askEvalAgent(self, response: Dict, queue: Queue) -> None:

        """ Evaluates trajectory, the full sequence of actions taken by an agent. 

        See: https://python.langchain.com/docs/guides/evaluation/trajectory/


        Args:

            response (str): response from agent chain that includes both "input" and "output" parameters

            queue (Queue): queue used in multiprocessing
             
        """
        
        try:
            evaluation_result = self.evaluator.evaluate_agent_trajectory(
                prediction=response["output"],
                input=response["input"],
                agent_trajectory=response["intermediate_steps"],
                )
        except Exception as e:
            evaluation_result = ""
        queue.put(evaluation_result)



    # def update_chat_history(self, memory) -> None:
    #     # memory_key = chain_memory.memory_key
    #     # chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
    #     extracted_messages = memory.chat_memory.messages
    #     self.chat_history = messages_to_dict(extracted_messages)
    

    # def update_tools(self, tools:List[Tool]) -> None:

    #     """ Update tools for chat agent."""

    #     self.tools += tools
    #     print(f"Succesfully updated tool {[t.name for t in tools]}for chat agent.")


    def update_entities(self,  text:str) -> None:

        """ Updates entities list for main chat agent. Entities are files user loads or topics of the files. """

        entity_type = text.split(":")[0]
        print(f"Entity type: {entity_type}")
        self.delete_entities(entity_type)
        self.entities += f"\n{text}\n"
        print(f"Successfully added entities {self.entities}.")

    def delete_entities(self, type: str) -> None:

        """ Deletes entities of specific type. """

        delimiter = "###"
        starting_indices = [m.start() for m in re.finditer(type, self.entities)]
        end_indices = [m.start() for m in re.finditer(delimiter, self.entities)]
        for i in range(len(starting_indices)):
            self.entities = self.entities.replace(self.entities[starting_indices[i]:end_indices[i]+len(delimiter)], "")

    # def update_instructions(self, meta_output:str) -> None:
    #     delimiter = "Instructions: "
    #     new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
    #     self.instructions = new_instructions
    #     print(f"Successfully updated instruction to: {new_instructions}")


    def update_meta_data(self, data: str) -> None:
        
        """ Adds custom data to log. """

        with open(log_path+f"{self.userid}.log", "a") as f:
            f.write(str(data))
            print(f"Successfully updated meta data: {data}")



    

    

    
        




    
    

    
        



        
        


    



    

    



    
    


    





             






    












    
    