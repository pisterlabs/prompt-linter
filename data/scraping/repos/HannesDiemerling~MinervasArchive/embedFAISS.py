import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.vectorstores import Chroma,FAISS
from langchain.document_loaders import WebBaseLoader
from typing import List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.agents import Tool, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
import prompts


class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # correct \
        llm_output = llm_output.replace('\\n', '\n')
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip().split("\nAction")[0].strip().split("\nThought")[0].strip().split("\nQuestion")[0].strip()},
                log=llm_output,
            )
        if "Preliminary Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Preliminary Answer:")[-1].split("\nAction")[0].strip().split("\nThought")[0].strip().split("\nQuestion")[0].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        
        pattern = r"(SimpleReportSearch|ReportSummarizer|OnePersonSearch|TermSearch)"
        # Suche nach der Aktion im Eingabestring
        matcher = re.search(pattern, llm_output)
        if matcher:
            # Extrahiere die gefundene Aktion
            action = matcher.group(0).strip()
        else:
            return prompts.parsingError
        
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)

        if match:
            action_input = match.group(2)
        else:
            return prompts.parsingError
        
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

def webpages(df):
    documents=[]
    for urls in list(df['loc']):
        loader = WebBaseLoader(urls)
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                length_function=len
            )
        document = loader.load_and_split(text_splitter=text_splitter)
        for splitdoc in document:
            documents.append(splitdoc)
    return documents

def report_db_dir():
    load_dotenv()
    return os.getenv("FAISS_REPORT_DIR")

def person_db_dir():
    load_dotenv()
    return os.getenv("FAISS_PERSON_DIR")    

def report_db_dir_exists():
    return os.path.isdir(report_db_dir())

def person_db_dir_exists():
    return os.path.isdir(person_db_dir())

def embedding():
    return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL_NAME"), deployment=os.getenv("OPENAI_EMBEDDING_DEPLOYMENT_NAME"), chunk_size=1)

def pages_to_vectorstore(pages):
    vectorstore = FAISS.from_documents(documents=pages, embedding=embedding())
    return vectorstore

def report_vectorstore():
    return FAISS.load_local(folder_path=report_db_dir(), embeddings=embedding())

def person_vectorstore():
    return FAISS.load_local(folder_path=person_db_dir(), embeddings=embedding())