import faiss
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT
from profit.tools import (
    ReadFileTool,
    WriteFileTool,
    process_csv,
    query_website_tool,
    # zapier_tools,
)

ROOT_DIR = "./data/"

from langchain.llms import Clarifai

clarifi = Clarifai(
    pat="890cdb0cb5aa4795ba51af9670120a1e", 
    user_id="meta", 
    app_id="Llama-2", 
    model_id="llama2-70b-chat"
)



class Agent:
    def __init__(
            self, 
            ai_name="Autobot Swarm Worker",
            ai_role="Worker in a swarm",
            external_tools = None,
            human_in_the_loop=False,
            llama = False,
            temperature = 0.5,
            openai_api_key = None,
        ):
        self.human_in_the_loop = human_in_the_loop
        self.ai_name = ai_name
        self.ai_role = ai_role

        self.temperature = temperature
        self.llama = llama
        self.openai_api_key = openai_api_key

        if self.llama is True:
            self.llm = clarifi
        else:
            pass

        self.setup_tools(external_tools)
        self.setup_memory()
        self.setup_agent()

    def setup_tools(self, external_tools):
        self.tools = [
            WriteFileTool(root_dir=ROOT_DIR),
            ReadFileTool(root_dir=ROOT_DIR),
            process_csv,

            query_website_tool,
            HumanInputRun(),
            # zapier_tools,

        ]
        if external_tools is not None:
            self.tools.extend(external_tools)

    def setup_memory(self):
        try:
            embeddings_model = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)
            self.vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
        except Exception as error:
            raise RuntimeError(f"Error setting up memory. Maybe try tuning the embedding size: {error}")

    def setup_agent(self):
        try: 
            self.agent = AutoGPT.from_llm_and_tools(
                ai_name=self.ai_name,
                ai_role=self.ai_role,
                tools=self.tools,
                llm=self.llm,
                memory=self.vectorstore.as_retriever(search_kwargs={"k": 8}),
                human_in_the_loop=self.human_in_the_loop
            )
        
        except Exception as error:
            raise RuntimeError(f"Error setting up agent: {error}")

    def run(self, task):
        try:
            result = self.agent.run([task])
            return result
        except Exception as error:
            raise RuntimeError(f"Error while running agent: {error}")

    def __call__(self, task):
        return self.run(task)
