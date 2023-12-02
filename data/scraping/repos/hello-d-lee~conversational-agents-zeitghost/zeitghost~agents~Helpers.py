from typing import Dict
from typing import List
from typing import Union
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any
from langchain.schema import AgentAction
from langchain.schema import AgentFinish
from langchain.schema import LLMResult
from zeitghost.vertex.LLM import VertexLLM
import time
import logging


QPS = 600

core_template = """Question: {question}

    Answer: """

core_prompt = PromptTemplate(
    template=core_template
    , input_variables=['question']
)

vector_template = """
    Question: Use [{name}]:
    {question}

    Answer: """

vector_prompt = PromptTemplate(
    template=vector_template
    , input_variables=['name', 'question']
)

bq_template = """{prompt} in {table} from this table of search term volume on google.com
            - do not download the entire table
            - do not ORDER BY or GROUP BY count(*)
            - the datetime field is called date_field
            """
bq_prompt = PromptTemplate(
    template=bq_template
    , input_variables=['prompt', 'table']
)

BQ_PREFIX = """
        LIMIT TO ONLY 100 ROWS - e.g. <QUERY> LIMIT 100
        REMOVE all observation output that has any special characters , or \n
        you are a helpful agent that knows how to use bigquery
        you are using sqlalchemy {dialect}
        Check the table schemas before constructing sql
        Only use the information returned by the below tools to construct your final answer.\nYou MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\n\n
        REMOVE all observation output that has any special characters , or \n
        you are a helpful agent that knows how to use bigquery
        READ THE SCHEMA BEFORE YOU WRITE QUERIES
        DOUBLE CHECK YOUR QUERY LOGIC
        you are using sqlalchemy for Big Query
        ALL QUERIES MUST HAVE LIMIT 100 at the end of them
        Check the table schemas before constructing sql
        Only use the information returned by the below tools to construct your final answer.\nYou MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.\n\n
        If you don't use a where statement in your SQL - there will be problems.
        To get hints on the field contents, consider a select distinct - I don't care about a where statement given there is low cardnality in the data set
        make sure you prepend the table name with the schema: eg: schema.tablename
        MAKE SURE the FROM statement includes the schema like so: schema.tablename
        THERE MUST BE A WHERE CLAUSE IN THIS BECAUSE YOU DON'T HAVE ENOUGH MEMORY TO STORE LOCAL RESULTS
        do not use the same action as you did in any prior step
        MAKE SURE YOU DO NOT REPEAT THOUGHTS - if a thought is the same as a prior thought in the chain, come up with another one
        """

bq_agent_llm = VertexLLM(stop=['Observation:'], #in this case, we are stopping on Observation to avoid hallucentations with the pandas agent
                         strip=True, #this strips out special characters for the BQ agent
                         temperature=0.0,
                         max_output_tokens=1000,
                         top_p=0.7,
                         top_k=40,
                         )

pandas_agent_llm = VertexLLM(stop=['Observation:'], #in this case, we are stopping on Observation to avoid hallucentations with the pandas agent
                             strip=False, #this strips out special characters for the BQ agent
                             temperature=0.0,
                             max_output_tokens=1000,
                             top_p=0.7,
                             top_k=40,
                             )

vectorstore_agent_llm = VertexLLM(stop=['Observation:'], #in this case, we are stopping on Observation to avoid hallucentations with the pandas agent
                             strip=False, #this strips out special characters for the BQ agent
                             temperature=0.0,
                             max_output_tokens=1000,
                             top_p=0.7,
                             top_k=40,
                             )


base_llm = VertexLLM(stop=None, #in this case, we are stopping on Observation to avoid hallucentations with the pandas agent
                     temperature=0.0,
                     max_output_tokens=1000,
                     top_p=0.7,
                     top_k=40
                     )


class MyCustomHandler(BaseCallbackHandler):
    def rate_limit(self):
        time.sleep(1/QPS)

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str],
        **kwargs: Any) -> Any:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.rate_limit()
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        pass

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any) -> Any:
        pass

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any],
        **kwargs: Any) -> Any:
        logging.info(serialized)
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        pass

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any) -> Any:
        pass

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str,
        **kwargs: Any) -> Any:
        logging.info(serialized)
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        pass

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt],
        **kwargs: Any) -> Any:
        pass

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        logging.info(action)
        pass

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        pass

    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
        # return str(text[:4000]) #character limiter
        # self.rate_limit()
