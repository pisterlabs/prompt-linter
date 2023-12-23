import csv
from ctypes import Array
from typing import Any, Coroutine, List, Tuple
import io
import time
import re
import os

from fastapi import UploadFile
import asyncio 
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_csv_agent, load_tools, initialize_agent, AgentType, create_pandas_dataframe_agent
from langchain.tools import HumanInputRun, PythonAstREPLTool
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.callbacks import HumanApprovalCallbackHandler
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain import PromptTemplate
import pandas as pd
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from util.tools import SessionHumanInputRun
import util.config as config
from util.model import TemplateMappingList, TemplateMapping, TemplateMappingCode, TransformValue
import redis

r = redis.from_url(os.environ.get("REDIS_URL"))
#r = redis.from_url('redis://:password@localhost:6379')
class Processor:
    def __init__(self, session):
        self.session = session
    async def extract_csv_description(self, df: UploadFile|str, llm, memory) -> Coroutine[Any, Any, Tuple[pd.DataFrame, str]] :
        df = pd.read_csv(df)
        agent = create_pandas_dataframe_agent(llm=llm,df=df, agent_executor_kwargs={'handle_parsing_errors':True, 'memory':memory},
                                            early_stopping_method="generate", verbose=True,
                                            temperature=0,agent_type=AgentType.OPENAI_FUNCTIONS,)
        descriptions = agent.run("""Describe what is the column name of each of the column table in detail in the following format:
                            <name of column 1>: <description of column 1>\n
                            <name of column 2>: <description of column 2>""", callbacks=[ConsoleCallbackHandler()])
        return df, descriptions
    async def _human_prompt(prompt, session):
        r.publish(f'human_prompt_{session}', prompt)
    
    async def _human_input(session):
        p = r.pubsub(ignore_subscribe_messages=True)
        p.subscribe(f'human_input_{session}')
        message = None
        while True:
            message = p.get_message()
            if message and message['type']=='message':
                break
            print("waiting for human input")
            await asyncio.sleep(1)
        return message['data'].decode('utf-8')

    async def process_files(self, table_file, template_file, file_guid):
        table_string = table_file.decode('utf-8')
        template_string = template_file.decode('utf-8')
        llm = ChatOpenAI(openai_api_key=config.OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo-0613", )
        memory = ConversationSummaryBufferMemory(llm=llm,memory_key="chat_history", return_messages=True, max_token_limit=1500)
        
        table_df, table_descriptions = await self.extract_csv_description(io.StringIO(table_string), llm, memory=memory)
        r.publish(f'{self.session}_response', 'table_descriptions')
        r.publish(f'{self.session}_response', table_descriptions)

        template_df, template_descriptions = await self.extract_csv_description(io.StringIO(template_string), llm, memory=memory)
        r.publish(f'{self.session}_response', 'template_descriptions')
        r.publish(f'{self.session}_response', template_descriptions)
        dfs =[table_df, template_df]
        human_tool = SessionHumanInputRun(session=self.session)
        human_tool.description = '''
        Use this tool to take human input. 
        If the mapping is ambiguous, ask 'human' a question with options in the following format. 
        Make the human confirm the mapping by selecting the appropriate number.
            - Question: The template column <template column name> should be mapped to which one of the table columns 
            (1: <table column name 1>, 2: <table column name 2> (Recommended), 3:<table column name 3>, ...)? Select the appropriate number or specify the column name. 
        '''
        human_tool.prompt_func= Processor._human_prompt
        human_tool.input_func = Processor._human_input
        
        mappings = await self.get_mappings(llm, table_descriptions, template_descriptions, human_tool)
        codes = await self.get_template_formatting_code(llm, table_df, template_df, human_tool, mappings, memory)
        new_table_df = table_df.loc[:,[code.table_column for code in codes]]
        for code in codes:
            new_table_df[code.table_column].apply(lambda x: self.format_value(x,code=code.code))
        r.set(f"{self.session}_{file_guid}", new_table_df.to_msgpack(compress='zlib'))
        r.publish(f'{self.session}_response', f'file_guid:{file_guid}')

    
    def format_value(self, source_value, code):
        value = TransformValue(source=source_value,destination=source_value)
        try:
            exec(code, {'value':value})
        except Exception as e:
            r.publish(f'{self.session}_response',f'ERROR: \nCode: \n {code} \n Failed with error: \n{e}')
            print(e)
        return value.destination
    
    async def get_mappings(self,llm, table_descriptions, template_descriptions, human_tool):
        parser = PydanticOutputParser(pydantic_object=TemplateMappingList)
        new_parser = OutputFixingParser.from_llm(parser=parser,llm=llm)
        agent = initialize_agent(
            [human_tool],
            llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
            early_stopping_method="force",
            temperature=0.3,
            output_parser=new_parser,
        )
        descriptions = await agent.arun("""Map all the columns of the Template descriptions to columns of the table Descriptions:                          
                                    - Table Descriptions:
                                    """ + table_descriptions + """
                                    - Template Descriptions:
                                    """ + template_descriptions + """
                                    Use the table and template descriptions above to determine the mapping based on similarity, formats and distribution.
                                    If the table column names are ambiguous take human input.
                                """,callbacks=[ConsoleCallbackHandler()],)
        print(descriptions)
        
        mappings = new_parser.parse(descriptions)
        return mappings
    

    async def get_template_formatting_code(self, llm, table_df, template_df, human_tool, mappings: TemplateMappingList, memory):
        
        dfs = []
        dfs.append(table_df)
        dfs.append(template_df)
        df_locals = {}
        df_locals[f"table_df"] = table_df
        df_locals[f"template_df"] = template_df
        parser = PydanticOutputParser(pydantic_object=TemplateMappingCode)
        new_parser = OutputFixingParser.from_llm(parser=parser,llm=llm)
        
        
        codes=[]
        #The code should be in the format of a Python function taking as input a string and returning a string. 
        
        for mapping in mappings.template_mappings:
            human_tool.description = f'''
                Use this tool to get human approval. Always show the samples and code. The human can edit the code and approve it.
            '''
            table_df_samples = table_df[mapping.table_column].sample(5).to_list()
            template_df_samples = template_df[mapping.template_column].sample(5).to_list()
            agent = initialize_agent(
                        [PythonAstREPLTool(locals=df_locals)],
                        llm,
                        agent=AgentType.OPENAI_FUNCTIONS,
                        handle_parsing_errors=True,
                        early_stopping_method="force",
                        temperature=0.3,
                        output_parser=new_parser,
                        memory = memory,
                        memory_key = 'chat_history'
                    )
            #The AI can determine the format of the column values only after sampling.
            #As shown in the output below, generate the code as a Python function taking as input a string and returning a string and also include a call to the generated function.
                                
            code = agent.run(f'''Provide the code to bring the format of values in table_df column  '{mapping.table_column}'
                                to the format of values in template_df column '{mapping.template_column}' based off the values, data types and formats. 
                                Additional samples to be used to generate the code:
                                    '{mapping.table_column}' sample values: [{table_df_samples}]
                                    '{mapping.template_column}' samples values: [{template_df_samples}]
                                The input to the code will be a value object with the following attributes:
                                - source: The value of the table_df column '{mapping.table_column}'.
                                - destination: The value of the template_df column '{mapping.template_column}'.
                                Show the sample values using which the code is generated. 
                                For example, for date columns, they may be in different formats, and it is necessary to change the format from dd.mm.yyyy to mm.dd.yyyy.
                                
                                Final Answer:
                                ```
                                    ```python
                                    def format_value(source_value):
                                        <code to transform source_value into destination_value>
                                        return destination_value
                                    value.destination = format_value(value.source)
                                    ```
                                ```
                                Final Answer should contain the samples and code.
                                ''', callbacks=[ConsoleCallbackHandler(), ])
            print(code)
            human_code = await human_tool.arun(code + '\nSpecify the code with ```python``` tags.')
            regex = r"```python((.|\n|\t)*?)```"
            code = human_code if re.match(regex, human_code) else code
            
            matches = re.findall(regex, code)
            code = ''
            for match in matches:
                code = code + '\n'+ '\n'.join(match)
            codes.append(TemplateMappingCode(template_column=mapping.template_column, 
                                             table_column=mapping.table_column, 
                                             code=code))
        return codes