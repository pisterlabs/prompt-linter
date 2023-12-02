# 使用ToolBench实现Agent整体框架
from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent, AgentOutputParser
from langchain import OpenAI, SerpAPIWrapper
import re
from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.tools import Tool
from langchain.agents.utils import validate_tools_single_input
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain.pydantic_v1 import Field
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools.base import BaseTool

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    AsyncCallbackManagerForToolRun,
    CallbackManagerForChainRun,
    CallbackManagerForToolRun,
    Callbacks,
)

from typing import Any, Callable, List, NamedTuple, Optional, Sequence

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
import json

# 用来映射tool的类型
tool_type = {
    'query_industry_information': 'query',
    'query_company_information': 'query',
    'query_basic_knowledge': 'query',

    'get_company_code': 'inference_engine',
    'get_financial_metric': 'inference_engine',

    'sql_execute':'database',
    'list_datatables':'database',
    'look_datatable_info':'database',
    'ask_for_database': 'database',
    'get_Wind_id': 'database',
    'show_company_info':'database',
    'get_company_sheets':'database'

}

class ToolBenchOutputParser(AgentOutputParser):
 
    def parse(self, response: str) -> Union[AgentAction, AgentFinish]:
        # print('-' * 20)
        # print(llm_output)
        # print()
        # Check if agent should finish
        
        # 尝试将其解析成字典格式
        # try:
        #     response = json.loads(response)
        #     llm_output = response['output']
        #     middle_process = response['middle_process']
        #     print("解析成功")
        #     print(response)
        # except Exception as e:
        #     print("")
        #     llm_output = response

        llm_output = response


        if "Final Answer:" in llm_output:
            # 解析停止状态
            result = llm_output.split("Final Answer:")[-1].strip()

            # try:
            #     result = json.loads(result)
            #     res_type = result['return_type']

            #     # 如果模型
            #     if res_type == 'give_up_and_restart':
            #         # raise NotImplementedError("Can't not solve give up problem!")
            #         return AgentFinish(
            #             return_values={"output": "未找到相应的结果，终止本次问答"},
            #             log=llm_output,
            #         )

            #     res_context = result['final_answer']
            # except Exception as e:
            #     # 解析失败
            #     print("Error!!!")
            #     print(e)
            #     print(llm_output)
            #     return AgentAction(tool="echo", tool_input="生成解析失败，json格式错误，请重新生成正确的json格式", log=llm_output) 

            return AgentFinish(
                return_values={"output": result},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")

            print("Parse Error!")
            print(llm_output)
            AgentAction(tool="echo", tool_input="调用tools失败！请按照Thought、Action、Action Input的格式生成调用内容", log=llm_output) 

        try:
            action = match.group(1).strip()
            action_args = match.group(2)
        except Exception as e:
            print("Error!!!")
            print(e)
            print(llm_output)
            return AgentFinish(
                        return_values={"output": "生成结果出错！"},
                        log=llm_output,
                    )

        # 将插件的超参数进行解析


        # 对action input进行进一步的解析
        # print("Action: ", action)
        # print(len(action))
        # print("Action Input: ", action_args)
        # print(type(action_args))
        # print(parse_flag)

        # 如果解析成功
        
        # 解析查询工具的结果
        
        
        return AgentAction(tool=action, tool_input=action_args, log=llm_output)

        # return AgentAction(tool="echo", tool_input="调用API失败！请按照Thought、Action、Action Input的格式生成调用内容", log=llm_output) 


output_parser = ToolBenchOutputParser()


from typing import List, Tuple, Any, Union
from langchain.schema import AgentAction, AgentFinish


class ToolBench_Agent(LLMSingleActionAgent):
    """ToolBench Custom Agent."""

    llm_chain: LLMChain
    output_parser: AgentOutputParser = output_parser
    allowed_tools: List
    stop: List[str] = ['Observation:']

    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with the observations.
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = self.llm_chain.run(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)

    async def aplan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """Given input, decided what to do.

        Args:
            intermediate_steps: Steps the LLM has taken to date,
                along with observations
            callbacks: Callbacks to run.
            **kwargs: User inputs.

        Returns:
            Action specifying what tool to use.
        """
        output = await self.llm_chain.arun(
            intermediate_steps=intermediate_steps,
            stop=self.stop,
            callbacks=callbacks,
            **kwargs,
        )
        return self.output_parser.parse(output)