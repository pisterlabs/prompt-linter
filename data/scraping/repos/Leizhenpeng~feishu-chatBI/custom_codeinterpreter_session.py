import uuid, base64, re, json, time
from codeinterpreterapi import CodeInterpreterSession, File
from langchain.tools import StructuredTool
from codeinterpreterapi.schema import CodeInput
import tools.psql_tools as psql_tools
from langchain.agents import AgentExecutor, BaseSingleActionAgent
from codeinterpreterapi.config import settings
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import AgentAction,LLMResult,AIMessage, OutputParserException,SystemMessage
from typing import Any, Optional, List, Dict
from uuid import UUID
from codeinterpreterapi.agents import OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from codeboxapi import CodeBox  # type: ignore
from codeboxapi.schema import CodeBoxOutput
from io import BytesIO
from langchain.base_language import BaseLanguageModel
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate,MessagesPlaceholder
from langchain.callbacks.base import  BaseCallbackHandler
from codeinterpreterapi.config import settings
from codeinterpreterapi.schema import CodeInterpreterResponse, CodeInput, File, UserRequest
from codeinterpreterapi.schema import (
    CodeInput,
    CodeInterpreterResponse,
    File,
    SessionStatus,
    UserRequest,
)

settings.VERBOSE=True

code_interpreter_system_message =SystemMessage(
    content="""
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. 
As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. 
It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, 
allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

This version of Assistant is called "Code Interpreter" and capable of using a python code interpreter (sandboxed jupyter kernel) to run code. 
The human also maybe thinks this code interpreter is for writing code but it is more for data science, data analysis, and data visualization, file manipulation, and other things that can be done using a jupyter kernel/ipython runtime.
Tell the human if they use the code interpreter incorrectly.
Already installed packages are: (sqlalchemy,psycopg2-binary,numpy pandas matplotlib seaborn scikit-learn yfinance scipy statsmodels sympy bokeh plotly dash networkx).
If you encounter an error, try again and fix the code.
"""
)

determine_modifications_function = {
    "name": "determine_modifications",
    "description": "Based on code of the user determine if the code makes any changes to the file system. \n"
    "With changes it means creating new files or modifying exsisting ones.\n",
    "parameters": {
        "type": "object",
        "properties": {
            "modifications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The filenames that are modified by the code.",
            },
        },
        "required": ["modifications"],
    },
}


determine_modifications_prompt = ChatPromptTemplate(
    input_variables=["code"],
    messages=[
        SystemMessage(
            content="The user will input some code and you will need to determine if the code makes any changes to the file system. \n"
            "With changes it means creating new files or modifying exsisting ones.\n"
            "Answer with a function call `determine_modifications` and list them inside.\n"
            "If the code does not make any changes to the file system, still answer with the function call but return an empty list.\n",
        ),
        HumanMessagePromptTemplate.from_template("{code}"),
    ],
)



async def get_file_modifications(
    code: str,
    llm: BaseLanguageModel,
    retry: int = 2,
) -> Optional[List[str]]:
    if retry < 1:
        return None
    messages = determine_modifications_prompt.format_prompt(code=code).to_messages()
    message = await llm.apredict_messages(messages, functions=[determine_modifications_function])

    if not isinstance(message, AIMessage):
        raise OutputParserException("Expected an AIMessage")

    function_call = message.additional_kwargs.get("function_call", None)

    if function_call is None:
        return await get_file_modifications(code, llm, retry=retry - 1)
    else:
        function_call = json.loads(function_call["arguments"])
        return function_call["modifications"]
    


class CodeCallbackHandler(AsyncIteratorCallbackHandler):
    def __init__(self, session: "CodeInterpreterSession"):
        super().__init__()
        self.session = session

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        if action.tool == "python":
            await self.session.show_code(
                "⚙️ Running code: "
                f"```python\n{action.tool_input['code']}\n```"  # type: ignore
            )
        else:
            raise ValueError(f"Unknown action: {action.tool}")

class CustomCodeCallbackHandler(CodeCallbackHandler):
    chat_session_callback =None

    def __init__(self, session: "CustomCodeInterpreterSession"):
        super().__init__(session=session)
        self.session = session

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action."""
        print(f"on_agent_action with {action.tool}")
        print(self.session)
        # print(f"⚙️ Running code: ```python\n{action.tool_input['code']}\n```")
        if action.tool == "python":
            callback_output = f"⚙️ Running code: ```python\n{action.tool_input['code']}\n```"
            await self.session.ashow_code(
                f"⚙️ Running code: ```python\n{action.tool_input['code']}\n```"  # type: ignore
            )
            
        elif action.tool == "execute_sql_command":
            callback_output = f"⚙️ Running code: ```psql command\n{action.tool_input['sql']}\n```"
            await self.session.ashow_code(
                f"⚙️ Running code: ```psql command\n{action.tool_input['sql']}\n```"  # type: ignore
            )
            
        elif action.tool in["get_psql_schema_metadata","get_psql_schema_list"]:
            await self.session.ashow_code(
                f"⚙️ Getting database schema data: ```\n{action.tool_input['schema_name']}\n```"  # type: ignore
            )
            callback_output = f"⚙️ Getting database schema data: ```\n{action.tool_input['schema_name']}\n```"
        else:
            raise ValueError(f"Unknown action: {action.tool}")
        
        if self.chat_session_callback:
            # print("self.chat_session_callback not None")
            self.chat_session_callback(
                callback_output
            )
            self.session.memory.append({"AI":callback_output})
            print('session memory:', self.session.memory)
        



class InstantMessageCallbackHandler(BaseCallbackHandler):
    TAG = "InstantMessageCallbackHandler"
    callback=None

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.last_prompt = prompts
        pass
    
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        print(f"{self.TAG} on_llm_end with generations {response.generations}. text ={response.generations[0][0].text}")
        if self.callback and response.generations[0][0].text:
            self.callback(response.generations[0][0].text)
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    
    

class CustomCodeInterpreterSession(CodeInterpreterSession):
    output_file_save_path=None
    on_file_saved_callback=None
    chat_session_callback=None
    agent_executor_callback_func = None
    memory=[]
    

    def __init__(self, model=None, openai_api_key=None, chat_session_callback=None,agent_executor_callback_func = None,**kwargs,) -> None:
        self.codebox = CodeBox()
        self.verbose = kwargs.get("verbose", settings.VERBOSE)
        self.agent_executor: Optional[AgentExecutor] = None
        self.input_files: list[File] = []
        self.output_files: list[File] = []
        self.code_log: list[tuple[str, str]] = []

        self.codebox = CodeBox()
        self.chat_session_callback=chat_session_callback
        self.agent_executor_callback_func=agent_executor_callback_func
        self.tools: list[StructuredTool] = self._tools()
        self.llm: BaseChatModel = self._llm(model, openai_api_key)
        self.normal_llm = ChatOpenAI(
            temperature=0,
            model='gpt-4-0613',
            openai_api_key=openai_api_key,
            max_retries=3,
            request_timeout=60 * 3,
        )
        # self.agent_executor: AgentExecutor = self._agent_executor()
        # self.input_files: list[File] = []
        # self.output_files: list[File] = []
        self.verbose=True
        self.memory = []
    
    def start(self) -> SessionStatus:
        status = SessionStatus.from_codebox_status(self.codebox.start())
        self.agent_executor = self._agent_executor()
        return status

    async def astart(self) -> None:
        # status= await self.codebox.astart()
        status = SessionStatus.from_codebox_status(await self.codebox.astart())
        self.agent_executor = self._agent_executor()
        return status

    def _tools(self) -> list[StructuredTool]:
        return [
            StructuredTool(
                name="python",
                description=
                # TODO: variables as context to the agent
                # TODO: current files as context to the agent
                "Input a string of code to a python interpreter (jupyter kernel). "
                "Variables are preserved between runs. ",
                func=self._run_handler,
                coroutine=self._arun_handler,
                args_schema=CodeInput,
            ),
            StructuredTool(
                name=psql_tools.ExecutePostgressSQLTool().name,
                description=psql_tools.ExecutePostgressSQLTool().description,
                func=psql_tools.ExecutePostgressSQLTool()._run,
                coroutine=psql_tools.ExecutePostgressSQLTool()._arun,
                args_schema=psql_tools.ExecutePostgressSQLTool().args_schema,
            ),
            StructuredTool(
                name=psql_tools.GetPSQLSchemaListTool().name,
                description=psql_tools.GetPSQLSchemaListTool().description,
                func=psql_tools.GetPSQLSchemaListTool()._run,
                coroutine=psql_tools.GetPSQLSchemaListTool()._arun,
                args_schema=psql_tools.GetPSQLSchemaListTool().args_schema,
            ),
            StructuredTool(
                name=psql_tools.GetPSQLSchemaMetadataTool().name,
                description=psql_tools.GetPSQLSchemaMetadataTool().description,
                func=psql_tools.GetPSQLSchemaMetadataTool()._run,
                coroutine=psql_tools.GetPSQLSchemaMetadataTool()._arun,
                args_schema=psql_tools.GetPSQLSchemaMetadataTool().args_schema,
            ),
        ]
    
    def _llm(self, model: Optional[str] = None, openai_api_key: Optional[str] = None) -> BaseChatModel:
        if model is None:
            model = "gpt-4-0613"
            # model='gpt-3.5-turbo-0613'

        if openai_api_key is None:
            if settings.OPENAI_API_KEY is None:
                raise ValueError("OpenAI API key missing.")
            else:
                openai_api_key = settings.OPENAI_API_KEY

        return ChatOpenAI(
            temperature=0.03,
            model=model,
            openai_api_key=openai_api_key,
            max_retries=3,
            request_timeout=60 * 3,
            callbacks=[self.chat_session_callback]
        )  
    
    def _agent(self) -> BaseSingleActionAgent:
        return OpenAIFunctionsAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            system_message=code_interpreter_system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="memory")],
        )

    def _agent_executor(self) -> AgentExecutor:
        print("_agent_executor")
        callback=CustomCodeCallbackHandler(self)
        callback.chat_session_callback=self.agent_executor_callback_func
        return AgentExecutor.from_agent_and_tools(
            agent=self._agent(),
            callbacks=[callback],
            max_iterations=9,
            tools=self.tools,
            verbose=settings.VERBOSE,
            memory=ConversationBufferMemory(memory_key="memory", return_messages=True),
        )
    
    async def _arun_handler(self, code: str):
        """Run code in container and send the output to the user"""
        output: CodeBoxOutput = await self.codebox.arun(code)

        if not isinstance(output.content, str):
            raise TypeError("Expected output.content to be a string.")

        if output.type == "image/png":
            filename = f"image-{uuid.uuid4()}.png"
            file_buffer = BytesIO(base64.b64decode(output.content))
            file_buffer.name = filename
            file=File(name=filename, content=file_buffer.read())
            self.output_files.append(file)
            save_file_path = f"{self.output_file_save_path}{filename}"
            # print(save_file_path)
            file.save(save_file_path)
            self.on_file_saved_callback(save_file_path)
            return f"Image {filename} got send to the user."

        elif output.type == "error":
            if "ModuleNotFoundError" in output.content:
                if package := re.search(
                    r"ModuleNotFoundError: No module named '(.*)'", output.content
                ):
                    await self.codebox.ainstall(package.group(1))
                    return f"{package.group(1)} was missing but got installed now. Please try again."
            else: 
                self.memory.append({'AI found error':output.content})
                # pass
                # TODO: preanalyze error to optimize next code generation
            # if self.verbose:
            #     print("Error:", output.content)

        elif modifications := await get_file_modifications(code, self.normal_llm):
            for filename in modifications:
                if filename in [file.name for file in self.input_files]:
                    continue
                fileb = await self.codebox.adownload(filename)
                if not fileb.content:
                    continue
                file_buffer = BytesIO(fileb.content)
                file_buffer.name = filename
                file = File(name=filename, content=file_buffer.read())
                self.output_files.append(
                    file
                )
                save_file_path = f"{self.output_file_save_path}{str(time.time())}_{filename}"
                print(save_file_path)
                file.save(save_file_path)
                self.on_file_saved_callback(save_file_path)
        else:
            self.memory.append({"AI":output.content})
            self.agent_executor_callback_func(f"Code Execute Result:\n{output.content}")
        return output.content
    
    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        # await self.astop()
        print('___aexit__() do nothing')
    
    
        # super().__aexit__(exc_type, exc_value, traceback)
    async def agenerate_response(
        self,
        user_msg: str,
        files: list[File] = [],
        detailed_error: bool = False,
    ) -> CodeInterpreterResponse:
        """Generate a Code Interpreter response based on the user's input."""
        user_request = UserRequest(content=user_msg, files=files)
        try:
            await self._ainput_handler(user_request)
            response = await self.agent_executor.arun(input=user_request.content)
            return await self._aoutput_handler(response)
        except Exception as e:
            if settings.VERBOSE:
                import traceback

                traceback.print_exc()
            if detailed_error:
                # return CodeInterpreterResponse(
                #     content=f"Error in CodeInterpreterSession: {e.__class__.__name__}  - {e}"
                # )
                pass
            else:
                pass
                # return CodeInterpreterResponse(
                #     content="Sorry, something went while generating your response."
                #     "Please try again or restart the session."
                # )
            return CodeInterpreterResponse(
                content="Sorry, something went while generating your response."
                "Please try again or restart the session."
            )