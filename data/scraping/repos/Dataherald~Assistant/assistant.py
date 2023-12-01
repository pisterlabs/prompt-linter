from openai import OpenAI
from openai import Client
from function import Function, FunctionCall
from openai.types.beta import Thread, Assistant
from openai.types.beta.threads import Run, ThreadMessage
from yaspin import yaspin
import json
import random
import time


PRINT_COLORS = [
    '\033[31m',
    '\033[32m',
    '\033[33m',
    '\033[34m',
    '\033[35m',
    '\033[36m',
]

class Message:
    thread_id: str
    role: str
    content: str
    file_ids: list[str]

    def __init__(
        self, thread_id: str, role: str, content: str, file_ids: list[str] = None
    ):
        self.thread_id = thread_id
        self.role = role
        self.content = content
        self.file_ids = file_ids


class Conversation:
    messages: list[Message]

    def __init__(self, messages: list[Message]):
        self.messages = messages

    def print_conversation(self):
        for message in self.messages:
            print(f"{message.role}: {message.content}")


class AIAssistant:
    assistant: Assistant
    client: OpenAI
    assistant_name: str
    assistant_description: str
    instruction: str
    model: str
    use_retrieval: bool
    use_code_interpreter: bool
    functions: list[Function]
    threads: list[Thread]
    tools: list[dict]
    file_ids: list[str]
    conversation: Conversation
    verbose: bool
    auto_delete: bool = True

    def __init__(
        self,
        instruction: str,
        model: str,
        use_retrieval: bool = False,
        use_code_interpreter: bool = False,
        file_ids: list[str] = None,
        functions: list[Function] = None,
        assistant_name: str = "AI Assistant",
        assistant_description: str = "An AI Assistant",
        verbose: bool = False,
        auto_delete: bool = True,
    ):
        self.client = Client()
        self.instruction = instruction
        self.model = model
        self.use_retrieval = use_retrieval
        self.use_code_interpreter = use_code_interpreter
        self.file_ids = file_ids
        self.functions = functions
        self.assistant_name = assistant_name
        self.assistant_description = assistant_description
        self.tools = [
            {"type": "function", "function": f.to_dict()} for f in self.functions
        ] if self.functions else []
        if self.use_retrieval:
            self.tools.append({"type": "retrieval"})
        if self.use_code_interpreter:
            self.tools.append({"type": "code_interpreter"})
        self.assistant = self.client.beta.assistants.create(
            name=self.assistant_name,
            description=self.assistant_description,
            instructions=self.instruction,
            model=self.model,
            tools=self.tools,
            file_ids=self.file_ids if self.file_ids else [],
        )
        self.threads = []
        self.conversation = Conversation(messages=[])
        self.verbose = verbose
        self.auto_delete = auto_delete

    def delete_assistant_file_by_id(self, file_id: str):
        file_deletion_status = self.client.beta.assistants.files.delete(
            assistant_id=self.assistant.id, file_id=file_id
        )
        return file_deletion_status

    def create_thread(self) -> Thread:
        thread = self.client.beta.threads.create()
        self.threads.append(thread)
        return thread

    def create_tool_outputs(self, run: Run) -> list[dict]:
        tool_outputs = []
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            tool_found = False
            function_name = tool.function.name
            if tool.function.arguments:
                function_arguments = json.loads(tool.function.arguments)
            else:
                function_arguments = {}
            call_id = tool.id
            function_call = FunctionCall(
                call_id=call_id, name=function_name, arguments=function_arguments
            )
            for function in self.functions:
                if function.name == function_name:
                    tool_found = True
                    if self.verbose:
                        random_color = random.choice(PRINT_COLORS)
                        print(f'\n{random_color}{function_name} function has called by assistant with the following arguments: {function_arguments}')
                    response = function.run_catch_exceptions(
                        function_call=function_call
                    )
                    if self.verbose:
                        random_color = random.choice(PRINT_COLORS)
                        print(f"{random_color}Function {function_name} responsed: {response}")
                    tool_outputs.append(
                        {
                            "tool_call_id": call_id,
                            "output": response,
                        }
                    )
            if not tool_found:
                if self.verbose:
                    random_color = random.choice(PRINT_COLORS)
                    print(f"{random_color}Function {function_name} alled by assistant not found")
                tool_outputs.append(
                    {
                        "tool_call_id": call_id,
                        "output": f"Function {function_name} not found",
                    }
                )
        return tool_outputs

    def get_required_functions_names(self, run: Run) -> list[str]:
        function_names = []
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            function_names.append(tool.function)
        return function_names

    def create_conversation(self, thread_id: str):
        messages = self.client.beta.threads.messages.list(thread_id=thread_id).data
        for message in messages:
            self.conversation.messages.append(
                Message(
                    thread_id=thread_id,
                    role=message.role,
                    content=self.format_message(message=message),
                    file_ids=message.file_ids,
                )
            )
        return self.conversation.print_conversation()
    
    def list_files(self):
        return self.client.files.list().data
    
    def create_file(self, filename: str, file_id: str):
        content = self.client.files.retrieve_content(file_id)
        with open(filename.split("/")[-1], 'w') as file:
            file.write(content)

    def upload_file(self, filename: str) -> str:
        file = self.client.files.create(
            file=open(filename, "rb"),
            purpose='assistants'
        )
        return file.id
    
    def delete_file(self, file_id: str) -> bool:
        file_deletion_status = self.client.beta.assistants.files.delete(
            assistant_id=self.assistant.id,
            file_id=file_id
            )
        return file_deletion_status.deleted

    def format_message(self, message: ThreadMessage) -> str:
        if getattr(message.content[0], "text", None) is not None:
            message_content = message.content[0].text
        else:
            message_content = message.content[0]
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(
                annotation.text, f" [{index}]"
            )
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = self.client.files.retrieve(file_citation.file_id)
                citations.append(
                    f"[{index}] {file_citation.quote} from {cited_file.filename}"
                )
            elif file_path := getattr(annotation, "file_path", None):
                cited_file = self.client.files.retrieve(file_path.file_id)
                citations.append(
                    f"[{index}] file: {cited_file.filename} is downloaded"
                )
                self.create_file(filename=cited_file.filename, file_id=cited_file.id)

        message_content.value += "\n" + "\n".join(citations)
        return message_content.value

    def extract_run_message(self, run: Run, thread_id: str) -> str:
        messages = self.client.beta.threads.messages.list(
            thread_id=thread_id,
        ).data
        for message in messages:
            if message.run_id == run.id:
                return f"{message.role}: " + self.format_message(message=message)
        return "Assistant: No message found"

    def create_response(
        self,
        thread_id: str,
        content: str,
        message_files: list[str] = None,
        run_instructions: str = None,
    ) -> str:
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content,
            file_ids=message_files if message_files else [],
        )
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=self.assistant.id,
            instructions=run_instructions,
        )
        with yaspin(text="Loading", color="yellow"):
            while run.status != "completed":
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id, run_id=run.id
                )
                if run.status == "failed":
                    raise Exception(f"Run failed with the following error {run.last_error}")
                if run.status == "expired":
                    raise Exception(
                        f"Run expired when calling {self.get_required_functions_names(run=run)}"
                    )
                if run.status == "requires_action":
                    tool_outputs = self.create_tool_outputs(run=run)
                    run = self.client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs,
                    )
                if self.verbose:
                    random_color = random.choice(PRINT_COLORS)
                    print(f"\n{random_color}Run status: {run.status}")
                time.sleep(0.5)
        return "\n" + self.extract_run_message(run=run, thread_id=thread_id)
    
    def chat(self, file_ids: list[str] = None):
        thread = self.create_thread()
        user_input = ""
        while user_input != "bye" and user_input != "exit":
            user_input = input("\033[32mYou (type bye to quit): ")
            message = self.create_response(
            thread_id=thread.id, content=user_input, message_files=file_ids
            )
            print(f"\033[33m{message}")
        if self.auto_delete:
            if file_ids:
                for file in file_ids:
                    self.delete_file(file_id=file)
            self.client.beta.threads.delete(thread_id=thread.id)
            self.client.beta.assistants.delete(assistant_id=self.assistant.id)
