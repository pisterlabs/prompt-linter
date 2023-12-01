import os.path
import shutil
from collections import defaultdict
from datetime import datetime
from multiprocessing import Lock
from typing import Optional, Dict, Any

from langchain import PromptTemplate, FAISS
from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import MessagesPlaceholder, \
    HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, AIMessage
from yid_langchain_extensions.agent.simple_agent import SimpleAgent
from yid_langchain_extensions.output_parser.action_parser import ActionParser
from yid_langchain_extensions.output_parser.utils import get_dict_without_extra_fields, \
    format_dict_to_json_md
from yid_langchain_extensions.tools.utils import format_tools, format_tool_names, FinalAnswerTool

from agents.stm_savable import SavableWindowMemory
from agents.utils import format_now, get_self_criticism_thought, get_thought_thought, get_important_info_thought, \
    get_conversation_summary_thought, get_end_detection_thought
from agents.web_researcher import WebResearcherAgent


class HelperAgent:
    def __init__(self, save_path: str, prompts: Dict[str, str], web_researcher_agent: WebResearcherAgent):
        self.prompts = prompts
        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        self.smart_llm = ChatOpenAI(model_name="gpt-4-0613", temperature=0)
        self.fast_llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0)

        final_answer_tool = FinalAnswerTool()
        web_search_tool = web_researcher_agent.as_tool()

        self.tools = [final_answer_tool, web_search_tool]
        self.output_parser = ActionParser.from_extra_thoughts(pre_thoughts=[
            get_thought_thought(), get_self_criticism_thought()
        ], after_thoughts=[
            get_end_detection_thought(),
            get_conversation_summary_thought(),
            get_important_info_thought(self.prompts["important_memory_description"]),
        ])
        self.format_message = PromptTemplate.from_template(
            self.output_parser.get_format_instructions(), template_format="jinja2").format(
            tool_names=format_tool_names(self.tools)
        )
        self.long_term_memory_embeddings = OpenAIEmbeddings()
        self._memory_locks = defaultdict(Lock)
        self._locks_lock = Lock()
        self.k_last_messages = 8

    def _get_memory_lock(self, user_id: int) -> Lock:
        with self._locks_lock:
            return self._memory_locks[user_id]  # noqa

    def _get_user_dir(self, user_id: int) -> str:
        user_dir = os.path.join(self.save_path, str(user_id))
        if not os.path.isdir(user_dir):
            os.makedirs(user_dir)
        return user_dir

    def _get_user_ltm_path(self, user_id: int) -> str:
        return os.path.join(self._get_user_dir(user_id=user_id), "ltm")

    def _load_short_term_memory(self, user_id: int) -> SavableWindowMemory:
        def add_date(full_input: Dict[str, Any]) -> Dict[str, Any]:
            updated_input = f"{format_now()}\n{full_input['input']}"
            return {**full_input, "input": updated_input}

        def strip_raw_output(full_output: Dict[str, Any]) -> Dict[str, Any]:
            if "raw_output" not in full_output:
                stripped_raw_output_dict = get_dict_without_extra_fields(full_output, ["action", "action_input"])
                stripped_raw_output = format_dict_to_json_md(stripped_raw_output_dict)
                return {**full_output, "raw_output": stripped_raw_output}
            return full_output

        with self._get_memory_lock(user_id=user_id):
            return SavableWindowMemory.load(
                llm=self.fast_llm, max_token_limit=6000,
                memory_key="chat_history", return_messages=True,
                save_path=self._get_user_dir(user_id=user_id),
                input_key="input", input_preprocessor=add_date,
                output_key="raw_output", output_preprocessor=strip_raw_output,
                k=self.k_last_messages
            )

    def _get_conversation_summary_path(self, user_id: int) -> str:
        return os.path.join(self._get_user_dir(user_id=user_id), "conversation_summary.txt")

    def _check_conversation_summary(self, user_id: int) -> bool:
        return os.path.exists(self._get_conversation_summary_path(user_id=user_id))

    def _load_conversation_summary(self, user_id: int) -> str:
        with self._get_memory_lock(user_id=user_id):
            if os.path.exists(conversation_summary_path := self._get_conversation_summary_path(user_id=user_id)):
                with open(conversation_summary_path, "r") as f:
                    return f.read()
            else:
                return "No conversation summary yet."

    def _update_conversation_summary(self, user_id: int, new_summary: str):
        with self._get_memory_lock(user_id=user_id):
            with open(self._get_conversation_summary_path(user_id=user_id), "w") as f:
                f.write(new_summary)

    def _get_memory_about_user_path(self, user_id: int) -> str:
        return os.path.join(self._get_user_dir(user_id=user_id), "memory_about_user.txt")

    def _load_memory_about_user(self, user_id: int) -> str:
        with self._get_memory_lock(user_id=user_id):
            if os.path.exists(memory_about_user_path := self._get_memory_about_user_path(user_id=user_id)):
                with open(memory_about_user_path, "r") as f:
                    return f.read()
            else:
                return "Nothing is known about this user yet."

    def _update_memory_about_user(self, user_id: int, new_memory: str):
        with self._get_memory_lock(user_id=user_id):
            with open(self._get_memory_about_user_path(user_id=user_id), "w") as f:
                f.write(new_memory)

    def _load_long_term_memory(self, user_id: int) -> Optional[FAISS]:
        with self._get_memory_lock(user_id=user_id):
            ltm_path = self._get_user_ltm_path(user_id=user_id)
            if os.path.exists(ltm_path):
                return FAISS.load_local(ltm_path, self.long_term_memory_embeddings)
            return None

    def _create_long_term_memory(self, first_memory: str) -> FAISS:
        return FAISS.from_texts([first_memory], self.long_term_memory_embeddings,
                                metadatas=[{"date": datetime.now().isoformat()}])

    def _add_to_long_term_memory(self, user_id: int, new_long_term_memory: str):
        long_term_memory = self._load_long_term_memory(user_id=user_id)
        with self._get_memory_lock(user_id=user_id):
            if long_term_memory:
                long_term_memory.add_texts([new_long_term_memory], metadatas=[{"date": datetime.now().isoformat()}])
            else:
                long_term_memory = self._create_long_term_memory(first_memory=new_long_term_memory)
            long_term_memory.save_local(self._get_user_ltm_path(user_id=user_id))

    def forget(self, user_id: int):
        short_term_memory = self._load_short_term_memory(user_id=user_id)
        with self._get_memory_lock(user_id=user_id):
            short_term_memory.clear()
            ltm_path = self._get_user_ltm_path(user_id=user_id)
            if os.path.exists(ltm_path):
                shutil.rmtree(ltm_path)
            if os.path.exists(conversation_summary_path := self._get_conversation_summary_path(user_id=user_id)):
                os.remove(conversation_summary_path)
            if os.path.exists(memory_about_user_path := self._get_memory_about_user_path(user_id=user_id)):
                os.remove(memory_about_user_path)
        print(f"Memory about user {user_id} removed")

    def _get_relevant_ltm(
            self, user_id: int, short_term_memory: BaseChatMemory, long_term_memory: Optional[FAISS]) -> Optional[str]:
        if long_term_memory is None:
            return None
        with self._get_memory_lock(user_id=user_id):
            short_term_memory.return_messages = False
            short_term_context = short_term_memory.load_memory_variables({})["chat_history"]
            short_term_memory.return_messages = True
            relevant_document = long_term_memory.similarity_search(short_term_context, k=1)[0]
            date = datetime.fromisoformat(relevant_document.metadata["date"]).strftime('%Y-%m-%d')
            thought = "Thought (user does not see it):\n" \
                      f"Hm, that reminds me another conversation I had {date} with user:\n" \
                      f"{relevant_document.page_content}"
            return thought

    def _format_conversation_summary(self, conversation_summary: str) -> str:
        result = \
            "CURRENT_CONVERSATION SUMMARY:\n" \
            f"{conversation_summary}\n\n" \
            f"Following messages are the last {self.k_last_messages} messages of that conversation"
        return result

    @staticmethod
    def _format_memory_about_user(memory_about_user: str) -> str:
        result = \
            "IMPORTANT INFO ABOUT USER:\n" \
            "What have I learned about user so far:\n" \
            f"{memory_about_user}"
        return result

    def _initialise_agent(
            self, user_id: int, short_term_memory: BaseChatMemory, conversation_summary: str,
            memory_about_user: str, long_term_memory: Optional[FAISS]) -> AgentExecutor:
        system_message = PromptTemplate.from_template(self.prompts["prefix"], template_format="jinja2").format(
            date=format_now()
        )
        messages = [
            SystemMessage(content=system_message),
            AIMessage(content=self._format_memory_about_user(memory_about_user)),
            AIMessage(content=self._format_conversation_summary(conversation_summary)),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{{input}}", "jinja2"),
        ]
        if (relevant_ltm := self._get_relevant_ltm(user_id, short_term_memory, long_term_memory)) is not None:
            messages.append(AIMessage(content=relevant_ltm))
        messages.extend([
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            SystemMessage(content=format_tools(self.tools)),
            SystemMessage(content=self.format_message),
        ])
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        agent_executor = SimpleAgent.from_llm_and_prompt(
            llm=self.smart_llm,
            prompt=prompt,
            output_parser=self.output_parser,
            stop_sequences=self.output_parser.stop_sequences,
        ).get_executor(tools=self.tools, memory=short_term_memory, verbose=True)
        return agent_executor

    async def arun(self, user_id: int, request: str) -> str:
        try:
            short_term_memory = self._load_short_term_memory(user_id=user_id)
            conversation_summary = self._load_conversation_summary(user_id=user_id)
            memory_about_user = self._load_memory_about_user(user_id=user_id)
            long_term_memory = self._load_long_term_memory(user_id=user_id)
            agent = self._initialise_agent(
                user_id, short_term_memory, conversation_summary, memory_about_user, long_term_memory)
            answer = await agent.acall(inputs={"input": request}, return_only_outputs=True)
            if "new_topic_started" in answer and answer["new_topic_started"] and \
                    self._check_conversation_summary(user_id):
                self._add_to_long_term_memory(user_id, conversation_summary)
                self._clear_short_term_memory(short_term_memory)
                self._clear_conversation_summary(user_id)
            elif "updated_conversation_summary" in answer:
                self._update_conversation_summary(user_id=user_id, new_summary=answer["updated_conversation_summary"])
            if "updated_important_info" in answer:
                self._update_memory_about_user(user_id=user_id, new_memory=answer["updated_important_info"])
            return answer["output"]
        except Exception as e:
            return f"Error in telegram bot: {e}. Report it to developer."

    def _clear_conversation_summary(self, user_id: int):
        with self._get_memory_lock(user_id=user_id):
            conversation_summary_path = self._get_conversation_summary_path(user_id=user_id)
            if os.path.exists(conversation_summary_path):
                os.remove(conversation_summary_path)

    @staticmethod
    def _clear_short_term_memory(memory: BaseChatMemory):
        last_request = memory.chat_memory.messages[-2].content
        last_answer = memory.chat_memory.messages[-1].content
        memory.clear()
        memory.save_context({"input": last_request}, {"raw_output": last_answer})

    async def after_message(self, user_id: int):
        pass
