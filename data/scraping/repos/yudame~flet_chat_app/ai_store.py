from typing import List, Optional

from icecream import ic
from langchain.chat_models import ChatOpenAI

from settings.ai_datatypes import LLMStep, LLMStepName
from settings.ai_config import AI_CONFIG
from stores.base_store import BaseStore
from stores.chat_store import Chat
from stores.user_store import User
from utils import count_tokens, dict_to_cheat_sheet


class AIStore(BaseStore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.page is provided by the BaseStore class
        self._validate_config()
        self.planning_llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            openai_organization=self.openai_api_org,
            model_name=self.open_ai_planning_model,
            temperature=0.1,
        )
        self.chat_llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            openai_organization=self.openai_api_org,
            model_name=self.openai_api_chat_model,
            temperature=0.7,
        )
        self.ai_role = AI_CONFIG.ai_role
        self.ai_user = User(id=AI_CONFIG.ai_role.name, name=AI_CONFIG.ai_role.name)
        self.system_user = User(id="system", name="system")
        self.conversation_start_message_list = [
            {"role": "system", "content": AI_CONFIG.ai_role.intro},
        ]

    def _validate_config(self) -> None:
        assert (
            AI_CONFIG.llm_steps
        ), "openai_api_config.llm_steps is missing required key 'response'"

        self.openai_api_key: str = AI_CONFIG.openai_api.get("api_key")
        self.openai_api_org: str = AI_CONFIG.openai_api.get("org_id")
        default_model: str = AI_CONFIG.openai_api.get("default_model")
        self.openai_api_chat_model: str = (
            AI_CONFIG.openai_api.get("chat_model") or default_model
        )
        self.open_ai_planning_model: str = (
            AI_CONFIG.openai_api.get("planning_model") or default_model
        )

    def prompt(
        self,
        chat: Chat,
        system_prompt: str,
        mode: str = "chat",
        max_tokens: Optional[int] = 90,  # 90 is about 3 long sentences
    ) -> str:
        messages_list: List = []
        # load summary if exists
        if chat.summary:
            messages_list.append(
                {
                    "role": "system",
                    "content": f"Here is a summary of conversation until now: {chat.summary}",
                }
            )
        else:
            messages_list += self.conversation_start_message_list
        # add all message history until now
        messages_list += chat.get_history_as_message_list()
        # add next step
        messages_list.append({"role": "system", "content": system_prompt})

        # convert to text
        full_message_text: str = "\n".join(
            [f"{message['role']}: {message['content']}" for message in messages_list]
        )
        total_tokens = count_tokens(full_message_text, self.openai_api_chat_model)
        if total_tokens > 2048:
            # todo: Handle token overflow
            return "Token limit exceeded"

        match mode:
            case "planning":
                ai_text_response = self.planning_llm.predict(
                    full_message_text, max_tokens=max_tokens
                )
            case "chat":
                ai_text_response = self.chat_llm.predict(
                    full_message_text, max_tokens=max_tokens
                )
            case _:
                raise ValueError(f"Invalid mode: {mode}")

        ai_text_response = ai_text_response.split(f"{AI_CONFIG.ai_role.name}: ")[-1]
        ai_text_response = ai_text_response.split(f"{AI_CONFIG.ai_role.title}: ")[-1]
        return ai_text_response

    def _generate_system_prompt(self, llm_step: LLMStep) -> str:
        system_prompt = f'Now you, {self.ai_user.name}, take a moment to do a "{llm_step.name}" step. {llm_step.instruction}'
        if llm_step.cheat_sheet:
            cheat_sheet_str = (
                llm_step.cheat_sheet
                if isinstance(llm_step.cheat_sheet, str)
                else dict_to_cheat_sheet(llm_step.cheat_sheet)
            )
            system_prompt += f"Use this reference cheat sheet:\n\n{cheat_sheet_str}\n\n"
        if llm_step.name not in (LLMStepName.PRACTICE, LLMStepName.RESPONSE):
            system_prompt += (
                "Be concise. Write a note to yourself using only a few words."
            )
        return system_prompt

    def _execute_planning_step(
        self, chat: Chat, llm_step: LLMStep, max_tokens: int
    ) -> None:
        system_prompt = self._generate_system_prompt(llm_step)

        ai_text_response = self.prompt(
            chat=chat,
            system_prompt=system_prompt,
            mode="planning",
            max_tokens=max_tokens,
        )

        ic(ai_text_response)
        chat.add_message(
            self.system_user, f"your notes on {llm_step.name}:\n{ai_text_response}\n"
        )

    def _take_planning_steps(self, chat: Chat):
        for llm_step in AI_CONFIG.llm_steps:
            if llm_step.name == LLMStepName.RESPONSE:  # should be the last one
                continue
            max_tokens = 120 if llm_step.name == LLMStepName.PRACTICE else 30
            self._execute_planning_step(chat, llm_step, max_tokens)

    def get_next_message(self, chat: Chat):
        self._take_planning_steps(chat=chat)

        response_step = next(
            (step for step in AI_CONFIG.llm_steps if step.name == LLMStepName.RESPONSE),
            None,
        )
        final_instruction = (
            response_step.instruction
            if response_step
            else "take a deep breath and prepare your best response"
        )

        ai_text_response = self.prompt(
            chat=chat, system_prompt=final_instruction, mode="chat"
        )

        chat.add_message(self.ai_user, ai_text_response)
        return ai_text_response
