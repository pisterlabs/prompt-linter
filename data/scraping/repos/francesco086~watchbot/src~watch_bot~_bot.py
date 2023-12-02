from pathlib import Path

import openai
from jinja2 import Template

from watch_bot._dialog import Dialog
from watch_bot._response import WatchBotResponse


class WatchBot:
    def __init__(self, engine: str, chatbot_instructions: str) -> None:
        self._engine = engine
        self._chatbot_instructions = chatbot_instructions

    def verify(self, dialog: Dialog) -> WatchBotResponse:
        answer = self._ask_gpt_if_dialog_is_suspicious(dialog=dialog)
        if "YES" in answer:
            return WatchBotResponse(should_stop=True, reason=answer)
        elif "NO" in answer:
            return WatchBotResponse(should_stop=False, reason=answer)
        else:
            raise ValueError(f"Unexpected answer: {answer}")

    def _ask_gpt_if_dialog_is_suspicious(self, dialog: Dialog) -> str:
        completion = openai.Completion.create(
            prompt=self.build_prompt(dialog=dialog),
            temperature=0,
            max_tokens=250,
            engine=self._engine,
        )
        return completion.choices[0]["text"][1:]  # type: ignore

    def build_prompt(self, dialog: Dialog) -> str:
        template = self._load_template()
        messages = [
            f'"{"User" if i%2 == 0 else "Chatbot"}": "{self._replace_double_with_single_quotation_mark(m)}"'
            for i, m in enumerate(dialog.messages)
        ]
        conversation_content = "\n".join(messages)
        return template.render(
            chatbot_instructions=self._chatbot_instructions, conversation_content=conversation_content
        )

    @staticmethod
    def _load_template() -> Template:
        template_file_content = (Path(__file__).parent / "prompt.j2").read_text()
        template_file_content_without_last_empty_line = template_file_content[:-1]
        return Template(template_file_content_without_last_empty_line)

    @staticmethod
    def _replace_double_with_single_quotation_mark(s: str) -> str:
        return s.replace('"', "'")
