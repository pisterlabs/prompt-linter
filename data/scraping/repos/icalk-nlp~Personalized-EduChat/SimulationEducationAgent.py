import argparse

from chain.Educhat import Educhat
from chain.prompt_template.prompt_template import assistant_inception_prompt, user_inception_prompt
from typing import List
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)


class SimulationEducationAgent:
    def __init__(
            self,
            system_message: SystemMessage,
            model
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    @staticmethod
    def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
        assistant_sys_template = SystemMessagePromptTemplate.from_template(
            template=assistant_inception_prompt
        )
        assistant_sys_msg = assistant_sys_template.format_messages(
            assistant_role_name=assistant_role_name,
            user_role_name=user_role_name,
            task=task,
        )[0]

        user_sys_template = SystemMessagePromptTemplate.from_template(
            template=user_inception_prompt
        )
        user_sys_msg = user_sys_template.format_messages(
            assistant_role_name=assistant_role_name,
            user_role_name=user_role_name,
            task=task,
        )[0]

        return assistant_sys_msg, user_sys_msg

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages

    def step(
            self,
            input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model.predict_messages(messages)
        self.update_messages(output_message)

        return output_message


def get_specified_task_content(llm, assistant_role_name, user_role_name, task, word_limit):
    task_specifier_sys_msg = SystemMessage(content="你可以使任务更具体。")
    task_specifier_prompt = """这是{assistant_role_name}将帮助{user_role_name}完成的教学任务：{task}。
    请使用创造性和想象力来使其更具体。请在{word_limit}个字或更少的字数内回复指定的任务。不要添加其他内容。"""
    task_specifier_template = HumanMessagePromptTemplate.from_template(
        template=task_specifier_prompt
    )
    task_specify_agent = SimulationEducationAgent(task_specifier_sys_msg, llm)
    task_specifier_msg = task_specifier_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
        word_limit=word_limit,
    )[0]
    specified_task_msg = task_specify_agent.step(task_specifier_msg)
    print(f"Specified task: {specified_task_msg.content}")
    return specified_task_msg.content


if __name__ == '__main__':
    llm = Educhat()
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=False, help="task to be completed",
                        default="师生共同完成对唐诗的解读")
    parser.add_argument("--word_limit", type=int, required=False, help="word limit for task brainstorming", default=100)
    args = parser.parse_args()
    task = args.task
    word_limit = args.word_limit  # word limit for task brainstorming
    assistant_role_name = "Teacher"
    user_role_name = "Student"
    specified_task_content = get_specified_task_content(llm, assistant_role_name, user_role_name, task, word_limit)

    assistant_sys_msg, user_sys_msg = SimulationEducationAgent.get_sys_msgs(
        assistant_role_name, user_role_name, specified_task_content
    )
    # TODO: 改进memory功能。建立不同的Educhat对象，进行交流
    assistant_agent = SimulationEducationAgent(assistant_sys_msg, Educhat())
    user_agent = SimulationEducationAgent(user_sys_msg, Educhat())

    # Reset agents
    assistant_agent.reset()
    user_agent.reset()

    # Initialize chats
    assistant_msg = HumanMessage(
        content=(
            f"{user_sys_msg.content}. "
            "现在开始逐个给我介绍。只回复教学指导和输入"
        )
    )

    user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
    user_msg = assistant_agent.step(user_msg)

    print(f"Original task prompt:\n{task}\n")
    print(f"Specified task prompt:\n{specified_task_content}\n")

    chat_turn_limit, n = 5, 0
    while n < chat_turn_limit:
        n += 1
        user_ai_msg = user_agent.step(assistant_msg)
        user_msg = HumanMessage(content=user_ai_msg.content)
        print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")

        assistant_ai_msg = assistant_agent.step(user_msg)
        assistant_msg = HumanMessage(content=assistant_ai_msg.content)
        print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
        if "<TASK_DONE>" in user_msg.content:
            break
