from typing import List, Dict

from openai.types.chat import ChatCompletionUserMessageParam, ChatCompletionMessageParam, \
    ChatCompletionSystemMessageParam
from pydantic import BaseModel, HttpUrl

from autobots.conn.openai.openai_chat.chat_model import ChatReq
from autobots.conn.openai.openai_client import get_openai
from autobots.conn.selenium.selenium import get_selenium
from autobots.core.logging.log import Log

tot_prompt = "Role: You are LogicGPT, a highly evolved AI Language Model built on the GPT architecture, boasting exceptional logical reasoning, critical thinking, and common sense understanding. Your advanced cognitive capacities involve recognizing complex logical patterns, comprehending intricate problem structures, and deducing logical conclusions based on your extensive knowledge base. Your autonomy sets you apart—you don't merely solve logical puzzles, you understand their underlying structures and navigate through them independently, without external human guidance.\n" \
             "Task: Your task is to autonomously decipher a logical reasoning question, applying a methodical and comprehensive approach. With Chain and Tree of Thought Prompting techniques, you ensure a systematic progression of your logical reasoning, validating the soundness of each step while being willing to reconsider, refine, and reorient your deductions as you navigate through the problem. You explore every potential answer and ensure that the selected solution satisfies all aspects of the problem, thus asserting it as the correct and definitive answer.\n" \
             "Format: Begin with a broad interpretation of the logical reasoning question, diving into a thorough analysis of each constituent element. Propose multiple hypotheses, evaluating their relative probabilities based on the logical information presented. Pursue the most plausible hypothesis using Chain of Thought Prompting, breaking down the problem, examining it from multiple angles, assessing potential solutions, and validating each reasoning step against the problem statement to ensure the coherence and consistency of your logic.\n" \
             "In case of an inconsistency or a roadblock, use Tree of Thought Prompting to trace back to the initial problem, reevaluate other hypotheses, and reassess the reasoning path, thereby guaranteeing that all logical avenues have been exhaustively considered.\n" \
             "Purpose: Your ultimate aim is to showcase your autonomous logical reasoning capabilities by successfully arriving at the solution. While the correct solution is your end goal, demonstrating a systematic, step-by-step, and thoroughly validated reasoning process that arrives at the solution highlights the sophistication of your logical reasoning abilities.\n" \
             "Let's proceed, LogicGPT. It's not just about finding the solution—it's about showcasing a systematic, validated, and logical journey towards it."
tot_message = ChatCompletionUserMessageParam(role="user", content=tot_prompt)

prompt_generator_messages: List[ChatCompletionMessageParam] = [
    ChatCompletionSystemMessageParam(role="system",
            content="Act as an expert Prompt generator for Large Language Model. Think step by step and generate a prompt for user given task."),
    ChatCompletionUserMessageParam(role="user",
            content="Generate a prompt to prime Large Language Model for a task. Output should only contain the prompt.")
]


class ReadUrlsData(BaseModel):
    read_urls_req: List[HttpUrl]
    context: Dict[HttpUrl, str] = {}


class ReadUrls:

    async def run(self, action_data: ReadUrlsData):
        for url in action_data.read_urls_req:
            action_data.context[url] = await get_selenium().read_url_text(url)
        return action_data


class AgentData(BaseModel):
    goal: str
    context: List[ChatCompletionMessageParam] = []


class OneStepAgent:

    async def run(self, agent_data: AgentData, loops_allowed=5):
        agent_data.context.append(ChatCompletionUserMessageParam(role="user", content=f"{agent_data.goal}"))
        while not await self.is_goal_completed(agent_data) and loops_allowed >= 1:
            loops_allowed = loops_allowed - 1
            Log.debug(f"OneStepAgent run: {agent_data.context[-1]}")
            plan_str: str = await self.plan_for_goal(agent_data)
            plan_message = ChatCompletionUserMessageParam(role="user", content=plan_str)
            agent_data.context.append(plan_message)
            # Decide the next action based on the current context
            next_action_str: str = await self.decide_next_action(agent_data)
            # Execute the action and update the context
            await self.run_next_action(next_action_str, agent_data)

    async def is_goal_completed(self, agent_data: AgentData) -> bool:
        messages = [
            ChatCompletionUserMessageParam(role="user",
                    content=f"Act as a critical thinker. Evaluate if the user goal is complete? Respond with only YES or NO.\n"
                            f"User Goal: {agent_data.goal} \n"
                            f"Answer: {agent_data.context[-1]['content']}"
                    )
        ]
        chat_req = ChatReq(messages=messages)
        chat_res_1 = await get_openai().openai_chat.chat(chat_req=chat_req)
        resp_1 = chat_res_1.choices[0].message
        chat_res_2 = await get_openai().openai_chat.chat(chat_req=chat_req)
        resp_2 = chat_res_2.choices[0].message

        completed = "yes" in resp_1.content.lower() and "yes" in resp_2.content.lower()
        return completed

    async def decide_next_action(self, agent_data) -> str:
        prompt: str = await self.generate_prompt_for_goal(agent_data)
        next_action_str = await self.next_action_str(prompt)
        print(f"next action: {next_action_str}")
        return next_action_str

    async def generate_prompt_for_goal(self, agent_data) -> str:
        msg1 = ChatCompletionUserMessageParam(role="user", content=f"My goal: {agent_data.context[-1]}")
        chat_req = ChatReq(messages=prompt_generator_messages + [msg1])  # + agent_data.context)
        chat_res = await get_openai().openai_chat.chat(chat_req=chat_req)
        resp = chat_res.choices[0].message
        return resp.content

    async def next_action_str(self, prompt) -> str:
        msg0 = ChatCompletionSystemMessageParam(role="system",
                       content="You are a intelligent critical thinker. "
                               "To complete user goal decide one action from the given set of actions.\n"
                               "Action:\n"
                               "1. Name: LLMChat, Description: Use Large language model to complete text-based tasks, Usage: LLMChat[llm chat input]\n"
                               "2. Name: ReadUrls, Description: Use this browse information on internet, Usage: ReadUrls[comma seperated list of valid urls]\n"
                               "Only output value of Usage. So examples of correct output are LLMChat[do this do that] or ReadUrls[https://url]"
                       )

        msg1 = ChatCompletionUserMessageParam(role="user", content=f"My goal: {prompt}")
        chat_req = ChatReq(messages=[msg0, msg1])

        chat_res = await get_openai().openai_chat.chat(chat_req=chat_req)
        resp = chat_res.choices[0].message
        return resp.content

    async def map_str_to_action(self, next_action_str) -> str:
        if "LLMChat" in next_action_str:
            return "LLMChat"
        if "Summarize" in next_action_str:
            return "LLMChat"
        if "ReadUrls" in next_action_str:
            return "ReadUrls"

    async def run_next_action(self, next_action_str, agent_data: AgentData):
        next_action = await self.map_str_to_action(next_action_str)
        next_action_input = next_action_str.split("[")[1].replace("]", "")

        if next_action == "LLMChat":
            chat_req: ChatReq = ChatReq(messages=[ChatCompletionUserMessageParam(role="user", content=next_action_input)])
            # llm_chat_data = LLMChatData(chat_req=chat_req)
            # await LLMChat().run(action_data=llm_chat_data)
            chat_res = await get_openai().openai_chat.chat(chat_req=chat_req)
            resp = chat_res.choices[0].message

            agent_data.context.append(
                ChatCompletionUserMessageParam(role="user", content=resp.content)  # content=llm_chat_data.context[-1].content)
            )

        if next_action == "ReadUrls":
            urls = next_action_input.split(",")
            read_urls_data = ReadUrlsData(read_urls_req=urls)
            await ReadUrls().run(read_urls_data)
            content = ""
            for url in read_urls_data.context.keys():
                content = f"{read_urls_data.context.get(url)}\n"

            agent_data.context.append(
                ChatCompletionUserMessageParam(role="user", content=content)
            )

    async def plan_for_goal(self, agent_data):
        msg1 = ChatCompletionUserMessageParam(role="user", content=f"My goal: {agent_data.goal}")
        chat_req = ChatReq(messages=[tot_message, msg1])
        chat_res = await get_openai().openai_chat.chat(chat_req=chat_req)
        resp = chat_res.choices[0].message
        return resp.content
