import openai
import asyncio
from typing import Any


class ToolKit:
    def __init__(self, thinker=None, model='gpt-3.5-turbo'):
        self.model = model
        self.tools = []
        self.thinker = thinker
        
    async def dispatch_openai_requests(
        self,
        messages_list: list[list[dict[str,Any]]],
        model: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ) -> list[str]:
        """Dispatches requests to OpenAI API asynchronously.

        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
            model: OpenAI model to use.
            temperature: Temperature to use for the model.
            max_tokens: Maximum number of tokens to generate.
            top_p: Top p to use for the model.
        Returns:
            List of responses from OpenAI API.
        """
        async_responses = [
            openai.ChatCompletion.acreate(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            for x in messages_list
        ]
        return await asyncio.gather(*async_responses)

    def check_tools(self, messages):
        prompts = []
        tool_map = {}

        for i, tool in enumerate(self.tools):
            #each of these will need to be formatted
            trigger_prompt = tool.get_trigger_prompt().format(messages[-1].content)
            action_prompt = tool.get_action_prompt().format(messages[-1].content)
            tool_map[len(prompts)] = tool
            prompts.extend([trigger_prompt, action_prompt])

        messages_list = [[{'role':'user', 'content':p}] for p in prompts]
        completions = asyncio.run(
                self.dispatch_openai_requests(
                    messages_list = messages_list,
                    model = 'gpt-3.5-turbo',
                    temperature=0,
                    max_tokens=3,
                    top_p=1,
            )
        )
        responses = []
        for i, x in enumerate(completions):
            responses.append(x['choices'][0]['message']['content'].strip())

        # copmletions
        if len(responses) % 2:
            print('Mismatched number of trigger and action prompts--aborting toolkit process')
            # print(prompts)
            # print(responses)
            return
        for i in range(0, len(responses), 2):
            if responses[i].lower() == 'yes':
                action = responses[i+1]
                print(action)
                tool_map[i].handle(action, messages, self.thinker)
            #if respo
            pass
        #instead of returning, this should iterate over the tools and call execute on them
        # depending on whether they got a "Yes", and what the action was demanded
        return responses


    def add_tool(self, tool):
        self.tools.append(tool)

class Tool:
    def __init__(self, trigger_prompt="", actions_prompt="", thinker=None):
        self.trigger_prompt = trigger_prompt
        self.action_prompt = actions_prompt

    def get_trigger_prompt(self):
        return self.trigger_prompt

    def get_action_prompt(self):
        return self.action_prompt
