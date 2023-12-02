import traceback
import config


#from flask import request
import json
import os
import re
import pika
import faiss
import time

from pydantic import BaseModel, Field
from datetime import datetime, date, time, timezone, timedelta
from typing import Any, Dict, Optional, Type
from bots.rabbit_handler import RabbitHandler
from common.rabbit_comms import consume, publish, publish_action, publish_actions, publish_email_card, publish_list, publish_draft_card, publish_draft_forward_card, send_to_bot
from common.utils import tool_description, tool_error

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool



class PlannerBot(BaseTool):
    parameters = []
    optional_parameters = []
    name = "PLANNER"
    summary = """useful for when you need to breakdown objectives into a list of tasks. """
    parameters.append({"name": "text", "description": "task or objective to create a plan for" })
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = False

    tools = []
    def init(self, tools):
        self.tools = tools
    
    #callbacks
    def get_plan_input(self, question):
        timeout = time.time() + 60*5   # 1 minutes from now
        #Calculate plan
        plan = self.model_response(question, tools)
        #publish(plan)
        publish_action(plan,"continue","pass")
        publish("Would you like to make any changes to the plan above?")
        #loop until human happy with plan
        #contents = []
        while True:
            msg = consume()
            if msg:
                question = msg
                if question.lower() == "continue":
                    return plan
                if question.lower() == "pass":
                    return "stop"
                if question.lower() == "break":
                    return "stop"
                else:
                    new_prompt = f"Update the following plan: {plan} using the following input: {question}"
                    plan = self.model_response(new_prompt, tools)
                    publish_action(plan,"continue","pass")
                    publish("Would you like to make any changes to the plan above?")
                    timeout = time.time() + 60*5
            if time.time() > timeout:
                return "stop"
            time.sleep(0.5)
        return plan

    def _run(self, text: str, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        try:
            print(text)

            

            plan = self.model_response(text, self.tools)
            buttons = [("Proceed", "Thinking step by step, action each of the following items in the plan using the tools available plan: "+ plan)]
            if publish.lower() == "true":
                publish_actions(plan,buttons)
                return config.PROMPT_PUBLISH_TRUE
            else:
                return plan
        except Exception as e:
            traceback.print_exc()
            return tool_error(e, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("PlannerBot does not support async")

    #this bot needs to provide similar commands as autoGPT except the commands are based on Check Email, Check Tasks, Load Doc, Load Code etc.
    def model_response(self, text, tools=None):
        try:
            #config
            
            
            current_date_time = datetime.now()
            handler = RabbitHandler()

            tool_details = ""
            for tool in tools:
                tool_details = tool_details + "\nName: " + tool.name + "\nDescription: " + tool.description + "\n"
            template="""You are a planner who can identify the right tool for the objective. If more then one tool is required, come up with a short todo lists of 1 to 5 tasks. 
            The objective is: {objective}.
            You have the following tools available {tools}
            """
            prompt = PromptTemplate(
                input_variables=["objective", "tools"], 
                template=template
            )

            chatgpt_chain = LLMChain(
                llm=ChatOpenAI(temperature=0), 
                prompt=prompt, 
                verbose=True,
                callbacks=[handler]
            )
            query = f"Given the current data and time of {current_date_time}, {text}"
            response = chatgpt_chain.run(objective=query, tools=tool_details)

            return response
        except Exception as e:
            traceback.print_exc()
            return tool_error(e, self.description)


