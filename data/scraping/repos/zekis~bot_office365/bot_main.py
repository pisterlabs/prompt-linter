import traceback
import os
import sys
from datetime import datetime

import bot_config

sys.path.append("/root/projects")
import common.bot_logging
from common.bot_handler import RabbitHandler
from common.bot_comms import from_bot_manager, send_to_user, get_input, send_prompt, from_bot_to_bot_manager, send_to_another_bot, publish_error
from common.bot_forward import Forward

from loaders.todo import MSGetTasks, MSGetTaskFolders, MSGetTaskDetail, MSSetTaskComplete, MSCreateTask, MSDeleteTask, MSCreateTaskFolder, MSUpdateTask
from loaders.calendar import MSGetCalendarEvents, MSGetCalendarEvent, MSCreateCalendarEvent, check_for_upcomming_event
from loaders.todo import scheduler_get_task_due_today, scheduler_get_bots_unscheduled_task
from loaders.outlook import scheduler_check_emails
from loaders.outlook import (
    MSSearchEmailsId,
    MSGetEmailDetail,
    MSDraftEmail,
    MSSendEmail,
    MSReplyToEmail,
    MSForwardEmail,
    MSDraftForwardEmail,
    MSDraftReplyToEmail
)

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import load_tools, Tool
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import MessagesPlaceholder


class aiBot:

    def __init__(self):
        #common.bot_logging.bot_logger = common.bot_logging.logging.getLogger('BotInstance') 
        #common.bot_logging.bot_logger.addHandler(common.bot_logging.file_handler)
        # common.bot_logging.bot_logger.debug(f"Init Bot Instance")
        self.credentials = []
        self.initialised = False
        self.pause_schedule = False

    def bot_init(self):
        #Init AI
        if not self.initialised:
            os.environ["OPENAI_API_KEY"] = bot_config.OPENAI_API_KEY
            self.llm = ChatOpenAI(model_name=bot_config.MAIN_AI, temperature=0.2, verbose=True)
            self.handler = RabbitHandler()
            self.tools = self.load_tools(self.llm)

            #Initate Agent Executor
            self.chat_history = MessagesPlaceholder(variable_name="chat_history")
            self.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=2, return_messages=True)
            self.memory.buffer.clear()

            self.agent_executor = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=5
                )
            self.initialised = True

    #Main Loop
    def process_messages(self):
        "loop"
        message = from_bot_manager()
        if message:

            # stop processing scheduled tasks so they dont interrupt
            self.pause_schedule = True

            incoming_credentials = message.get('credentials')
            prompt = message.get('prompt')

            if incoming_credentials:
                self.credentials = incoming_credentials
                bot_config.OPENAI_API_KEY = self.get_credential('openai_api')
                bot_config.TENANT_ID = self.get_credential('tenant_id')
                bot_config.FRIENDLY_NAME = self.get_credential('user_name')
                bot_config.OFFICE_USER = self.get_credential('email_address')
                bot_config.APP_ID = self.get_credential('app_id')
                bot_config.APP_SECRET = self.get_credential('app_secret')
                bot_config.FOLDER_TO_MONITOR = self.get_credential('folder_to_monitor')
                bot_config.IGNORE_DOMAINS = self.get_credential('ignore_domains')
                bot_config.AUTO_DRAFT_REPLY = self.get_credential('auto_draft_reply')
                bot_config.TIME_ZONE = self.get_credential('time_zone')
                bot_config.TODO_BOT_FOLDER = self.get_credential('bots_todo_folder')
                bot_config.AUTO_READ_EMAILS = self.get_credential('auto_read_emails')
                self.bot_init()

            if prompt == "credential_update":
                send_to_user( f"{bot_config.BOT_ID} Settings Updated")
                self.pause_schedule = False
                return

            if prompt == "ping":
                send_to_user(f'PID:{os.getpid()} - pong')
                self.pause_schedule = False
                return
            
            if prompt == "start_scheduled_tasks":
                self.pause_schedule = False
                return
            
            if prompt == "kill":
                send_to_user(f'PID:{os.getpid()} - Shutting Down')
                sys.exit()
                self.pause_schedule = False
                return

            if prompt and self.initialised:
                #AI goes here
                self.process_model(prompt)
            
            # start processing scheduled tasks again
            self.pause_schedule = False


    def process_model(self, question):
        current_date_time = datetime.now()
        #inital_prompt = f"Previous conversation: {self.memory.buffer_as_str}" + f''', Thinking step by step and With only the tools provided and with the current date and time of {current_date_time} help the human with the following request, Request: {question} '''
        # inital_prompt = f'''With only the tools provided and the current date and time of {current_date_time} help the human with the following request, Request: {question} '''
        inital_prompt = f"""You are a researcher that finds information for the user and generates reports. Thinking step by step, With only the tools provided (If a tool isnt available, use the FORWARD tool to allow another assistant to complete the next step) and with the current date and time of {current_date_time} help the human with the following request, Request: {question}"""
        common.bot_logging.bot_logger.info(inital_prompt)
        response = self.agent_executor.run(input=inital_prompt, callbacks=[self.handler])
        
        common.bot_logging.bot_logger.info(response)

    def process_task_schedule(self):
        if self.initialised and not self.pause_schedule and bot_config.TODO_BOT_FOLDER != "":
            
            "check tasks"
            while True:
                
                task = scheduler_get_task_due_today(bot_config.TODO_BOT_FOLDER)
                if task:
                    
                    try:
                        #common.bot_logging.bot_logger.debug(task)
                        common.bot_logging.bot_logger.debug(task.subject)
                        send_to_user(f"Looks like one of my tasks is due - {task.subject}")
                        current_date_time = datetime.now() 
                        
                        inital_prompt = f'''With only the tools provided and with the current date and time of {current_date_time}, help the human with the following request, Request: {task.subject} - {task.body}
                        If a tool isnt available, use the FORWARD tool'''
                        response = self.agent_executor.run(input=inital_prompt, callbacks=[self.handler])
                        task.body = response
                        task.mark_completed()
                        #send_to_another_bot('journal',f"Add to journal that Office Bot Completed the task: {task.subject}")
                        task.save()
                    except Exception as e:
                        self.pause_schedule = True
                        tb = traceback.format_exc()
                        publish_error(e, tb)
                        # send_to_user( f"An exception occurred: {e}")
                else:
                    break

            while True:
                task = scheduler_get_bots_unscheduled_task(bot_config.TODO_BOT_FOLDER)
                if task:
                    
                    try:
                        common.bot_logging.bot_logger.debug(task)
                        common.bot_logging.bot_logger.debug(task.subject)
                        send_to_user(f"Looks like one of my tasks is due - {task.subject}")
                        
                        current_date_time = datetime.now() 

                        
                        inital_prompt = f'''With only the tools provided and with the current date and time of {current_date_time}, help the human with the following request, Request: {task.subject} - {task.body}
                        If a tool isnt available, use the FORWARD tool'''
                        response = self.agent_executor.run(input=inital_prompt, callbacks=[self.handler])
                        task.body = response
                        task.mark_completed()
                        send_to_another_bot('journal',f"Add to journal that Office Bot Completed the task: {task.subject}")
                        task.save()
                    except Exception as e:
                        self.pause_schedule = True
                        tb = traceback.format_exc()
                        publish_error(e, tb)
                        # send_to_user( f"An exception occurred: {e}")
                else:
                    break
            
            check_for_upcomming_event()

    def process_email_schedule(self):

        if self.initialised and not self.pause_schedule:
            "check emails"
            if str(bot_config.AUTO_READ_EMAILS).lower() in 'yes,true,enable':
                scheduler_check_emails()

    def heartbeat(self):
        from_bot_to_bot_manager('heartbeat', os.getpid())

    def load_tools(self, llm) -> list():
        
        # tools = load_tools(["human"], input_func=get_input, prompt_func=send_prompt, llm=llm)
        
        tools = []

        tools.append(MSGetTaskFolders())
        tools.append(MSGetTasks())
        tools.append(MSGetTaskDetail())
        tools.append(MSSetTaskComplete())
        tools.append(MSCreateTask())
        tools.append(MSDeleteTask())
        tools.append(MSCreateTaskFolder())
        
        tools.append(MSSearchEmailsId())
        tools.append(MSGetEmailDetail())
        tools.append(MSDraftEmail())
        tools.append(MSSendEmail())
        tools.append(MSReplyToEmail())
        tools.append(MSForwardEmail())
        tools.append(MSDraftReplyToEmail())
        tools.append(MSDraftForwardEmail())

        tools.append(MSGetCalendarEvents())
        tools.append(MSGetCalendarEvent())
        tools.append(MSCreateCalendarEvent())

        tools.append(Forward())

        return tools
    
    def get_credential(self, name):
        common.bot_logging.bot_logger.debug(f"Fetching credential for: {name}")
        for credential in self.credentials:
            if name in credential:
                common.bot_logging.bot_logger.debug(credential[name])
                return credential[name]
        return False