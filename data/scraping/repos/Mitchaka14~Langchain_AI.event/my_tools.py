# my_tools.py

from typing import Optional, Type
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from pydantic import BaseModel, Extra
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
import streamlit as st
from dotenv import load_dotenv
import openai

try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    os.environ["serpapi_api_key"] = st.secrets["SERPAPI_API_KEY"]
except Exception:
    load_dotenv()
    os.environ["serpapi_api_key"] = os.getenv("SERPAPI_API_KEY")


class DataInput(BaseModel):
    question: str


class DataTool(BaseTool):
    name = "custom_dataTool"
    description = """ 
    This tool provides information about the bussiness. 
    it has the bussiness name and other bussiness related stuff,if its not here then Search
    """
    # args_schema: Type[BaseModel] = DataInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        from .data_function import (
            data_function,
        )  # Import the function here to avoid circular imports

        output = data_function(query)
        return output

    async def _arun(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("/data does not support async")


class SQLAgentInput(BaseModel):
    query: str


class SQLAgentTool(BaseTool):
    name = "ClinicDBTool"
    description = """
    This tool interacts with a Clinic SQL database, facilitating operations related to patients, appointments, employees, and related medical records,availability.
    With this tool, you can:
        - Create, update, and retrieve details about patients, employees, and appointments.
        - Create and fetch related data such as prescriptions, test results, billing information, complaints, referrals, and interactions.
    If the input is a patient or employee name, it retrieves all related information. This tool also enables creating new entries in the database when needed, a new or existing patient name is needed .
    Don't give patients information they shoould and dont need to know
    """
    args_schema: Type[BaseModel] = SQLAgentInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            from sql_agent_function import sql_agent_function
        except ImportError:
            from .sql_agent_function import sql_agent_function
        # Import the function here to avoid circular imports

        output = sql_agent_function(query)
        return output

    async def _arun(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("/SQLAgentTool does not support async")


class EmailToolInput(BaseModel):
    query: str


class EmailTool(BaseTool):
    name = "EmailTool"
    description = """
    This tool sends an email message extracted from the provided content string to the detected email address.
    The content string should contain an email address and the message body.
    """
    args_schema: Type[BaseModel] = EmailToolInput
    llm = OpenAI(temperature=0.9)

    def _run(
        self,
        query: EmailToolInput,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        try:
            password = st.secrets["CodedP"]
        except Exception:
            load_dotenv()
            password = os.getenv("CodedP")

        return self.send_email(query, password)

    async def _arun(
        self,
        query: EmailToolInput,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        return self._run(query, run_manager)

    def send_email(self, content: str, password: str) -> str:
        # Extract email address from the content string
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        match = re.search(email_pattern, content)

        if not match:
            return "No email address detected in the content"

        user_email = match.group()
        raw_message = content.replace(
            user_email, ""
        )  # Remove email from the raw message

        # Use GPT-3.5 Turbo to process the raw message
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Extract/add/make subject and content from the following message,return it in this formart..>subject: (.*?)<  >content: (.*?)<.",
                },
                {"role": "user", "content": raw_message},
            ],
        )

        formatted_query = completion.choices[0].message

        # default values
        subject = "email from voiceverse agent"
        message = raw_message

        # Check if GPT-3.5 Turbo formatted it correctly, otherwise, fallback to the original method
        if ">subject:" in formatted_query and ">content:" in formatted_query:
            subject = re.search(r">subject: (.*?)<", formatted_query).group(1).strip()
            message = re.search(r">content: (.*?)<", formatted_query).group(1).strip()
        else:
            # Extract subject and content if exists in raw_message
            raw_message_lines = raw_message.split(",")
            for line in raw_message_lines:
                line = line.strip()
                if line.lower().startswith("subject:"):
                    subject = line[8:].strip()  # skip "subject:" part
                elif line.lower().startswith("content:"):
                    message = line[8:].strip()  # skip "content:" part

        # create message object instance
        msg = MIMEMultipart()

        # setup the parameters of the message
        msg["From"] = "voiceverseverse@gmail.com"
        msg["To"] = user_email
        msg["Subject"] = subject

        # add in the message body
        msg.attach(MIMEText(message, "plain"))

        # create server
        server = smtplib.SMTP("smtp.gmail.com", 587)

        # starting the server instance
        server.starttls()

        # Login Credentials for sending the mail
        server.login(msg["From"], password)

        # send the message via the server
        server.sendmail(msg["From"], msg["To"], msg.as_string())
        server.quit()

        return f"Email successfully sent to {user_email}"


# class InteractiveTool(BaseTool):
#     class Config:
#         extra = Extra.allow

#     name = "InteractiveTool"
#     description = """
#     This tool interacts with the user, allowing the system to ask for additional information.
#     Tool selection after...call custom_dataTool if it is Business related,
#     call Appointments/scheduler if its Appointments/scheduler related, etc.
#     dont end conversation unless, the customer is satisfied
#     """

#     def __init__(self, event_handler):
#         super().__init__()
#         self.event_handler = event_handler

#     def _run(
#         self,
#         query: Optional[str] = None,
#         run_manager: Optional[CallbackManagerForToolRun] = None,
#     ) -> str:
#         """Use the tool."""
#         return self.event_handler("input_required", query if query else "Enter input: ")

#     async def _arun(
#         self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool asynchronously."""
#         # The input function is not compatible with asynchronous programming.
#         raise NotImplementedError("InteractiveTool does not support async")


# class FeedbackTool(BaseTool):
#     name = "FeedbackTool"
#     description = """
#     This tool should be called last,or  InteractiveTool can be called after if you ask a question
#     use this tool to determine if you should end the chain or you should ask the customer if the have more question.

#     """

#     def _run(
#         self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool."""
#         return f"""
#                     agent input was: {query}. Ask customer if they satisfied?(if they are end the chain), Do you still need help?(if they do provide the help)
#                     InteractiveTool use that tool next if you are asking the customer a question, or end conversation if you are saying goodbye
#                     """

#     async def _arun(
#         self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
#     ) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError("FeedbackTool does not support async")
