import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.fastapi import SlackRequestHandler
from slack_bolt import App
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request

from langchain.callbacks.manager import tracing_v2_enabled

from LLM.LLaMaQuant import LLaMaQuant

from DB.Vector import QdrantVector

from DataLogger.DataLogger import GithubDailyDiscussion

class SlackClass :
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv(find_dotenv())
        # Set Slack API credentials
        self.SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
        self.SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
        self.SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
        # Initialize the Slack app
        self.app = App(token=self.SLACK_BOT_TOKEN)
        self.client = WebClient(token=self.SLACK_BOT_TOKEN)
        self.handler = SlackRequestHandler(self.app)
        # Intialize the LLaMaQuant model
        self.llm = LLaMaQuant()
        # Initialize the vector database
        self.vectorDB = QdrantVector()

        self.dailyDiscussion = GithubDailyDiscussion()
        
    

    def GetBotUserId(self):
        """
        Get the bot user ID using the Slack API.
        Returns:
            str: The bot user ID.
        """
        try:
            # Initialize the Slack client with your bot token
            slack_client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
            response = slack_client.auth_test()
            return response["user_id"]
        except SlackApiError as e:
            print(f"Error: {e}")


    def MyFunction(self,text):
        """
        Custom function to process the text and return a response.
        In this example, the function converts the input text to uppercase.

        Args:
            text (str): The input text to process.

        Returns:
            str: The processed text.
        """
        print("called agent")
        contextChoice = self.llm.agent(text)
        contextType = ""
        if contextChoice.lower().find("githublogs") != -1:
            contextType = "githublogs"
            context = self.vectorDB.getContext(text,1)
        elif contextChoice.lower().find("attendance") != -1:
            contextType = "attendance"
            context = self.vectorDB.getContext(text,2)
        elif contextChoice.lower().find("users") != -1:
            contextType = "users"
            context = self.vectorDB.getContext(text,0)
        elif contextChoice.lower().find("dailyreport") != -1:
            contextType = "dailyreport"
            day = contextChoice.split(",")[-1]
            context = self.dailyDiscussion.getContext(day)
        else:
            context = ""
        print(contextChoice)
        print(context)
        response = self.llm.respond(text,context,contextType)
        return response
