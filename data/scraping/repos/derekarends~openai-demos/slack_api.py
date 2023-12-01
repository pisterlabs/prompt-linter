""" Util that calls Slack. """
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from slack_prompts import (
    SLACK_WRITE_MESSAGE_TO_CHANNEL,
    SLACK_READ_CHANNELS,
)
from langchain.utils import get_from_dict_or_env


class SlackApiWrapper(BaseModel):
    """ Wrapper for Slack API. """

    slack: WebClient  #: :meta private:

    # List of operations that this tool can perform
    operations: List[Dict] = [
        {
            "mode": "channels_read",
            "name": "Read the channels",
            "description": SLACK_READ_CHANNELS,
        },
        {
            "mode": "chat_write",
            "name": "Write a chat message to the channel",
            "description": SLACK_WRITE_MESSAGE_TO_CHANNEL,
        }
    ]

    class Config:
        """ Configuration for this pydantic object. """
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def list(self) -> List[Dict]:
        return self.operations

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """ Validate that api key and python package exists in environment. """
        bot_token = get_from_dict_or_env(values, "bot_token", "SLACK_BOT_TOKEN")

        slack = WebClient(token=bot_token)
        values["slack"] = slack

        return values
    
    def run(self, mode: str, text: Optional[str]) -> str:
        """ Based on the mode from the caller, run the appropriate function. """
        if mode == "channels_read":
            return self.channels_read()
        elif mode == "chat_write":
            return self.chat_write(text)
        else:
            raise ValueError(f"Got unexpected mode {mode}")

    def channels_read(self) -> str:
        """ Read the channels from the slack workspace """
        try:
            import json
            # Call the conversations.list method using the slack WebClient
            response = self.slack.conversations_list()
            channels = []

            for channel in response["channels"]:
                channels.append({"name": channel["name"], "id": channel["id"]})

            json_channels = json.dumps(channels)
            return json_channels

        except ImportError:
            raise ImportError(
                "json is not installed. " "Please install it with `pip install json`"
            )
        except SlackApiError as e:
            print("Error: {}".format(e))
            raise Exception("Failed to write chat message")

    def chat_write(self, query: str) -> str:
        """ 
        Take a json object of {"channel": "channel_name", "message": "message"}
        and write it to the channel
        """
        try:
            import json
            params = json.loads(query)
            fields = dict(params)
            # Call the chat.postMessage method using the slack WebClient
            self.slack.chat_postMessage(
                channel=fields["channel"],
                text=fields["message"]
            )
            return "Successfully wrote chat message"
        except ImportError:
            raise ImportError(
                "json is not installed. " "Please install it with `pip install json`"
            )
        except SlackApiError as e:
            print("Error: {}".format(e))
            raise Exception("Failed to write chat message")
        
