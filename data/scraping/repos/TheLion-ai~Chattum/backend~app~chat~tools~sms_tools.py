"""Tool for sending sms to a phone number using twillio.""" ""


from typing import Optional

from langchain.utilities.twilio import TwilioAPIWrapper
from pybars import Compiler
from pydantic import BaseModel, create_model

from .base_tool import ToolTemplate, UserVariable


class TwilloTool(ToolTemplate):
    """Tool for sending sms to a phone number using twillio."""

    name: str = "Send sms tool"
    user_description: str = "sent sms to a phone number."

    user_variables: list[UserVariable] = [
        UserVariable(name="account_sid", description="Account SID", form_type="text"),
        UserVariable(name="auth_token", description="Auth Token", form_type="text"),
        UserVariable(name="from_number", description="From number", form_type="text"),
    ]

    @property
    def args_schema(self) -> BaseModel:
        """Return the args schema for langchain."""

        class ArgsSchema(BaseModel):
            message: str
            to_number: str

        return ArgsSchema

    def __init__(
        self, user_variables: list[dict] = [], bot_description: Optional["str"] = None
    ) -> None:
        """Initialize the tool using the user variables with TwilioAPIWrapper."""
        super().__init__(user_variables)
        self.twilio = TwilioAPIWrapper(
            account_sid=self.variables_dict["account_sid"],
            auth_token=self.variables_dict["auth_token"],
            from_number=self.variables_dict["from_number"],
        )

    def run(self, **kwargs: dict) -> str:
        """Run the tool."""
        self.twilio.run(kwargs["message"], kwargs["to_number"])
        return "message sent"

    @property
    def description(
        self,
    ) -> str:
        """Return the tool description for llm."""
        return "use this tool to sent sms to a phone number."
