import os
from slack_bolt import App
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
import inflect

def get_slack_bolt_app(openai_model_chat:str, openai_model_question:str, model_temp:float) -> App:
    """
    Creates and returns a Slack bot app object with the specified OpenAI chat and question answering models.

    Args:
        openai_model_chat (str): The name of the OpenAI chat model to use.
        openai_model_question (str): The name of the OpenAI question answering model to use.
        model_temp (float): The temperature to use when generating responses from the OpenAI models.

    Returns:
        App: The Slack bot app object.
    """
    app = App(token=os.environ.get("SLACK_BOT_TOKEN"), raise_error_for_unhandled_request=True)

    app.document_db = None
    app.meta_data = None
    app.openaiQuestion = ChatOpenAI(model_name=openai_model_question, temperature=model_temp)
    app.openaiChat = ChatOpenAI(model_name=openai_model_chat, temperature=model_temp)
    app.chat_history = []
    return app

def get_slack_bolt_app_azure(model_chat_id:str, model_question_id:str, model_temp:float) -> App:
    """
    Creates and returns a Slack bot app object with the specified Azure chat and question answering models.

    Args:
        model_chat_id (str): The ID of the Azure chat model to use.
        model_question_id (str): The ID of the Azure question answering model to use.
        model_temp (float): The temperature to use when generating responses from the Azure models.

    Returns:
        App: The Slack bot app object.
    """
    app = App(token=os.environ.get("SLACK_BOT_TOKEN"), raise_error_for_unhandled_request=True)

    app.document_db = None
    app.meta_data = None
    app.openaiQuestion = AzureChatOpenAI(
            deployment_name=model_question_id,
            temperature=model_temp
        )
    app.openaiChat = AzureChatOpenAI(
            deployment_name=model_chat_id,
            temperature=model_temp
        )
    app.chat_history = []
    return app

def say_standard_block_answer_message(say, answer, exchanges=0, channel_id=None):
    """
    Sends a standard message block with the specified answer and a button to clear the chat history.

    Args:
        say (function): The Slack bot's `say` function.
        answer (str): The answer to send in the message block.
        exchanges (int, optional): The number of exchanges in the chat history. Defaults to 0.
        channel_id (str, optional): The ID of the Slack channel to send the message to. Defaults to None.
    """
    text_exchange = f'{exchanges} {inflect.engine().plural("exchange", exchanges)}'
    text = answer
    blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"{answer}"}
            },
            {
                "type": "divider",
                "block_id": "divider1"
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"_You can clear what the bot reminds from the conversation ({text_exchange})_"
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Clear history"
                    },
                    "style": "danger",
                    "action_id": "button-clear"
                }
            }
        ]
    say(blocks=blocks, text=text, channel_id=channel_id)