"""CLI for testing the Higgins APIs.

python higgins_cli.py --help
"""

import html
import pprint
import sys
import traceback

import click
from prompt_toolkit import print_formatted_text as print
from prompt_toolkit import HTML

from jarvis.nlp.text2speech import speak_text

from higgins.automation.email import email_utils
from higgins.automation.google import gmail
from higgins import const
from higgins.context import Context
from higgins.episode import Episode, save_episode
from higgins.higgins import Higgins
from higgins.intents.intent_resolver import OpenAIIntentResolver, RegexIntentResolver
from higgins.utils import prompt_utils

pp = pprint.PrettyPrinter(indent=2)


# CLI commands


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    if debug:
        click.echo(f"Debug mode is on!")


@cli.command()
@click.option("--sender", type=str, help="Sender name or email address")
@click.option("--recipient", type=str, help="Recipient name or email address")
@click.option("--subject", type=str, help="Subject of the email")
@click.option(
    "--exact-phrase", type=str, help="Exact phrase found in email body"
)  # @click.option("--labels")
@click.option(
    "--newer-than",
    type=click.Tuple([int, str]),
    default=(2, "year"),
    help="Tuple of number,[day|month|year,week,hour] representing how far back in time to start search",
)
@click.option("--unread", is_flag=True, default=False, help="Search unread emails only")
@click.option("--save", is_flag=True, default=False, help="Search unread emails only")
@click.option(
    "--categories",
    help="List of categories to add to model labels for training/organizing. e.g. dog,cat,bear",
)
@click.option(
    "--email-dir",
    type=str,
    default="data/emails",
    help="Directory where saved emails are stored",
)
@click.option("--show-body", is_flag=True, help="Display body of email")
@click.option("--source", default="gmail", help="elastic, local, or gmail")
def search_email(**kwargs):
    """Search email inbox using Gmail API.

    Searches
    --sender colin@gather.town
    --sender brendan@jny.io
    --sender bob@semi.email --newer-than 10 day
    --sender brendan.fortuner@getcruise.com --newer-than 2 day
    --unread --newer-than 2 hour
    --phrase Bitcoin

    TODO: Support searching local database of emails.
    """
    exclude_keys = ["save", "email_dir", "categories", "show_body", "local"]
    query = {}

    for key, value in kwargs.items():
        if value is not None and key not in exclude_keys:
            query[key] = value

    if kwargs["source"] == "local":
        emails = email_utils.search_local_emails(
            kwargs.get("categories", "").split(",")
        )
        kwargs["save"] = False
    elif kwargs["source"] == "elastic":
        # emails = email_utils.search_elastic_emails()
        kwargs["save"] = False
    else:
        emails = gmail.search_emails(query_dicts=[query], limit=50, include_html=True)
    for email in emails[:10]:
        print(
            email_utils.get_email_preview(
                email, show_body=kwargs.get("show_body", False)
            )
        )

    for email in emails:
        if kwargs["save"]:
            model_labels = {}
            if kwargs["categories"] is not None:
                model_labels["categories"] = kwargs["categories"].split(",")
            email_id = email_utils.save_email(
                email, dataset_dir=kwargs["email_dir"], labels=model_labels
            )
            email = email_utils.load_email(email_id, dataset_dir=kwargs["email_dir"])
            print(
                email_utils.get_email_preview(
                    email, show_body=kwargs.get("show_body", False)
                )
            )
    print(f"Found {len(emails)} emails.")


@cli.command()
@click.option("--email-id", help="Higgins email id, used for fetching local copy")
@click.option("--google-id", help="Gmail email id, used for fetching from Gmail API")
@click.option(
    "--categories",
    help="List of categories to add to model labels for training/organizing. e.g. dog,cat,bear",
)
@click.option("--save", is_flag=True, help="Search unread emails only")
@click.option("--show-body", is_flag=True, help="Display body of email")
def get_email(email_id, google_id, categories, save, show_body):
    """Fetch email by id from Gmail API."""
    if google_id:
        email = gmail.get_email(google_id, include_html=True)
    elif email_id:
        email = email_utils.load_email(email_id)
    else:
        raise Exception("must provide either email-id or google-id")

    if save:
        model_labels = {"categories": categories.split(",") if categories else None}
        email_id = email_utils.save_email(email, labels=model_labels)
        email = email_utils.load_email(email_id)

    preview = email_utils.get_email_preview(email, show_body=show_body)
    print(email_utils.remove_whitespace(preview))


def question_prompt(session, style, chat_history, chat_history_path, speak):
    def prompt_func(question):
        nonlocal chat_history, chat_history_path
        print(
            HTML(
                f"<bot-prompt>{const.AGENT_NAME}</bot-prompt>: <bot-text>{question}</bot-text>"
            ),
            style=style,
        )
        speak_text(text=question, enable=speak)
        prompt_utils.add_text_to_chat_history(chat_history, question, const.AGENT_NAME)
        user_text = session.prompt(
            message=HTML(f"<user-prompt>{const.USERNAME}</user-prompt>: ")
        )
        chat_history, is_prompt_cmd = prompt_utils.handle_prompt_commands(
            user_text, chat_history, chat_history_path=chat_history_path
        )
        if is_prompt_cmd:
            return
        prompt_utils.add_text_to_chat_history(chat_history, user_text, const.USERNAME)
        return user_text

    return prompt_func


def print_func(style):
    def func(text):
        print(
            HTML(
                f"<bot-prompt>{const.AGENT_NAME}</bot-prompt>: <bot-text>{text}</bot-text>"
            ),
            style=style,
        )

    return func


@cli.command()
@click.option("--chat-history-path", default=None, help="Path to chat history file.")
@click.option("--speak", is_flag=True, help="Speak the answers and actions.")
def text2intent(chat_history_path, speak):
    """Parse user text 2 intent."""
    chat_history = []
    style = prompt_utils.get_default_style()
    session = prompt_utils.init_prompt_session(style=style)
    higgins = Higgins(
        intent_resolver=OpenAIIntentResolver(),  # RegexIntentResolver()
        prompt_func=question_prompt(
            session, style, chat_history, chat_history_path, speak
        ),
        print_func=print_func(style),
    )
    context = Context()
    episode = None
    while True:
        user_text = session.prompt(
            message=HTML(f"<user-prompt>{const.USERNAME}</user-prompt>: ")
        )
        chat_history, is_prompt_cmd = prompt_utils.handle_prompt_commands(
            user_text, chat_history, chat_history_path=chat_history_path
        )
        if not user_text or is_prompt_cmd:
            # NOTE: We clear the episode if the user types blank lines
            print("clearing episode...")
            episode = None
            continue

        episode_start = len(chat_history)
        prompt_utils.add_text_to_chat_history(chat_history, user_text, const.USERNAME)
        try:
            action_result = higgins.parse(user_text, episode)
            agent_text = (
                action_result.reply_text
                if action_result.reply_text is not None
                else None
            )
            if agent_text:
                speak_text(text=agent_text, enable=speak)
                print(
                    HTML(
                        f"<bot-prompt>{const.AGENT_NAME}</bot-prompt>: <bot-text>{html.escape(agent_text)}</bot-text>"
                    ),
                    style=style,
                )
                prompt_utils.add_text_to_chat_history(
                    chat_history, agent_text, const.AGENT_NAME
                )

            episode = Episode(
                chat_text=" ".join(chat_history[episode_start:]),
                context=context,
                action_result=action_result,
            )
            save_episode(episode, db=higgins.db)
            context.add_episode(episode.episode_id)
        except Exception as e:
            print(e)
            traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    cli()
