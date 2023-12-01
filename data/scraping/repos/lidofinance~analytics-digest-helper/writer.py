from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from .prompts import DIGEST_SYSTEM_PROMPT
from datetime import datetime


def write_thread(formatted_dune_info: dict[str, str], start_date, end_date) -> str:
    chat = ChatOpenAI(temperature=0, model="gpt-4")  # type: ignore
    dune_info = ""
    for k, v in formatted_dune_info.items():
        dune_info += f"{k}: {v}\n"

    current_date = datetime.today().strftime("%B %d")

    dune_info += f"\n\nWrite a thread with the above information. The current date is {current_date}. This digest contains information for the week of {start_date} to {end_date}. Use previous examples to guide the format and styling of your writing. You do not necessarily need to the same sectioned content as in the examples. Your client specifically wants to know about more important changes in the data."

    digest_user_prompt = HumanMessage(content=dune_info)
    today = datetime.today().strftime("%B %d %Y")
    thread = chat.predict_messages(
        [
            SystemMessage(content=DIGEST_SYSTEM_PROMPT.format(DATE=today)),
            digest_user_prompt,
        ]
    )

    return thread.content
