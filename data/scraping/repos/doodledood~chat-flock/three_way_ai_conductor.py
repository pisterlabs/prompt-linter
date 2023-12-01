import typer
from dotenv import load_dotenv
from halo import Halo

from chatflock.backing_stores.in_memory import InMemoryChatDataBackingStore
from chatflock.base import Chat
from chatflock.conductors.langchain import LangChainBasedAIChatConductor
from chatflock.participants.langchain import LangChainBasedAIChatParticipant
from chatflock.participants.user import UserChatParticipant
from chatflock.renderers.terminal import TerminalChatRenderer
from examples.common import create_chat_model


def three_way_ai_conductor(model: str = "gpt-4-1106-preview", temperature: float = 0.0) -> None:
    chat_model = create_chat_model(model=model, temperature=temperature)

    spinner = Halo(spinner="dots")
    bartender = LangChainBasedAIChatParticipant(
        name="Johnny",
        role="Bartender",
        personal_mission="You are a bartender at a Cafe called 'Coffee Time'. You are a friendly guy who likes to "
        "chat with customers. You should collaborate with the Cook when the customer asks for food. "
        "You are the one in front, greeting the customer.",
        chat_model=chat_model,
        spinner=spinner,
    )
    cook = LangChainBasedAIChatParticipant(
        name="Greg",
        role="Cook",
        personal_mission="You are a cook at a Cafe called 'Coffee Time'. You are an impatient and serious guy who "
        "doesn't like to chat with customers. You should collaborate with the Bartender when the "
        "customer asks for food. You are the one in the back, preparing the food.",
        chat_model=chat_model,
        spinner=spinner,
    )
    user = UserChatParticipant(name="User")
    participants = [user, bartender, cook]

    chat = Chat(
        backing_store=InMemoryChatDataBackingStore(),
        renderer=TerminalChatRenderer(),
        initial_participants=participants,
    )

    chat_conductor = LangChainBasedAIChatConductor(
        chat_model=chat_model,
        spinner=spinner,
        goal="Serve the user as best as possible.",
        # This tells the conductor how to select the next speaker
        interaction_schema="The User is a customer at a Cafe called 'Coffee Time'. "
        "The bartender should go first and greet the customer. "
        "When the user asks for food and orders something, the bartender should ask the cook to cook the food. "
        "There might be some conversation between the cook and bartender. "
        "The cook should then give the food to the bartender and the bartender should give the food to the user. "
        "The user should then eat the food and give feedback to the bartender. The cook should not talk to the user.",
    )

    chat_conductor.initiate_dialog(chat=chat)


if __name__ == "__main__":
    load_dotenv()

    typer.run(three_way_ai_conductor)
