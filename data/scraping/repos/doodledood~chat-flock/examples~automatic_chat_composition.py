import typer
from dotenv import load_dotenv
from halo import Halo
from langchain.memory import ConversationSummaryBufferMemory

from chatflock.backing_stores import InMemoryChatDataBackingStore
from chatflock.backing_stores.langchain import LangChainMemoryBasedChatDataBackingStore
from chatflock.base import Chat, ChatDataBackingStore
from chatflock.code.langchain import CodeExecutionTool
from chatflock.code.local import LocalCodeExecutor
from chatflock.composition_generators.langchain import LangChainBasedAIChatCompositionGenerator
from chatflock.conductors import LangChainBasedAIChatConductor
from chatflock.participants.user import UserChatParticipant
from chatflock.renderers import TerminalChatRenderer
from examples.common import create_chat_model, get_max_context_size


def automatic_chat_composition(model: str = "gpt-4-1106-preview", temperature: float = 0.0) -> None:
    chat_model = create_chat_model(model=model, temperature=temperature)

    def create_default_backing_store() -> ChatDataBackingStore:
        max_context_size = get_max_context_size(chat_model)
        if max_context_size is not None:
            return LangChainMemoryBasedChatDataBackingStore(
                memory=ConversationSummaryBufferMemory(llm=chat_model, max_token_limit=max_context_size)
            )
        else:
            return InMemoryChatDataBackingStore()

    spinner = Halo(spinner="dots")
    user = UserChatParticipant(name="User")
    chat_conductor = LangChainBasedAIChatConductor(
        chat_model=chat_model,
        spinner=spinner,
        # Set up a proper goal so the composition generator can use it to generate the composition that will best fit
        goal="Come up with a plan for the user to invest their money. The goal is to maximize wealth over the "
        "long-term, while minimizing risk.",
        # Pass in a composition generator to the conductor
        composition_generator=LangChainBasedAIChatCompositionGenerator(
            fixed_team_members=[user],
            chat_model=chat_model,
            spinner=spinner,
            participant_available_tools=[CodeExecutionTool(executor=LocalCodeExecutor(), spinner=spinner)],
        ),
    )
    chat = Chat(backing_store=create_default_backing_store(), renderer=TerminalChatRenderer())

    # It's not necessary in practice to manually call `initialize_chat` since initiation is done automatically
    # when calling `initiate_dialog`. However, this is needed to eagerly generate the composition.
    # Default is lazy and will happen when the chat is initiated.
    chat_conductor.prepare_chat(chat=chat)
    print(f"\nGenerated composition:\n=================\n{chat.active_participants_str}\n=================\n\n")

    result = chat_conductor.initiate_dialog(chat=chat)
    print(result)


if __name__ == "__main__":
    load_dotenv()

    typer.run(automatic_chat_composition)
