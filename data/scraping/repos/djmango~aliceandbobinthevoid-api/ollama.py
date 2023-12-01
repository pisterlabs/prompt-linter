from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOllama
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

chat_model = ChatOllama(
    #    model="mistral-openorca",
    model="mistral",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)

alice_system_message = SystemMessage(content="You are Alice, a capable adventurer.")
bob_system_message = SystemMessage(content="You are Bob, a capable adventurer.")

alice_messages: list[BaseMessage] = [
    alice_system_message,
    AIMessage(content="Hello, Bob, I am Alice. How are you?"),
    HumanMessage(
        content="I am Bob, you are Alice. We are stuck in the void. What should we do, Alice?"
    ),
]

bob_messages: list[BaseMessage] = [
    bob_system_message,
    HumanMessage(content="Hello, Bob, I am Alice. How are you?"),
    AIMessage(
        content="I am Bob, you are Alice. We are stuck in the void. What should we do, Alice?"
    ),
]

while True:
    print("\n----ALICE----")
    alice_response: BaseMessage | AIMessage = chat_model(alice_messages)
    alice_messages.append(alice_response)
    bob_messages.append(HumanMessage(content=alice_response.content))

    print("\n----BOB----")
    bob_response: BaseMessage | AIMessage = chat_model(bob_messages)
    bob_messages.append(bob_response)
    alice_messages.append(HumanMessage(content=bob_response.content))
