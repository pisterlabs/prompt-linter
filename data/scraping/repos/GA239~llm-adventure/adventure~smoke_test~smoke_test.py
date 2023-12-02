import time

from dotenv import load_dotenv, find_dotenv
from langchain import ConversationChain, PromptTemplate

from adventure.utils import get_model
from langchain import PromptTemplate

TEMPLATE = """
The following is a friendly conversation between a human and an AI. 
The AI provides very short and precise answers from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
You: {input}
AI:"""


def create_conversation_chain(model_name):
    prompt = PromptTemplate(input_variables=["history", "input"], template=TEMPLATE)
    llm = get_model(model_name)
    return ConversationChain(llm=llm, prompt=prompt, verbose=True)


def test_my_name(model_name):
    conversation = create_conversation_chain(model_name)
    print(conversation.run("Hi!, my name is Andrei."))
    print(conversation.run("What's my name? say my name in one word."))


def test_character_name(model_name):
    conversation = create_conversation_chain(model_name)
    print(conversation.run("What the real name of Spider man? provide the precise answer."))
    print(conversation.run("What the real name of Batman? Answer in two words."))


def test_calculate(model_name=""):
    conversation = create_conversation_chain(model_name)
    prompt = "Calculate ```{}``` and tell me the result as a number. Be precise in calculation!"
    print(conversation.run(prompt.format("2 + 2 * 3")))
    print(conversation.run(prompt.format("(2 + 2) * 3")))


def test_cohere():
    test_my_name("Cohere")
    time.sleep(60)  # API limitation: 5 calls per minute
    test_character_name("Cohere")
    time.sleep(60)  # API limitation: 5 calls per minute
    test_calculate("Cohere")


def test_huggingface_google_flan():
    test_my_name("HuggingFace_google_flan")
    test_character_name("HuggingFace_google_flan")
    test_calculate("HuggingFace_google_flan")


def test_huggingface_mbzai_lamini_flan():
    test_my_name("HuggingFace_mbzai_lamini_flan")
    test_character_name("HuggingFace_mbzai_lamini_flan")
    test_calculate("HuggingFace_mbzai_lamini_flan")


def test_local_gp2_model():
    test_my_name("Local_gpt2")
    test_character_name("Local_gpt2")
    test_calculate("Local_gpt2")


def test_local_lama_model():
    test_my_name("Local_lama")
    test_character_name("Local_lama")
    test_calculate("Local_lama")


if __name__ == "__main__":
    load_dotenv(find_dotenv(raise_error_if_not_found=True))
    # test_local_lama_model()
    # test_huggingface_mbzai_lamini_flan()
    test_huggingface_google_flan()
    print("Smoke test passed!")
