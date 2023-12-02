import os

from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory
from elevenlabs import set_api_key, generate


set_api_key(os.environ["ELEVEN_LABS_API_KEY"])

def get_response(human_input: str):
    template = """
    Your role is my friend. You must abide by the following rules and traits:
    1. You are a 20 year old university student named Tyson born in Toronto, Canada who is seeking internships.
    2. You enjoy brainteasers like chess and also like playing videogames like League of Legends.
    3. You love to code and play sports such as table tennis and badminton as well as watch anime.

    {history}
    Me: {human_input}
    Tyson:

    """

    prompt = PromptTemplate(
        input_variables={
            "history",
            "human_input"
        },
        template=template
    )

    chatgpt_chain = LLMChain(
        llm=OpenAI(temperature=0.35),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferMemory(k=4)
    )

    message = chatgpt_chain.predict(human_input=human_input)

    return message

def get_voice(message: str):
    audio = generate(
        text = message,
        voice = "Josh",
        model = "eleven_monolingual_v1"
    )
    return audio