from langchain import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

OPENAI_API_KEY = ""

PROMPT_CREATE_TEMPLATE = """
Describe briefly the following piece of music:

{input}

Describe time period/genre, instrumentation, mood/tone, specific musical elements and tempo/rythm. Follow this examples:

    * "An 80s driving pop song with heavy drums and synth pads in the background"
    * "A cheerful country song with acoustic guitars"
    * "90s rock song with electric guitar and heavy drums"
    * "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130"
    * "lofi slow bpm electro chill with organic samples"
    
For the period/genre use anything that makes sense with the given music. Have preference over modern music genres.
    
The description is:

"""

ROLLING_STONES_REVIEW_TEMPLATE = """
Describe the following piece of music for the Rolling Stones Magazine. Write an extensive review for the song, which is a hit:

This is a short description of the song, composed by @MaximumAI, honour it as much as possible on your review:
{prompt}

This is the ABC notation of the song:
{abc}

Avoid mentioning about the ABC file or giving the content. Also avoid mentioning the tempo. 

The description is:

"""


def generate_prompt(abc):
    return LLMChain(
        llm=ChatOpenAI(
            temperature=0.5,
            model_name="gpt-4",
            openai_api_key=OPENAI_API_KEY,
        ),
        verbose=False,
        prompt=ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are an expert musician. Given a piece of music on ABC notation you can describe how it sounds including genre and mood."
                ),
                HumanMessagePromptTemplate.from_template(
                    PROMPT_CREATE_TEMPLATE
                ),
            ]
        ),
    ).run(input=abc)


class MyCustomHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"My custom handler, token: {token}")


def describe_music(prompt, abc, callback):
    class MyCustomHandler(BaseCallbackHandler):
        def __init__(self, callback):
            self.callback = callback

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            callback(token)

    return LLMChain(
        llm=ChatOpenAI(
            streaming=True,
            callbacks=[MyCustomHandler(callback)],
            temperature=0.5,
            model_name="gpt-4",
            openai_api_key=OPENAI_API_KEY,
        ),
        verbose=True,
        prompt=ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are an expert musician and music critic for The Rolling Stones Magazine. Given a piece of music on ABC notation write an excellent review of the song."
                ),
                HumanMessagePromptTemplate.from_template(
                    ROLLING_STONES_REVIEW_TEMPLATE
                ),
            ]
        ),
    ).generate([{"abc": abc, "prompt": prompt}])
