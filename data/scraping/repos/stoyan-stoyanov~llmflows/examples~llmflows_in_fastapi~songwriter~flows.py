# pylint: skip-file
import os
from llmflows.flows import AsyncFlow, AsyncFlowStep
from llmflows.llms import OpenAI
from llmflows.prompts import PromptTemplate

openai_api_key = os.environ.get("OPENAI_API_KEY", "<your-api-key>")


def create_flow():
    flowstep1 = AsyncFlowStep(
        name="Flowstep 1",
        llm=OpenAI(api_key=openai_api_key),
        prompt_template=PromptTemplate("What is a good title of a movie about {topic}?"),
        output_key="movie_title",
    )

    flowstep2 = AsyncFlowStep(
        name="Flowstep 2",
        llm=OpenAI(api_key=openai_api_key),
        prompt_template=PromptTemplate(
            "What is a good song title of a soundtrack for a movie called {movie_title}?"
        ),
        output_key="song_title",
    )

    flowstep3 = AsyncFlowStep(
        name="Flowstep 3",
        llm=OpenAI(api_key=openai_api_key),
        prompt_template=PromptTemplate(
            "What are two main characters for a movie called {movie_title}?"
        ),
        output_key="main_characters",
    )

    flowstep4 = AsyncFlowStep(
        name="Flowstep 4",
        llm=OpenAI(api_key=openai_api_key),
        prompt_template=PromptTemplate(
            "Write lyrics of a movie song called {song_title}. The main characters are"
            " {main_characters}"
        ),
        output_key="song_lyrics",
    )

    # Connect flowsteps
    flowstep1.connect(flowstep2, flowstep3, flowstep4)
    flowstep2.connect(flowstep4)
    flowstep3.connect(flowstep4)

    soundtrack_flow = AsyncFlow(flowstep1)

    return soundtrack_flow
