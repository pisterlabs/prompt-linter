from dotenv import load_dotenv

load_dotenv()

from langchain.chains import SequentialChain, LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# This is an LLMChain to write a synopsis given a title of a play and the era it is set in.
model = ChatOpenAI()
idea_template = (
    """ You are a poet. Generate a single topic idea for a poem written by {person}"""
)

poem_template = """Write a {style} poem about {topic}."""

feedback_template = """You are a poetry teacher teaching {age_level}. \ 
                    Given the following poem, provide concise feedback to the student.
                    Poem: {poem}"""

idea_prompt_template = ChatPromptTemplate.from_template(idea_template)
poem_prompt_template = ChatPromptTemplate.from_template(poem_template)
feedback_prompt_template = ChatPromptTemplate.from_template(feedback_template)

idea_chain, poem_chain, feedback_chain = (
    LLMChain(
        llm=model,
        prompt=idea_prompt_template,
        output_key="topic",
    ),
    LLMChain(
        llm=model,
        prompt=poem_prompt_template,
        output_key="poem",
    ),
    LLMChain(
        llm=model,
        prompt=feedback_prompt_template,
        output_key="feedback",
    ),
)

overall_chain = SequentialChain(
    chains=[idea_chain, poem_chain, feedback_chain],
    input_variables=["person", "style", "age_level"],
    # Here we return multiple variables
    output_variables=["poem", "feedback"],
)

# .run is not supported for SequentialChain since there is more than one output key, .generate not supportede
result = overall_chain(
    {"person": "a pirate", "style": "haiku", "age_level": "9th graders"}
)
print(result)

# .run is not supported for SequentialChain since there is more than one output key, .generate not supportede
result = overall_chain.apply(
    [
        {"person": "a pirate", "style": "haiku", "age_level": "9th graders"},
        {"person": "a carpenter", "style": "sonnet", "age_level": "7th graders"},
    ]
)
print(result)
