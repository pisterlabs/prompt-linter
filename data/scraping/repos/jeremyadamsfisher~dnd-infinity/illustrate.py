from functools import lru_cache

from langchain import LLMChain, OpenAI, PromptTemplate

prompt = """\
This is what my dungeon master just said.

> "{description}"

Imagine someone were to make a painting of this scene. What would be the \
caption for this painting?"""


@lru_cache
def generate_illustration_caption(dialog_turn):
    chat_prompt = PromptTemplate(
        input_variables=["description"],
        template=prompt,
    )
    chain = LLMChain(
        llm=OpenAI(temperature=1.0),
        prompt=chat_prompt,
        verbose=True,
    )
    return (
        chain.predict(description=dialog_turn)
        .strip()
        .replace('"', "")
        .replace(".", "")
        .title()
    )
