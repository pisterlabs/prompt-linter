from langchain.prompts.prompt import PromptTemplate

_PROMPT_TEMPLATE = """You are given an entity, here is the description: {selected}.

You have to embed this entity into the text where it makes sense. Here is the text: {text_to_personalize}.

"""


PROMPT = PromptTemplate(
    input_variables=["selected", "text_to_personalize"],
    template=_PROMPT_TEMPLATE,
)
