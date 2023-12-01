from langchain.prompts.prompt import PromptTemplate

NOTE_PROMPT = PromptTemplate(
    input_variables=[],
    template="You note that: ",
)

NOTE_LOGGER = PromptTemplate(
    input_variables=["note"],
    template="\nYou note that {note}",
)
