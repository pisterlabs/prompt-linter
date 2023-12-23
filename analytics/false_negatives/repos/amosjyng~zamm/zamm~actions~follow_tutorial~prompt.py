from langchain.prompts.prompt import PromptTemplate

FOLLOW_TUTORIAL_PROMPT = PromptTemplate(
    input_variables=[],
    template="""
You decide to follow instructions in another file. You must enter in the path to the file (do not pick a different file -- and do not use the link name, but the actual path to the link), and the task you expect to accomplish with this file. For example, if you were to encounter the following step in the training manual:

> Follow the instructions at [`goober.md`](./path/to/goober.md) to floof the goober

then you should enter something like:

Path to the instructions file: `./path/to/goober.md`
Task: Floof the goober

Do it now for the current task.

""".lstrip(),  # noqa
)

FOLLOW_TUTORIAL_LOGGER = PromptTemplate(
    input_variables=["documentation_path", "task"],
    template="\nYou followed the instructions at `{documentation_path}` to '{task}'.",
)
