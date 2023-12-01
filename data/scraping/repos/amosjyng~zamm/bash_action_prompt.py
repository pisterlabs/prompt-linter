from langchain.prompts.prompt import PromptTemplate

from zamm.actions.use_terminal.prompt import TerminalPromptTemplate

MANAGER_PROMPT = "You are a manager who decides to give his subordinate the task: "
MANAGER_TEMPLATE = PromptTemplate(input_variables=[], template=MANAGER_PROMPT)


EMPLOYEE_DOCUMENTATION_INTRO = """
You are a button presser who has access to a Bash terminal. You have diligently pored over your employee training manual, which reads:

-----
{documentation}
-----

Now your boss has a task for you:

{task}""".lstrip()  # noqa

EMPLOYEE_DOCUMENTATION_PROMPT = PromptTemplate(
    input_variables=["documentation", "task"],
    template=EMPLOYEE_DOCUMENTATION_INTRO,
)

EMPLOYEE_TEACHING_INTRO = """
You are a state-of-the-art LLM, hired as an AI employee for the ZAMM firm. Your boss has asked you to perform the following task:

> {task}
{agent_scratchpad}
""".lstrip()  # noqa

EMPLOYEE_TEACHING_INTRO_TEMPLATE = PromptTemplate(
    input_variables=["task", "agent_scratchpad"],
    template=EMPLOYEE_TEACHING_INTRO,
)

FOLLOW_INSTRUCTIONS_INTRO = """
You are a simple button presser who simply follows instructions without doing things very creatively. You always follow every instruction, in order, until the task is done. This includes following instructions in the **Confirmation** section of your employee training manual.

You have access to a Bash terminal and a file editor. The Bash terminal is unable to edit files, so instead you always use the file editor for that.

You have diligently pored over your employee training manual, which reads:

-----
{documentation}
-----

Your boss has asked you to perform the following task:

> {task}

Fortunately, this is exactly the task that the training manual has prepared you for! You follow its instructions closely.

{agent_scratchpad}
""".lstrip()  # noqa

FOLLOW_INSTRUCTIONS_TEMPLATE = PromptTemplate(
    input_variables=["task", "documentation", "agent_scratchpad"],
    template=FOLLOW_INSTRUCTIONS_INTRO,
)

BASH_COMMAND_TEMPLATE = TerminalPromptTemplate(prefix=EMPLOYEE_DOCUMENTATION_PROMPT)


SUCCESS_CRITERIA_PROMPT = (
    EMPLOYEE_DOCUMENTATION_INTRO
    + """

You jot down a one-line version of each item in the success checklist for this task on a notepad:

1. """  # noqa
)

SUCCESS_CRITERIA_TEMPLATE = PromptTemplate(
    input_variables=["documentation", "task"], template=SUCCESS_CRITERIA_PROMPT
)
