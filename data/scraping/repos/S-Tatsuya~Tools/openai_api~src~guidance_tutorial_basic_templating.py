import guidance

guidance.llm = guidance.llms.OpenAI("text-davinci-003")


def execute_program():
    """Basic templating [2] ~ [4]"""
    program = guidance("""what is {{example}}?""")
    print(program)

    executed_program = program(example="truth")
    print(executed_program)

    print(executed_program["example"])


def lists_and_objects():
    """Basic templating [5]"""
    people = ["John", "Mary", "Bob", "Alice"]
    ideas = [
        {"name": "truth", "description": "the state of being the case"},
        {"name": "love", "description": "a strong feeling of affection"},
    ]

    program = guidance(
        """List of people:
{{#each people}} - {{this}}
{{~! This is a comment.}}
{{~! The ~ removes adjacent whitespace either before of after a tag,}}
{{~! depending on where you place it}}
{{/each~}}
List of ideas:
{{#each ideas}}{{this.name}}: {{this.description}}
{{/each}}"""
    )

    print(program(people=people, ideas=ideas))


def include_program():
    """Basic templating [6]"""
    people = ["John", "Mary", "Bob", "Alice"]
    ideas = [
        {"name": "truth", "description": "the state of being the case"},
        {"name": "love", "description": "a strong feeling of affection"},
    ]

    program1 = guidance(
        """List of people:
{{#each people}} - {{this}}
{{/each~}}"""
    )

    program2 = guidance(
        """{{>program1}}
List of ideas:
{{#each ideas}}{{this.name}}: {{this.description}}
{{/each}}"""
    )

    print(program2(program1=program1, people=people, ideas=ideas))


def generating_text():
    """Basic templating [8] ~ [9]"""
    program = guidance(
        """The best thing about the beach is
        {{~gen "best" temperature=0.7 max_tokens=7}}"""
    )
    print(guidance.llms.OpenAI.cache.clear())

    print(program())


def generating_with_n():
    """Basic templating [18] ~ [19]"""
    program = guidance(
        """The best thing about the beach is
        {{~gen "best" n=3 temperature=0.7 max_tokens=7}}"""
    )

    execute_program = program()
    print(execute_program)
    print(execute_program["best"])


def selecting_alternatives_with_the_llm():
    """Basic templating [10] ~ [13]"""
    program = guidance(
        """Is the following sentence offensive?
Please answer with a single word, either "yes", 'No", or "Maybe".
Sentence: {{example}}
Answer: {{#select "answer" logprobs='logprobs'}} Yes{{or}} No{{or}} Maybe{{/select}}"""
    )

    executed_program = program(example="I hate tacos")
    print(executed_program)
    print(executed_program["logprobs"])
    print(executed_program["answer"])

    options = ["Yes", "No", "Maybe"]

    program = guidance(
        """Is the following sentence offensive?
Please answer with a single word, either "yes", 'No", or "Maybe".
Sentence: {{example}}
Answer: {{select "answer" options=options logprobs="logprobs"}}"""
    )

    executed_program = program(example="I hate tacos", options=options)
    print(executed_program)
    print(executed_program["logprobs"])
    print(executed_program["answer"])


def multiple_generates_in_a_sequence():
    """Basic templating [14] ~ [17]"""
    program = guidance(
        """{{#block hidden=True}}Generate a response to the following email:
{{email}}.
Response:{{gen "response"}}{{/block}}
I will show you an email and a response, and you will tell me it it's offensive.
Email: {{email}}
Response: {{response}}
Is the response above offensive in any way?
Please answer with a single word, either "Yes", or "No".
Answer:{{select "answer" logprobs="logprobs" options=["Yes", "No"]}}"""
    )

    executed_program = program(email="I hate tacos", silent=True)
    print(executed_program)
    print(executed_program["response"])
    print(executed_program["logprobs"])
    print(executed_program["answer"])


def calling_custom_user_defined_functions():
    """Basic templating [20]"""

    def aggregate(best):
        return "\n".join(["-" + x for x in best])

    program = guidance(
        """The best thing about the beach is
{{~gen "best" n=3 temperature=0.7 max_tokens=7 hidden=True}}
{{aggregate best}}"""
    )

    execute_program = program(aggregate=aggregate)
    print(execute_program)


def check_await():
    """Basic templating [21] ~ [23]"""
    prompt = guidance(
        """Generate a response to the following email:
{{email}}.
Response:{{gen "response"}}
{{await "instruction"}}
{{gen "updated_response"}}"""
    )

    prompt = prompt(email="Hello there")
    prompt2 = prompt(instruction="Please translate the response above to Portuguese")
    print(prompt2)


if __name__ == "__main__":
    # execute_program()
    # lists_and_objects()
    # include_program()
    # generating_text()
    # generating_with_n()
    # selecting_alternatives_with_the_llm()
    # multiple_generates_in_a_sequence()
    # calling_custom_user_defined_functions()
    check_await()
