import guidance

# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.TextGenerationWebUI()
guidance.llm.caching = False

# define a guidance program that adapts a proverb
program = guidance("""Tweak this proverb to apply to model instructions instead.

{{proverb}}
- {{book}} {{chapter}}:{{verse}}

UPDATED
Where there is no guidance{{gen 'rewrite' stop="\\n-"}}
- GPT {{gen 'chapter'}}:{{gen 'verse'}}""",
)

# execute the program on a specific proverb
executed_program = program(
    proverb="Where there is no guidance, a people falls,\nbut in an abundance of counselors there is safety.",
    book="Proverbs",
    chapter=11,
    verse=14,
    stream=None,
    async_mode=False
)
print("\n\nProgram Result:")
print(executed_program)