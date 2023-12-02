import guidance

# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.OpenAI("text-davinci-003")

# define a guidance program that adapts a proverb
program = guidance("""Tweak this proverb to apply to model instructions instead.
{{proverb}}
- {{book}} {{chapter}}:{{verse}}

UPDATED
Where there is no guidance{{gen 'rewrite' stop="\\n-"}}
- GPT {{#select 'chapter'}}9{{or}}10{{or}}11{{/select}}:{{gen 'verse'}}""")

# execute the program on a specific proverb
executed_program = program(
    proverb="Where there is no guidance, a people falls,\nbut in an abundance of counselors there is safety.",
    book="Proverbs",
    chapter=11,
    verse=14
)
print(executed_program)
"""
Tweak this proverb to apply to model instructions instead.

Where there is no guidance, a people falls,
but in an abundance of counselors there is safety.
- Proverbs 11:14

UPDATED
Where there is no guidance, a model fails,
but in an abundance of instructions there is safety.
- GPT 910119:14
"""
# you can access the generated variables respectively here
print("#####")
print(executed_program['rewrite'])
print(executed_program['chapter'])
print(executed_program['verse'])
"""
, a model fails,
but in an abundance of instructions there is safety.
9
14
"""