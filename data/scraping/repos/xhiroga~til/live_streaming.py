import guidance
from dotenv import load_dotenv

load_dotenv()


# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.OpenAI("text-davinci-003")

# define a guidance program that adapts a proverb
program = guidance(
    """Tweak this proverb to apply to model instructions instead.

{{proverb}}
- {{book}} {{chapter}}:{{verse}}

UPDATED
Where there is no guidance{{gen 'rewrite' stop="\\n-"}}
- GPT {{gen 'chapter'}}:{{gen 'verse'}}"""
)

# execute the program on a specific proverb
executed_program = program(
    # 箴言(11:14): 指導者がなければ民は倒れ、助言者が多ければ安全である。
    proverb="Where there is no guidance, a people falls,\nbut in an abundance of counselors there is safety.",
    book="Proverbs",
    chapter=11,
    verse=14,
)
print(executed_program)
