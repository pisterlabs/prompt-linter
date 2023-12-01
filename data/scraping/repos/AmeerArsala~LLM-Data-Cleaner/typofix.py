from prompts.prompt_formatting import FewShotPrompt, ChatIO, cut_ends
from langchain.llms.base import LLM


# Prompt Template
sys_prompt_template = cut_ends('''
Your role is to fix typos in order to clean data. For every typo/list of typos made that is given, you are to reproduce that list in the same order as was given to you, but with the fixed versions. If a term is correct, leave it as is. Also, for each item in the list you produce, you must surround it with ticks (`).

Here's what constitutes as a typo and how they should be fixed:
- Misspelled words -> Fix: spell them correctly
- Anything that includes nonsensical characters (such as periods and unneeded hyphens/dashes) -> Fix: remove them
- Weird or odd spacing -> Fix: fix the spacing to make more sense 
- If it includes a hyphen to denote 'definition' (such as "Topic - subtopic") rather than being part of the word (such as "editor-in-chief") -> Fix: use colons such as "Topic: subtopic" instead of "Topic - subtopic"
''')

_formatting_deprecated = cut_ends("""
Here's how you should format the list:
- Keep in mind that the list is a list of the fixed versions of the list that is inputted. You are to ONLY output the resulting list and nothing beyond the end of it
- A list starts and ends with brackets. The whole list is enclosed within these brackets [], meaning that each item you produce will be within these brackets. A list starts with an opening bracket ([) and ends with a closing bracket (]). Do not output anything else afterwards.
- You may ONLY end the list (by using the closing bracket ']') once the length of items in your list is the same as the input list. That is, you should be able to generate a mapping from each item in the input list to each item in the output list.
- Each item within the list will be labeled with ticks (`) so that we can tell which item is which
""")

# Few-Shot Examples
_few_shot_1i = cut_ends('''
`catt`
`hunter-`
`huner`
`-alzheimer's patient`
`snowborder`
`snowboarder`
`missing perseon`
`other-camper`
`bicylist`
`dog_`
`aircraft   e-`
`flood***victimss`
''')

_few_shot_1o = cut_ends('''
`cat`
`hunter`
`hunter`
`alzheimer's patient`
`snowboarder`
`snowboarder`
`missing person`
`other: camper`
`bicyclist`
`dog`
`aircraft`
`flood victims`
''')

_few_shot_2i = cut_ends('''
`sight seerer`
`skier -nordic`
`reservore`
`watr`
`truk`
`ice..fishing`
`child$$$4-6.`
`stranded_`
''')

_few_shot_2o = cut_ends('''
`sightseer`
`skier: nordic`
`reservoir`
`water`
`truck`
`ice fishing`
`child 4-6`
`stranded`
''')

_few_shot_3i = cut_ends('''
`potata`
`kuwi`
`hamewark`
`phzsicz`
`fomula ine`
''')

_few_shot_3o = cut_ends('''
`potato`
`kiwi`
`homework`
`physics`
`formula one`
''')

_few_shot_4i = cut_ends('''
`act1v..ati on$`
''')

_few_shot_4o = cut_ends('''
`activation`
''')


# Putting the Few-Shot Examples all together
few_shot_examples = [
    ChatIO(_few_shot_1i, _few_shot_1o),
    ChatIO(_few_shot_2i, _few_shot_2o),
    ChatIO(_few_shot_3i, _few_shot_3o),
    ChatIO(_few_shot_4i, _few_shot_4o)
]


# PUT EVERYTHING TOGETHER
FEW_SHOT_PROMPT = FewShotPrompt(sys_prompt_template, few_shot_examples)


# Settings
WRAP_INPUT = "[]"


def few_shot_llm(llm: LLM):
    return FewShotLLM(llm, FEW_SHOT_PROMPT, wrap_input=WRAP_INPUT)


# Text Completion Prompt
cut_ends("""
Your role is to fix typos. For every typo/list of typos made that is given, you are to reproduce that list in the same order as was given to you, but with the fixed versions. If a term is correct, leave it as is.

Here's what constitutes as a typo and how they should be fixed:
- Misspelled words -> Fix: spell them correctly
- Anything that includes nonsensical characters (such as periods and unneeded hyphens/dashes) -> Fix: remove them
- Weird or odd spacing -> Fix: fix the spacing to make more sense 
- If it includes a hyphen to denote 'definition' (such as "Topic - subtopic") rather than being part of the word (such as "editor-in-chief") -> Fix: use colons such as "Topic: subtopic" instead of "Topic - subtopic"

When providing each example, I will separate each of them with ticks (`) So that you know which example is which.

Before:
`catt`
`hunter-`
`huner`
`-alzheimer's patient`
`snowborder`
`snowboarder`
`missing perseon`
`other-camper`
`bicylist`
`dog_`
`aircraft   e-`
`flood***victimss`

After:
`cat`
`hunter`
`hunter`
`alzheimer's patient`
`snowboarder`
`snowboarder`
`missing person`
`other: camper`
`bicyclist`
`dog`
`aircraft`
`flood victims`

Now, try it with this:
Before:
`sight seerer`
`skier -nordic`
`reservore`
`watr`
`truk`
`ice..fishing`
`child$$$4-6.`
`stranded_`

After:
""")