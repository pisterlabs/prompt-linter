print("Importing Guidance")
import guidance
                                     
# set the default language model used to execute guidance programs
print("Getting LLM")
guidance.llm = guidance.llms.Transformers("gpt2")
print("Done importing...")

# define the few shot examples
examples = [
    {'input': 'I wrote about shakespeare',
    'entities': [{'entity': 'I', 'time': 'present'}, {'entity': 'Shakespeare', 'time': '16th century'}],
    'reasoning': 'I can write about Shakespeare because he lived in the past with respect to me.',
    'answer': 'No'},
    {'input': 'Shakespeare wrote about me',
    'entities': [{'entity': 'Shakespeare', 'time': '16th century'}, {'entity': 'I', 'time': 'present'}],
    'reasoning': 'Shakespeare cannot have written about me, because he died before I was born',
    'answer': 'Yes'}
]

print("Defining Guidance")
# define the guidance program
structure_program = guidance(
'''Given a sentence tell me whether it contains an anachronism.
----

{{~! display the few-shot examples ~}}
{{~#each examples}}
Sentence: {{this.input}}
Entities and dates:{{#each this.entities}}
{{this.entity}}: {{this.time}}{{/each}}
Reasoning: {{this.reasoning}}
Anachronism: {{this.answer}}
---
{{~/each}}

{{~! place the real question at the end }}
Sentence: {{input}}
Entities and dates:
{{gen "entities"}}
Reasoning:{{gen "reasoning"}}
Anachronism:{{#select "answer"}} Yes{{or}} No{{/select}}''')

print("Executing Guidance")
# execute the program
out = structure_program(
    examples=examples,
    input='The T-rex bit my dog'
)

print("Printing Output")
print(out)

""" from syllables import syllable_count, remove_punctuation

print(syllable_count("malo"))
# Load lyrics and count syllables
with open("./files/lyrics/Havana.txt", "r") as f:
    pattern = []
    lyrics = f.readlines()
    for line in lyrics:
        line = remove_punctuation(line)
        pattern.append(line)

    print(pattern)
    # count the number of unique entries
    unique = []
    for line in pattern:
        if not line in unique:
            unique.append(line)
        else:
            print("Appeared again:", line)

    print("Total", len(pattern))
    print("Unique lines:", len(unique)) """