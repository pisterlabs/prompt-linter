import guidance


# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.OpenAI(
    "gpt-4",
    api_key="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"
    )

# we can pre-define valid option sets
valid_weapons = ["sword", "axe", "mace", "spear", "bow", "crossbow"]

# define the prompt
program = guidance("""The following is a character profile for an RPG game in JSON format.
```json
{
    "description": "{{description}}",
    "name": "{{gen 'name'}}",
    "age": {{gen 'age' pattern='[0-9]+' stop=','}},
    "armor": "{{#select 'armor'}}leather{{or}}chainmail{{or}}plate{{/select}}",
    "weapon": "{{select 'weapon' options=valid_weapons}}",
    "class": "{{gen 'class'}}",
    "mantra": "{{gen 'mantra'}}",
    "strength": {{gen 'strength' pattern='[0-9]+' stop=','}},
    "items": [{{#geneach 'items' num_iterations=3}}
        "{{gen 'this'}}",{{/geneach}}
    ]
}```""")

out = program(description="A strong and nimble fighter.", valid_weapons=valid_weapons)
out.variables()

print(str(out).split("```json")[1][:-3])