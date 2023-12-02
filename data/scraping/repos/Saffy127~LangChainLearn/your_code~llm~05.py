from langchain import PromptTemplate, FewShotPromptTemplate

examples = [
  {"animal": "octopus", "home": "reef"},
  {"animal": "chimp", "home": "jungle"},
]

example_template ="""\
Animal: {animal}
Home: {home}\
"""

