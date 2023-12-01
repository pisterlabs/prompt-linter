import guidance
                                                      
# set the default language model used to execute guidance programs
guidance.llm = guidance.llms.OpenAI("text-davinci-003") 


# define the guidance program
structure_program = guidance(
'''Given a sentence tell me whether it contains an anachronism (i.e. whether it could have happened or not based on the time periods associated with the entities).

Answer in the following format:
---
Sentence: the sentence given 
Entities and dates: the entities (e.g. person, animal, etc) with their dates (either past, present or future)
Reasoning: show your reasoning on whether or not the sentence contains anarchonism
Anachronism: either conclude with Yes or No
---

{{~! place the real question at the end }}
Sentence: {{input}}
Entities and dates: {{gen "entities"}}
Reasoning: {{gen "reasoning"}}
Anachronism:{{#select "answer"}} Yes{{or}} No{{/select}}''')

# execute the program
out = structure_program(
    input='The T-rex bit my  dog'
)
print(out)

"""
Given a sentence tell me whether it contains an anachronism (i.e. whether it could have happened or not based on the time periods associated with the entities).

Answer in the following format:
---
Sentence: the sentence given 
Entities and dates: the entities (e.g. person, animal, etc) with their dates (either past, present or future)
Reasoning: show your reasoning on whether or not the sentence contains anarchonism
Anachronism: either conclude with Yes or No
---
Sentence: The T-rex bit my  dog
Entities and dates:  T-rex (extinct 65 million years ago), dog (present)
Reasoning: The T-rex is extinct 65 million years ago, while the dog is present, thus it is impossible for the T-rex to bite the dog.
Anachronism: Yes
Reasoning: 
"""

print("########")
print(out['answer'])
"""
Yes
"""

"""
예시를 넣으면 semantic contamination의 위험이 있다. 예시를 넣는 대신 각 스텝의 설명을 충분히 하는 것이 나을 수 있다. 
위처럼 설명하면 되는 것이다. 그래도 답은 동일하다.
"""