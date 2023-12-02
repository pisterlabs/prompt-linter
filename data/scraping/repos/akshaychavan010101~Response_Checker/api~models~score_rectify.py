import openai
import os
from dotenv import load_dotenv

load_dotenv()


def generatePrompt(score, feedback):
    return f'''
You are a critic. You will be given feedback, rubrics, the grading system, and a score between 0 to 10. You tend to reduce the score if the feedback has scope for improvement or is suggested by the below rubrics and the grading system to improve the score. 
The rubrics to be followed:
```
Rubric 1: Tone is not matched
Rubric 2: Brief- not too lengthy not too short
Rubric 3: Should not include complex or technical child psychology terms.
Rubric 4: Should have Personal touch
```
Grading system:
```
0-2: For Irrelevant or out of context(Factually incorrect or irrelevant info) answers.
3-4: For Unusable answers too long or formal, Bullet points, and concepts like answers.
5-6: For Not matching the tone and language of the answers with a score near to 10.
7-8: For answers with the personal touch
9-10: For the Best and ideal answer 
```
Generate the score after reducing the amount of negation factor in the feedback or the score based on the grading system.
Note: Only generate the final score without any explaination.
--------------------------------------------
Score: {score}
Feedback: {feedback}
'''


def rectifier(s, f):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "system", "content": "You are a helpful assistant that speaks the same language as the user."},
                  {"role": "user", "content": generatePrompt(s, f)}],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    result = response["choices"][0]["message"]["content"]
    return result
