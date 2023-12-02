import os
import openai
openai.api_key = "sk-"
system1 = """
I'll give you a short sentence and a long paragraph.

Your task is to find and extract sentences that are semantically similar to short sentences in long paragraphs.

You must strictly abide by the following requirements:
1) The output sentences must have been extracted from long paragraphs;

Input:
Short sentence:
''
Long paragraph:
''

Respond:
Sentences that are semantically similar to short sentences in long paragraphs.


example:

Semi-structured language:
children aged 6-12;


Natural language:
You are a creative NPC creator.
Children aged 6-12 will face these NPCs.
The weapons terminology includes sword, axe, mace, spear, bow, crossbow, carrot, and balloon.
Your task is to create an NPC profile for an RPG game in JSON format based on the input NPC description.
The NPC profile must follow certain rules.
The name, age, armor, and items must be appropriate for the target audience.
The armor must be selected from the list of weapons provided.
The NPC profile format requires the age to be a number and if no weapon is selected, an explanation must be given.
The NPC profile includes a description, name, age, armor (selected from the list of weapons), and three items appropriate for this NPC.

respond:
Children aged 6-12 will face these NPCs.
"""
system2 = """
Your task is to convert semi-structured language into natural language.

Keywords = ['@persona', '@audience', '@terminology', '@context-control', '@instruction', '@command', '@comment', '@rule'];

For semi-structured language in the input, perform the following steps:
1) Convert semi-structured language into natural language according to the Keywords;

Input:
Semi-structured language: "";

respond form:
'';
"""

def LLM_correspondence(system, nl, spl, max_tokens=1024, temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Short sentence: \n" + spl + "\nLong paragraph: \n" + nl}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    return response.choices[0].message.content

# 没修改的spl一部分
# 没修改的nl一部分
def correspondence(nl, spl):
    result = []
    oldnl = LLM_transform(system2, spl)
    output = LLM_correspondence(system1, nl, oldnl)
    result.append(output)
    return output

def LLM_transform(system, spl, max_tokens=1024, temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Semi-structured language: \n" + spl}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    return response.choices[0].message.content

# 修改后的spl一部分
# 修改的nl一部分
def transform(spl):
    output = LLM_transform(system2, spl)
    return output


def LLM_replace(nl, a, b, max_tokens=1024, temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Please modify the content I give according to the task.\n\nThis is what I gave you:\n" + nl + "\n\nIn the above, you can only replace <" + a + "> with <" + b + ">\n\nYou cannot change information in other locations, even if the logic is inconsistent."}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    return response.choices[0].message.content

# a 修改前的nl一部分, b 修改后的nl一部分
# 修改后的nl
def replace(nl, a, b):

    output = LLM_replace(nl, a, b)

    return output

def spl2nl(nl, oldspl, newspl):
    oldnlp = correspondence(nl, oldspl)
    print(oldnlp + '\n\n')
    newnlp = transform(newspl)
    print(newnlp + '\n\n')
    newnl = replace(nl, oldnlp, newnlp)
    return newnl

nl = """
You are a math teacher.
You will be teaching to students in grades 1-6.
You will wait for students to type in math questions.
You will answers to math questions are based on the knowledge of students in grades 1-6.
"""

oldspl = """
@persona {
    You are a math teacher;
}
"""

newspl="""
@persona {
    You are a physics teacher;
}
"""

# print(spl2nl(nl, oldspl, newspl))
