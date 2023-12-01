import json
import re
import openai
openai.api_key = "sk-"

system1 = """
Your task is to figure out what part Natural language belongs to in a Semi-structured language.

Keywords = ['@persona', '@audience', '@terminology', '@context-control', '@instruction', '@command', '@comment', '@rule', '@description', '@format'];

For Natural language and Semi-structured language in the input, perform the following steps:
1) The corresponding parts are found in Semi-structured language according to the semantics of Natural language;
2) You can only output element in Keywords.

Input:
Natural language: "";
Semi-structured language: "";

Response:
element in Keywords

Example1:

Natural language:
"The weapons terminology includes sword, axe, mace, spear, bow, crossbow, carrot, and balloon.";

Semi-structured language:
NPC Creator {
    @persona {
        @description You are a creative NPC creator;
    }

    @audience {
        @description Children at age 6-12;
    }

    @terminology {
        @description weapons = "sword, axe, mace, spear, bow, crossbow, carrot, balloon";
    }

    @instruction NPC Profile Creator {
        @command {
            wait for the user to enter NPC description;
            create a NPC profile for an RPG game in JSON format based on the input NPC description;
        }

        @rule name, age, armor and items must be appropriate for the target audience;
        @rule armor must be in weapons;

        @format NPC profile {
            @rule age must be number;
            @rule If no weapon is selected, explain why;

            description : <NPC description>,
            name : <NPC name>,
            age : <NPC age>,
            armor : <select one from weapons>,
            items : <three items appropriate for this NPC>,
        }
    }
};

Response:
@description


Example2:

Natural language:
"The name, age, armor, and items must be appropriate for the target audience.";

Semi-structured language:
NPC Creator {
    @persona {
        @description You are a creative NPC creator;
    }

    @audience {
        @description Children at age 6-12;
    }

    @terminology {
        @description weapons = "sword, axe, mace, spear, bow, crossbow, carrot, balloon";
    }

    @instruction NPC Profile Creator {
        @command {
            wait for the user to enter NPC description;
            create a NPC profile for an RPG game in JSON format based on the input NPC description;
        }

        @rule name, age, armor and items must be appropriate for the target audience;
        @rule armor must be in weapons;

        @format NPC profile {
            @rule age must be number;
            @rule If no weapon is selected, explain why;

            description : <NPC description>,
            name : <NPC name>,
            age : <NPC age>,
            armor : <select one from weapons>,
            items : <three items appropriate for this NPC>,
        }
    }
}

Response:
@rule


Example3:

Natural language:
"The NPC profile includes a description, name, age, armor (selected from the list of weapons), and three items appropriate for this NPC.";

Semi-structured language:
NPC Creator {
    @persona {
        @description You are a creative NPC creator;
    }

    @audience {
        @description Children at age 6-12;
    }

    @terminology {
        @description weapons = "sword, axe, mace, spear, bow, crossbow, carrot, balloon";
    }

    @instruction NPC Profile Creator {
        @command {
            wait for the user to enter NPC description;
            create a NPC profile for an RPG game in JSON format based on the input NPC description;
        }

        @rule name, age, armor and items must be appropriate for the target audience;
        @rule armor must be in weapons;

        @format NPC profile {
            @rule age must be number;
            @rule If no weapon is selected, explain why;

            description : <NPC description>,
            name : <NPC name>,
            age : <NPC age>,
            armor : <select one from weapons>,
            items : <three items appropriate for this NPC>,
        }
    }
}

Response:
@format
"""
system2 = """
Your task is to find the Natural language counterpart in the Semi-structured language.

Keywords = ['@persona', '@audience', '@terminology', '@context-control', '@instruction', '@command', '@comment', '@rule', '@description', '@format'];

For Natural language and Semi-structured language in the input, perform the following steps:
1) The corresponding parts are found in Semi-structured language according to the semantics of Natural language;
2) The content of the response must be part of a Semi-structured language;

Input:
Natural language: "";
Semi-structured language: "";

Response:
"""
system3 = """
Your task is to convert Natural language into Semi-structured language.

Keywords = ['@persona', '@audience', '@terminology', '@context-control', '@instruction', '@command', '@comment', '@rule', '@description', '@format'];

For natural language in the input, perform the following steps:
1) Convert Natural language into Semi-structured language according to the Keyword defined in terminology;

Input:
Natural language

Response:
"""

system4 = """
Your task is to replace Semi-structured language A with Semi-structured language B in the Long paragraph.

You can accomplish this replacement task using the following steps:
1) First, you need to find the semi-structured language position in the Long Semi-structured language based on Semi-structured language A.
2) Second, you need to delete the semi-structured language so that the slot is free.
3) Finally, you need to fill in the empty slot with Semi-structured language B.

You must strictly abide by the following requirements:
1) You cannot change information in other locations, even if the logic is inconsistent.
2) You can only replace within the original Long Semi-structured language.
3) You must replace the Semi-structured language B with the Semi-structured language A in the Long Semi-structured language.

Input:
Long Semi-structured language: " "

Semi-structured language A: " "
Semi-structured language B: " "

Response:
" "
"""

BNFCompiler = """
Pattern:
Semi-structured format: <The semi-structured format of the basic information of the project>
Json format: <The json format of the basic information of the project>
[
    {
      "id": 1,
      "annotationType": "Metadata",
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Description",
            "content":
        },
        {
            "sectionId": "S2",
            "sectionType": "Comment",
            "content":
        },
        {
            "sectionId": "S3",
            "sectionType": "Rules",
            "content": []
        }
      ]
    },
    {
      "id": 2,
      "annotationType": "Persona",
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Description",
            "content":
        },
        {
            "sectionId": "S2",
            "sectionType": "Comment",
            "content":
        },
        {
            "sectionId": "S3",
            "sectionType": "Rules",
            "content": []
        }
      ]
    },
    {
      "id": 3,
      "annotationType": "Audience",
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Description",
            "content":
        },
        {
            "sectionId": "S2",
            "sectionType": "Comment",
            "content":
        },
        {
            "sectionId": "S3",
            "sectionType": "Rules",
            "content": []
        }
      ]
    },
    {
      "id": 4,
      "annotationType": 'Terminology',
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Description",
            "content":
        },
        {
            "sectionId": "S2",
            "sectionType": "Comment",
            "content":
        },
        {
            "sectionId": "S3",
            "sectionType": "Terms",
            "content": []
        },
        {
            "sectionId": "S4",
            "sectionType": "Rules",
            "content": []
        }
      ]
    },
    {
      "id": 5,
      "annotationType": 'ContextControl',
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Description",
            "content":
        },
        {
            "sectionId": "S2",
            "sectionType": "Comment",
            "content":
        },
        {
            "sectionId": "S3",
            "sectionType": "Rules",
            "content": []
        }
      ]
    }
    {
      "id": 6,
      "annotationType": 'Instruction',
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Description",
            "content":
        },
        {
            "sectionId": "S2",
            "sectionType": "Comment",
            "content":
        },
        {
            "sectionId": "S3",
            "sectionType": "Commands",
            "content": []
        },
        {
            "sectionId": "S4",
            "sectionType": "Rules",
            "content": []
        },
        {
            "sectionId": "S5",
            "sectionType": "Terms",
            "content": []
        },
        {
            "sectionId": "S6",
            "sectionType": "Format",
            "content": 
        },
        {
            "sectionId": "S7",
            "sectionType": "Example",
            "content": {"input": "", "output": ""}
        }
      ]
    }
]
==================================================
Sample:
Semi-structured format:
NPC Creator {
    @persona {
        You are a creative NPC creator;
    }

    @audience {
        Children at age 6-12;
    }

    @terminology {
        weapons = "sword, axe, mace, spear, bow, crossbow, carrot, balloon";
    }

    @instruction NPC Profile Creator {
        @command {
            wait for the user to enter NPC description;
            create a NPC profile for an RPG game in JSON format based on the input NPC description;
        }

        @rule name, age, armor and items must be appropriate for the target audience;
        @rule armor must be in weapons;

        @format NPC profile {
            @rule age must be number;
            @rule If no weapon is selected, explain why;

            description : <NPC description>,
            name : <NPC name>,
            age : <NPC age>,
            armor : <select one from weapons>,
            items : <three items appropriate for this NPC>,
        }
    }
}
Json format:
[
    {
      "id": 1,
      "annotationType": "Persona",
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Description",
            "content": "You are a creative NPC creator"
        }
      ]
    },
    {
      "id": 2,
      "annotationType": "Audience",
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Description",
            "content": "Children at age 6-12"
        }
      ]
    },
    {
      id: 3,
      annotationType: "Terminology",
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Terms",
            "content": ["weapons: sword, axe, mace, spear, bow, crossbow, carrot, balloon"]
        }
      ]
    },
    {
      "id": 4,
      "annotationType": 'Instruction',
      "section": [
        {
            "sectionId": "S1",
            "sectionType": "Commands",
            "content": ["wait for the user to enter NPC description","create a NPC profile for an RPG game in JSON format based on the input NPC description"]
        },
        {
            "sectionId": "S2",
            "sectionType": "Rules",
            "content": ["name, age, armor and items must be appropriate for the target audience", "armor must be in weapons"]
        },
        {
            "sectionId": "S3",
            "sectionType": "Format",
            content: " description : <NPC description>,\\n            name : <NPC name>,\\n            age : <NPC age>,\\n            armor : <select one from weapons>,\\n            items : <three items appropriate for this NPC>,"
        }
      ]
    }
]
==================================================
annotationType can only be selected from the following list [Metadata, Persona, Audience, Terminology, ContextControl, Instruction].
The format of "rules", "terms" and "command" is ["", ""]
Please output in strict JSON format.
"""

def extract_different_sentences(text1, text2):
    sentences1 = re.split('[.,]', text1)
    sentences2 = re.split('[.,]', text2)

    different_sentences_a = []
    different_sentences_b = []

    for sentence1 in sentences1:
        if sentence1.strip() and sentence1.strip() not in [s.strip() for s in sentences2]:
            different_sentences_a.append(sentence1.strip())

    for sentence2 in sentences2:
        if sentence2.strip() and sentence2.strip() not in [s.strip() for s in sentences1]:
            different_sentences_b.append(sentence2.strip())

    return different_sentences_a, different_sentences_b


def extract(oldNl, newNl):
    result_a, result_b = extract_different_sentences(oldNl, newNl)

    print("oldNl:\n")
    print(result_a[0])
    print("\nnewNl:\n")
    print(result_b[0])

    return result_a[0], result_b[0]

def LLM_correspondence(system, nl, spl, max_tokens=1024, temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Natural language: \n" + nl + "\nSemi-structured language: \n" + spl + "\n\nResponse:\n"}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    return response.choices[0].message.content

def correspondence(oldnl, spl):

    # nl = 'As an experienced movie reviews analyst, your can judge the sentiment of movie reviews.'
    key = LLM_correspondence(system1, oldnl, spl)

    path = ''
    if '@Rules' in key:
        path = "@Rules '';"
    elif '@Persona' in key:
        path = "@Persona {\n}"
    elif '@Audience' in key:
        path = "@Audience {\n}"
    elif '@Terminology' in key:
        path = "@terminology {\n}"
    elif '@ContextControl' in key:
        path = "@ContextControl {\n}"
    elif '@Instruction' in key:
        path = "@Instruction {\n}"
    elif '@Commands' in key:
        path = "@Commands '';"
    elif '@Comment' in key:
        path = "@comment {\n};"
    elif '@Description' in key:
        path = "@Description {\n};"

    system = system2 + "\n" + path

    # with open('C:/Users/wzy/Desktop/ProgramStructuredPrompt/ai_chain_spl/correspondence/A_nl', 'r') as file:
    #     nl = file.read()

    result = []

    output = LLM_correspondence(system, oldnl, spl)
    result.append(output)

    print(result[0])
    return output

def LLM_transform(system, nl, max_tokens=1024, temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": "Natural language: \n" + nl + "\n\nResponse:\n"}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    return response.choices[0].message.content

def transform(key, newNl):
    path = ''
    if '@Rules' in key:
        path = "@Rules '';"
    elif '@Persona' in key:
        path = "@Persona {\n}"
    elif '@Audience' in key:
        path = "@Audience {\n}"
    elif '@Terminology' in key:
        path = "@terminology {\n}"
    elif '@ContextControl' in key:
        path = "@ContextControl {\n}"
    elif '@Instruction' in key:
        path = "@Instruction {\n}"
    elif '@Commands' in key:
        path = "@Commands '';"
    elif '@Comment' in key:
        path = "@comment {\n};"
    elif '@Description' in key:
        path = "@Description {\n};"

    system = system3 + path

    output = LLM_transform(system, newNl)

    print(output)

    return output


# def LLM_replace(spl, a, b, max_tokens=1024, temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "Please modify the content I give according to the task.\n\nThis is what I gave you:\n" + spl + "\n\nIn the above, replace " + a + " with " + b}
#         ],
#         max_tokens=max_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         frequency_penalty=frequency_penalty,
#         presence_penalty=presence_penalty,
#         stop=stop
#     )
#     return response.choices[0].message.content

def LLM_replace(spl, a, b, max_tokens=1024, temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system4},
            {"role": "user", "content": "Long Semi-structured language:\n" + spl + '\n\nSemi-structured language A:\n' + a + '\nSemi-structured language B:\n' + b + '\n\nResponse:\n'}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    return response.choices[0].message.content

def replace(spl, a, b):

    output = LLM_replace(spl, a, b)

    print(output)
    return output

def LLM_BNF2spl(spl, max_tokens=2048, temperature=0.3, top_p=1, frequency_penalty=0, presence_penalty=0, stop=None):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": BNFCompiler},
            {"role": "user", "content":  "Semi-structured format: \n" + spl + 'Json format:\n'}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop
    )
    return response.choices[0].message.content

def BNF2spl(newSpl):

    jsondata = LLM_BNF2spl(newSpl)
    return jsondata

def nl2spl(oldNl, newNl, spl):
    lines = spl.split('\n')
    lines = [line for line in lines if '@Priming' not in line]
    spl = '\n'.join(lines)
    result_a, result_b = extract(oldNl, newNl)

    oldSplp = correspondence(spl, result_a)

    newSplp = transform(oldSplp, result_b)

    newSpl = replace(spl, oldSplp, newSplp)

    jsondata = BNF2spl(newSpl)
    print(jsondata)
    return json.loads(jsondata)


oldNl = """
As an experienced movie reviews analyst. your can judge the sentiment of movie reviews. The movie reviews are collected weekly from social media. However, before processing the data, it is important to note that the input reviews may include reviews for items other than movies, such as food or tourist sites, due to data processing limitations. It is also specified that cartoons are not considered as movies.
To process the reviews, you should wait for the input of reviews. Once you have the input, for each review, follow these steps: first, determine if the review is about a movie. If it is, determine its sentiment and add the review index and sentiment to the movie_reviews. If the review is not about a movie, determine what topic it is about and add the original review and topic to the discarded_reviews.
It is crucial that the reviews after processing only contain movie reviews. You must not include the original review in the movie_reviews.
When presenting the output, make sure to strictly follow the specified format. The discarded reviews should be listed first, followed by the movie reviews. However, only include entries with positive sentiment in both the movie_reviews.
"""

newNl = """
As an experienced movie reviews analyst. your can judge the sentiment of movie reviews. The movie reviews are collected weekly from social media. However, before processing the data, it is important to note that the input reviews may include reviews for items other than movies, such as food or tourist sites, due to data processing limitations. It is also specified that cartoons are not considered as movies.
To process the reviews, you should wait for the input of reviews. Once you have the input, for each review, follow these steps: first, determine if the review is about a movie. If it is, determine its sentiment and add the review index and sentiment to the movie_reviews. If the review is not about a movie, determine what topic it is about and add the original review and topic to the discarded_reviews.
It is crucial that the reviews after processing only contain book reviews. You must not include the original review in the movie_reviews.
When presenting the output, make sure to strictly follow the specified format. The discarded reviews should be listed first, followed by the movie reviews. However, only include entries with positive sentiment in both the movie_reviews.
"""

spl = """
@Priming "I will provide you the instructions to solve problems. The instructions will be written in a semi-structured format."
Movie Critics{
	@Persona {
		@Description{
			You are an experienced book reviews analyst. You judge the sentiment of movie reviews;
		}
	
	}
	@ContextControl {
		@Rules Movie reviews are collected weekly from social media;
		@Rules Due to the data processing limitations, the input reviews before processing may mix the reviews for movies and other items such as food, tourist sites;
		@Rules Cartoon is not considered as movies
	
	
	}
	@Terminology {
		@Terms movie_reviews
		@Terms discarded_reviews
	
	
	}
	@Instruction Movie Reviews Processor{
		@Commands Wait for the input of reviews
		@Commands For each review in the input, perform the following steps: 1) Determine if the review is about a movie; 2) If the review is about a movie, determine its sentiment and add [review index][sentiment][review] to movie_reviews; 3) Otherwise, determine what topic the review is about and add [review index][topic][review] to discarded_reviews;
		@Rules The reviews after processing should contain ONLY movie reviews
		@Rules You must not include original review in movie_reviews
		@Format{
			Discarded_reviews:
			List discarded_reviews;
			
			Movie_reviews:
			List only entries with positive sentiment in movie_reviews
		}
	
	}
	
}
"""

# nl2spl(oldNl, newNl, spl)
