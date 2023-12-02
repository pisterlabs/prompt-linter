from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.llms import OpenAI


example_formatter_template = """
Search: {search_term}
Context: {context}
Output: {output}
"""

search_1 = "Vultures"
output1 = """
Relevant Quotes:
    "Slowly, little by little, I lifted the cloth, until a small, small light escaped from under it to fall upon — to fall upon that vulture eye! It was open — wide, wide open, and my anger increased as it looked straight at me. (Poe, 65)"
    "His eye was like the eye of a vulture, the eye of one of those terrible birds that watch and wait while an animal dies, and then fall upon the dead body and pull it to pieces to eat it. When the old man looked at me with his vulture eye a cold feeling went up and down my back; even my blood became cold. (Poe, 64)"
"""
context1 = """
In Edgar Allan Poe's "The Tell-Tale Heart," the presence of vultures is a powerful symbol of the narrator's guilt and fear. The vultures represent death and decay, and serve as a reminder of the narrator's guilt and the consequences of his actions. The vultures also represent the old man's death, as they are often associated with death and decay. The vultures also represent the narrator's fear of being discovered and the consequences of his actions. The vultures serve to emphasize the narrator's guilt and fear, and to remind the reader of the consequences of his actions. Quotes to include could be "Slowly, little by little, I lifted the cloth, until a small, small light escaped from under it to fall upon — to fall upon that vulture eye! It was open — wide, wide open, and my anger increased as it looked straight at me. (Poe, 65)" and 
or 
"His eye was like the eye of a vulture, the eye of one of those terrible birds that watch and wait while an animal dies, and then fall upon the dead body and pull it to pieces to eat it. When the old man looked at me with his vulture eye a cold feeling went up and down my back; even my blood became cold. (Poe, 64)"
"""

search_2 = "Windows"
output2 = """
Relevant Quotes:
    "I am getting angry enough to do something desperate. To jump out of the window would be admirable exercise, but the bars are too strong even to try."
    "That spoils my ghostliness, I am afraid; but I don’t care—there is something strange about the house—I can feel it. I even said so to John one moonlight evening, but he said what I felt was a draught, and shut the window."
"""

context2 = """
In Charlotte Perkins Stetson's "The Yellow Wall-Paper," Windows serve as a symbol of the narrator's longing for a more romantic life, as well as a metaphor for her relationship with her husband. Windows also provide the narrator with a physical barrier between her and the outside world, allowing her to work in peace and observe the beauty of the countryside. Furthermore, Windows can be seen as a metaphor for the narrator's experience in "The Yellow Wall-Paper," as she is confined to a bed and forced to follow a pattern that has no purpose or meaning. By using Windows, users are often confined to a certain set of rules and regulations, and must follow a certain pattern in order to achieve their desired results. In this way, Windows can be seen as a symbol of the mundane and the everyday, and yet it is also a place of refuge and comfort.
Relevant quotes to include are "I am getting angry enough to do something desperate. To jump out of the window would be admirable exercise, but the bars are too strong even to try." and "That spoils my ghostliness, I am afraid; but I don’t care—there is something strange about the house—I can feel it. I even said so to John one moonlight evening, but he said what I felt was a draught, and shut the window."
"""

examples = [
    {
        "search_term": search_1,
        "context": context1,
        "output": output1,
    },
    {
        "search_term": search_2,
        "context": context2,
        "output": output2,
    },
]

def generate_quotes(search_term: str, context: str, openai_api_key: str):
    try:
        if openai_api_key:
          llm_complete = OpenAI(model_name="text-davinci-003", n=1, best_of=1, openai_api_key=openai_api_key)
        else:
          llm_complete = OpenAI(model_name="text-davinci-003", n=1, best_of=1)
        example_prompt = PromptTemplate(
            input_variables=[
                "search_term",
                "context",
                "output",
            ],
            template=example_formatter_template,
        )
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Extract all quotations (anything in quotes) in the context that may relate to the search term. If no quotations are found, return: No relevant quotations found. Check the analysis for some insight.",
            suffix="Search: {search_term}\nContext: {context}\nOutput:",
            input_variables=["search_term","context"],
            example_separator="\n\n",
        )

        final_prompt = few_shot_prompt.format(
            search_term=search_term.strip(),
            context=context.strip(),
        )
        # call API with prompt
        return llm_complete(final_prompt)
    except Exception as e:
        print("Error generating completion: ", e)
        raise e