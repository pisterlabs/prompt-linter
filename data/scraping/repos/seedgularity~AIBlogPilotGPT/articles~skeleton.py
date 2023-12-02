import sys
import openai
from dotenv import load_dotenv
import colorama
colorama.init(autoreset=True)
from config import Config

load_dotenv()

cfg = Config()

# Configure OpenAI API key
try:
    openai.api_key = cfg.openai_api_key
    browserless_api_key = cfg.browserless_api_key
    llm_model = cfg.llm_model
except KeyError:
    sys.stderr.write("OpenAI key configuration failed.")
    exit(1)

def write_skeleton_product_review(title):
    prompt = """Write a product review for """+title+"""". Use the following information:
    
    Have you been looking for the best [keyword] recipe? I've got you covered!
>write an [tone] introduction about [keyword] including [experience], [expertise], authority and trustworthiness talking to [audience] who are [pain point].
Optional add ons to the intro prompt to keep it on track :
Describe the flavors of the dish as [flavors].
The ingredients in this dish are [x].
The dish is made with [type of cooking method]
[highlight an affiliate related to this post]
What is [keyword]?
>write a high-level overview of [keyword] to [audience] explaining [feature(s)]
[Keyword] Ingredients
>write a paragraph of the ingredients needed to make [keyword]. This ingredients are [list ingredients]. Include [note and specific features/health benefits or allergen notices]
How to Make [keyword]
>write a list of # steps explaining how to [keyword] to [audience]
You will take this list and put them into h3s. You can skip this if you have particular steps already.
Step 1: [step here]
>write a paragraph explaining [step] in [how to do X] in simple terms. Explain why this is necessary and how it helps
Step 2: [step here]
>write a paragraph explaining [step] in [how to do X] in simple terms. Explain why this is necessary and how it helps
[highlight an affiliate related to this post. Duplicate this or move this to areas where you can promote a certain cooking tool or an ingredient from Amazon]
Substitutes/Alternatives/Variations for [Keyword]
>write a paragraph about alternative/substitute/variations for [keyword] replacing [ingredient/cooking method] with [replacement]
You can rerun this command if there are multiple substitutes or variations.
[highlight an affiliate related to this post]
Any Other Headings Go Here
>write a paragraph about [header] for [audience] including [x]
This is a great place to answer questions about the best brands for ingredients, where to buy things, if they can replace x with y, and varied cooking methods.
[highlight an affiliate related to this post]
[keyword] Recipe
[insert the recipe block from either WP recipe maker or Create by Mediavine. I recommend filling this in manually. Include an image in the recipe template]
FAQs
>write a list of FAQs that [audience] would ask about [keyword] and answer them
Alternatively, find questions then ask Jasper to:
>write a concise answer to this question: "question" including [x]
[highlight an affiliate related to this post]
Conclusion/Final Thoughts: How to [keyword]
>write an engaging conclusion for a blog post about [keyword] talking to [audience]. Include a call to action for readers to [action. i.e. read next post titled “x”, leave a comment, buy a course titled “x” that does “y”]
Read More:
[insert relevant read more block here]

Strictly follow the example structure of JSON:
    
    {
    "Title": "The Role of Energy Efficiency in Home Design",
    "Description": "This article focuses on the importance and incorporation of energy efficiency in home design and its resulting benefits",
    "Sections": [
        {
            "Heading_H2": "Incorporating Energy Efficiency in Home Design",
            "Description": "A detailed guide on how energy efficiency can be embedded into home design, with sub-sections including",
            "SubSections": [
                {
                "Heading_H3": "Passive Design Strategies",
                "Description": "Explanation of passive design strategies to optimize energy efficiency."
                },
                {
                "Heading_H3": "Selecting Efficient Appliances",
                "Description": "Guide on choosing energy-efficient appliances for the home."
                },
                {
                "Heading_H3": "Material Selection for Energy Efficiency",
                "Description": "Discussion on how material selection impacts energy efficiency and the best materials to choose.",
                  "SubSections": [
                      {
                      "Heading_H4": "Insulation Materials",
                      "Description": "Analysis on the role of insulation materials in enhancing energy efficiency."
                      },
                      {
                      "Heading_H4": "Window Materials",
                      "Description": "Guide to choosing energy-efficient window materials."
                      }
                  ]
                }
            ]
        }
    ]
}
"""

    chunked_output = ""
    for chunk in openai.ChatCompletion.create(
            model=llm_model,
            stream=True,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'system', 'content': "You're an expert in blogging and SEO."},
                {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                {'role': 'system', 'content': 'Return the output as JSON.'},
                {"role": "user", "content": prompt}
            ]
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            #print(content, end='')
            chunked_output += content

    return chunked_output

def write_skeleton(title):
    prompt = """Propose a structure for an article with title """+title+"""". 
    Make sure you include headings - H2, H3, H4.
    Please ensure to include actual names in the structure.
    Include description for each section.
    Add FAQ section if relevant. Strictly follow the example structure of JSON below. Make sure the JSON is valid.
    
    {
    "Title": "The Role of Energy Efficiency in Home Design",
    "Description": "This article focuses on the importance and incorporation of energy efficiency in home design and its resulting benefits",
    "Sections": [
        {
            "Heading_H2": "Incorporating Energy Efficiency in Home Design",
            "Description": "A detailed guide on how energy efficiency can be embedded into home design, with sub-sections including",
            "SubSections": [
                {
                "Heading_H3": "Passive Design Strategies",
                "Description": "Explanation of passive design strategies to optimize energy efficiency."
                },
                {
                "Heading_H3": "Selecting Efficient Appliances",
                "Description": "Guide on choosing energy-efficient appliances for the home."
                },
                {
                "Heading_H3": "Material Selection for Energy Efficiency",
                "Description": "Discussion on how material selection impacts energy efficiency and the best materials to choose.",
                  "SubSections": [
                      {
                      "Heading_H4": "Insulation Materials",
                      "Description": "Analysis on the role of insulation materials in enhancing energy efficiency."
                      },
                      {
                      "Heading_H4": "Window Materials",
                      "Description": "Guide to choosing energy-efficient window materials."
                      }
                  ]
                }
            ]
        }
    ]
}"""

    chunked_output = ""
    for chunk in openai.ChatCompletion.create(
            model=llm_model,
            stream=True,
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'system', 'content': "You're an expert in blogging and SEO."},
                {'role': 'system', 'content': 'Your name is BloggingGPT.'},
                {'role': 'system', 'content': 'Return the output as JSON.'},
                {"role": "user", "content": prompt}
            ]
    ):
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            #print(content, end='')
            chunked_output += content

    return chunked_output

if __name__ == "__main__":
    write_skeleton_product_review("Review of Acure Ultra Hydrating Shampoo")