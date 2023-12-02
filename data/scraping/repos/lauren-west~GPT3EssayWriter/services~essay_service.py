import os
import openai

# authentication
openai.organization = "org-4NKqHPrsBbd1Yi3HVZP1cZr2"
if (os.getenv("OPENAI_ORG")):
    openai.organization = os.getenv("OPENAI_ORG")
openai.api_key = os.getenv("OPENAI_API_KEY")
essay_engine_id = 'text-davinci-002'

def make_GPT3_api_call(prompt):
    result = openai.Completion.create(
        engine=essay_engine_id,
        prompt=prompt,
        max_tokens=1000,
        frequency_penalty=2
    )
    return result["choices"][0]["text"]

def make_grade(essay):
    result = openai.Completion.create(
        engine=essay_engine_id,
        prompt=f"Grade the following essay on a 100 point scale: {essay}\"",
        max_tokens=1000,
        frequency_penalty=2,
        stop=["\""]
    )
    return result["choices"][0]["text"]

def write_essay(essay_prompt, paragraph_number = 5):
    prompt = f"Make an Outline with {paragraph_number} paragraphs answering the following prompt:\n{essay_prompt}"
    outline = make_GPT3_api_call(prompt)

    outline = outline.split("\n")
    outline = [x for x in outline if x != ""]
    outline = outline[1:]
    outline = [ts[3:] for ts in outline]

    prompt = f"Create a strong, original thesis statement about this essay prompt:\n{essay_prompt}."
    thesis = make_GPT3_api_call(prompt)

    paragraphs = []
    for topic_sentence in outline:
        if topic_sentence == outline[0]:
            # intro paragraph
            prompt = f"Generate a convincing, factful paragraph with the topic sentence: {topic_sentence}. Be sure to \
            include authentic insights into the topics and this thesis: {thesis}. Minimum total word count is {paragraph_number * 175} words. \
            Split it into {paragraph_number} paragraphs"
        elif topic_sentence == outline[-1]:
            # conclusion
             prompt = f"Generate a convincing, factful paragraph with the topic sentence: {topic_sentence}. Minimum total word count \
                is {paragraph_number * 175} words. Summarize the key supporting ideas you discussed in the thesis ({thesis}), and \
                offer your final impression on the central idea."
        else:
            prompt = f"Generate a convincing, factful paragraph with the topic sentence: {topic_sentence}. Be sure to \
                include authentic insights into the topics. Minimum total word count is {paragraph_number * 175} words. \
                Split it into {paragraph_number} paragraphs."
        paragraphs.append(make_GPT3_api_call(prompt))
    paragraphs = '\n\n'.join(paragraphs)

    essay = paragraphs

    grade = make_grade(essay)

    return essay, grade


# print(write_essay("Evaluate the state of World Politics and its influence on an average U.S. citizen."))








