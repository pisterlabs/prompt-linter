import openai
import os

# Using my API key to use the openai python kit
os.environ["OPENAI_API_KEY"] = "Insert open ai api key here"
openai.api_key = os.getenv("OPENAI_API_KEY")

def content_analysis(title, article):
    # Using OpenAI's GPT-3 model to analyse headline and article
    prompt = f"Please evaluate the following article based on how well it addresses or answers the narrative presented in the headline. Use a score from 0 to 10 (with decimals allowed), where 0 represents the least relevant and very clickbaity it means there is a lack of relevance and the article does not sufficiently address the headline, this may also be articles that can address the headline but the article does not go to the lengths not hold the investigatory prowess a headline that is baiting the viewer in should hold, while a score of 10 represents an extremely thorough and well-crafted article that addresses the headline perfectly and goes to great investigatory lengths to answer the headline and answer any questions or curiosity a viewer might have after reading the headline. To determine if a headline is clickbait or not, please evaluate it in light of the article's content, look for evidence, and consider the tone and all aspects of the language which may suggest if the article is clickbait or not. The key here above all else is: does the article satisfy the curiosity the headline raises in the headline. Make sure to be strict with scoring process. \n\nHeadline: {title}\n\nArticle: {article}\n\nScore: "
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.4,
    )
    
    # Extracting the score from the OpenAI response
    score = float(response.choices[0].text.strip())
    
    # Using OpenAI's GPT-3 model to provide an explanation for the score
    prompt = f"Please explain why you gave the article a score of {score}/10 based on how well it addresses or answers the narrative presented in the headline. \n\nHeadline: {title}\n\nArticle: {article}\n\nExplanation: "
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.5,
    )
    
    # Extracting the explanation from the OpenAI response
    explanation = response.choices[0].text.strip()
    
    # Returning the score and explanation as a tuple
    return (score, explanation)

title = input("Headline: ")
article = input("Article: ")
score, explanation = content_analysis(title, article)
print(f"\033[91mOverall relevance of the article to the headline: {score}/10\033[0m")
print(f"\033[94mExplanation: {explanation}\033[0m")
