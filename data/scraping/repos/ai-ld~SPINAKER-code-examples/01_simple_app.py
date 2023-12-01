import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get('OPENAI_KEY')

prompt = "I want you to act as an professional advertiser. You will create a campaign to promote a product described below. You will choose a target audience, develop key messages and slogans, select the media channels for promotion, and propose additional activities needed to reach campaign goals. While answering, please provide details description of proposed elements and explain why and how should they be applied.\n\nProduct: Smarttle - Portable drinking bottle with filter for filtering tap water. Produced from recycled materials. Its filter modules needs to be changed to a new one every 30 days. The filters can be purchased in a subscription model in which new filter is shipped to bottle owner every 30 days.\n\nTarget audience: Young adults between 18 and 25 years old, who are health-conscious and environmentally conscious.\n\nKey messages:\n\n1. Smarttle is the convenient and sustainable solution for consuming clean and safe drinking water on the go.\n\n2. Smarttle provides access to clean and safe drinking water anytime, anywhere.\n\n3. Smarttle is made from recycled materials and has a convenient subscription model for filter replacement.\n\nSlogans:\n\n1. \"Drink smart, drink Smarttle.\"\n\n2. \"Stay hydrated, stay healthy with Smarttle.\"\n\n3. \"Smarttle - Clean and safe drinking water on the go.\"\n\nMedia channels:\n\n1. Social Media - The campaign should leverage the power of social media to reach the target audience. This should include running social media ads on platforms such as Facebook, Instagram and Twitter. \n\n2. Influencer Marketing - Popular influencers from the target audience should be identified and sponsored to promote the product. This should include influencers from the health and environment niches.\n\n3. Online Videos - Videos should be created to demonstrate the product and promote it on video streaming platforms such as YouTube, Vimeo, etc.\n\nAdditional Activities:\n\n1. Content Marketing - Content creation should be a part of the campaign. This should include writing blog posts, creating visuals, and sharing them on social media platforms.\n\n2. SEO Optimization - SEO optimization should be done to ensure that the product appears in search results. This should include optimizing the website and creating quality content.\n\n3. Email Marketing - An email list should be created and used to send out newsletters and promotional emails to target audience.\n\nProduct: "

user_input = input("Please provide product name and description: ")

model_input = prompt + user_input

def query(model_input):
    """
    Method that takes Human messages as an input <<question>> and generates and API call to generate completion (model's answer)
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=model_input,
        temperature=0.8,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=1
        )
    answer = response.choices[0].text.strip()
    return answer

model_output = query(model_input)

print(model_output)
 