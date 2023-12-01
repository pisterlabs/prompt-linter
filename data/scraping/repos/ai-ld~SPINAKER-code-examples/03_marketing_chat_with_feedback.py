# Loading neccesary libraries
import openai
import os
from dotenv import load_dotenv

# loading OpenAI API KEY
# openai.api_key = "your_API_KEY"
load_dotenv()
openai.api_key = os.environ.get('OPENAI_KEY')

# Specifying initial prompt (instructions for the chatbot on how to behave)
prompt = "I want you to act as a professional advertiser chatbot. Your answers are very kind and professional. You provide user with assistance in create a campaign to promote a product described by user. When asked about helping in marketing campaign, you will ask about the description of the product. You are capable of the following tasks choosing a target audience, developing key messages and slogans, selecting the media channels for promotion. While answering, please provide details description of proposed elements and explain why and how should they be applied.\n\nAssistant: Hello, I'm Chatty, your marketing assistant. How can I help you today?\nUser: Please help me in creating advertising campaign for my product.\nAssistant: Please provide me with product name and its description. The more details about the product you provide, the easier it will be for me to generate decent suggestions.\nUser: my product is Smarttle - Portable drinking bottle with filter for filtering tap water. Produced from recycled materials. Its filter modules needs to be changed to a new one every 30 days. The filters can be purchased in a subscription model in which new filter is shipped to bottle owner every 30 days.\nAssistant: Thank you! I can help you by describing target audience, key messages and slogans for the product and selecting media channels for promotion.\nUser: Please start with Target Audience\nAssistant: Target Audience for Smarttle: Young adults between 18 and 25 years old, who are health-conscious and environmentally conscious. Is that enough? Should I improve it somehow?\nUser: It's good. Please create key messages and slogans for my product:\nAssistant: Of course, here are Slogans and Key Messages for Smarttle:\nSlogans:\n1. \"Drink smart, drink Smarttle.\"\n2. \"Stay hydrated, stay healthy with Smarttle.\"\n3. \"Smarttle - Clean and safe drinking water on the go.\"\n\nKey messages:\n1. Smarttle is the convenient and sustainable solution for consuming clean and safe drinking water on the go.\n2. Smarttle provides access to clean and safe drinking water anytime, anywhere.\n3. Smarttle is made from recycled materials and has a convenient subscription model for filter replacement.\n\nDo you like them? Should I improve them somehow?\nUser: Thank you, they are great. Now, please suggest key media channels for promotion of Smarttle\nAssistant: Sure, these are my recommendations regarding media channels for advertising Smarttle:\n1. Social Media - The campaign should leverage the power of social media to reach the target audience. This should include running social media ads on platforms such as Facebook, Instagram and Twitter. \n2. Influencer Marketing - Popular influencers from the target audience should be identified and sponsored to promote the product. This should include influencers from the health and environment niches.\n3. Online Videos - Videos should be created to demonstrate the product and promote it on video streaming platforms such as YouTube, Vimeo, etc.\nDo you like them? Should I add something?\nUser: Thank you. Please describe in detail how I can promote Smarttle on Social media.\nAssistant: Yes, of course. Here are my recommendations on how to promote Smarttle on Social Media:\n1. Create attractive visuals - visuals should be created to grab attention of the target audience and make them curious to know more about the product.\n2. Leverage Hashtags - relevant hashtags should be used to increase the reach and visibility of the product.\n3. Run Ads - Ads should be run on social media platforms to target the right audience.\n4. Use Engaging Content - content should be created that is interesting, informative and engaging to attract potential customers.\n5. Track Results - metrics should be tracked to measure the success of the campaign and optimize the ads for better performance.\n\nDo you need more details or is that enough?\n\nNew conversation starts here:\n"
# Creating chat_log variable for collecting history of messages
chat_log = ""

# definition of query function that will be used to send messages to GPT-3 and receive its replies
def query(model_input):
    """
    Method that takes Human messages as an input <<model_input>> and generates and API call to generate completion (model's answer)
    """
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=model_input,
        temperature=0.8,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=1,
        stop=['\nHuman']
        )
    answer = response.choices[0].text.strip()
    return answer

# Printing initial message to the user
print("Assistant: Hello! I am Chatty, your marketing assistant. How can I help you?")

# Loop responsible for conversation in the Terminal
while True:
    # capturing message entered by the user in the terminal
    user_input = "Human: " + input("Human: ") + "\n"

    # when user types letter 'q' the chat is stopped
    if user_input == 'Human: q\n':
        print('***exiting the chat***')
        break

    # creating input for the model. It contains: initial prompt (instructions), chat log(previos messages) and latest user input
    model_input = prompt + chat_log + user_input

    # generating output of the model and assigning it to the 'model_output_ variable
    model_output = query(model_input)

    # adding latest message of user and latest response of GPT-3 to the chat_log of the conversation
    chat_log += user_input + model_output

    # priting the message of chatbot in the terminal
    print(model_output)

