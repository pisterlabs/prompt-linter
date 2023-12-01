import guidance
from dotenv import load_dotenv
import urllib.parse
import streamlit as st

load_dotenv()
guidance.llm = guidance.llms.OpenAI("text-davinci-003") 

def generate_prompt(query):
    prompt_generator = guidance(
    '''
    {{#block hidden=True~}}
    You are a world class prompt generator; Your goal is to generate a prompt that can be used for other LLMs;

    Here is the query: {{query}}
    Prompt: {{gen 'prompt'}}
    {{/block~}}

    {{prompt}}
    ''')

    prompt = prompt_generator(query=query)
    print(prompt)
    
    return prompt

def generate_chart(query):
    
    def parse_chart_link(chart_details):
        encoded_chart_details = urllib.parse.quote(chart_details, safe='')

        output = "![](https://quickchart.io/chart?c=" + encoded_chart_details + ")"

        return output
    
    examples = [
        {
            'input': "Make a chart of the 5 tallest mountains",
            'output': {"type":"bar","data":{"labels":["Mount Everest","K2","Kangchenjunga","Lhotse","Makalu"], "datasets":[{"label":"Height (m)","data":[8848,8611,8586,8516,8485]}]}}
        },
        {
            'input': "Create a pie chart showing the population of the world by continent",
            'output': {"type":"pie","data":{"labels":["Africa","Asia","Europe","North America","South America","Oceania"], "datasets":[{"label":"Population (millions)","data": [1235.5,4436.6,738.8,571.4,422.5,41.3]}]}}
        }
    ]

    guidance.llm = guidance.llms.OpenAI("text-davinci-003") 

    chart = guidance(
    '''    
    {{#block hidden=True~}}
        You are a world class data analyst, You will generate chart output based on a natural language;

        {{~#each examples}}
        Q:{{this.input}}
        A:{{this.output}}
        ---
        {{~/each}}
        Q:{{query}}
        A:{{gen 'chart' temperature=0 max_tokens=500}}    
    {{/block~}}
    Hello here is the chart you want
    {{parse_chart_link chart}}
    ''')

    return chart(query=query, examples=examples, parse_chart_link=parse_chart_link)

def generate_story(story_idea):
        
    story = guidance('''
    {{#block hidden=True~}}
    You are a world class story teller; Your goal is to generate a short tiny story less than 200 words based on a story idea;

    Here is the story idea: {{story_idea}}
    Story: {{gen 'story' temperature=0}}
    {{/block~}}

    {{#block hidden=True~}}
    You are a world class AI artiest who are great at generating text to image prompts for the story; 
    Your goal is to generate a good text to image prompt and put it in a url that can generate image from the prompt;

    Story: You find yourself standing on the deck of a pirate ship in the middle of the ocean. You are checking if there are still people on the ship
    Image url markdown: ![Image](https://image.pollinations.ai/prompt/a%203d%20render%20of%20a%20man%20standing%20on%20the%20deck%20of%20a%20pirate%20ship%20in%20the%20middle%20of%20the%20ocean)
                    
    Story: {{story}}
    Image url markdown: {{gen 'image_url' temperature=0 max_tokens=500}})
    {{~/block~}}
                        
    Story: {{~story~}}
    {{image_url}}
    ''')

    story = story(story_idea=story_idea)
    print(story)
    return story

def generate_email(email):        
    priorities = ["low priority", "medium priority", "high priority"]

    email_generator = guidance(
    '''    
    {{#block hidden=True~}}
    Here is the customer message we received: {{email}}
    Please give it a priority score 
    priority score: {{select "priority" options=priorities}}
    {{~/block~}}
                
    {{#block hidden=True~}}
    You are a world class customer support; Your goal is to generate an response based on the customer message and next steps;
    Here is the customer message to respond: {{email}}
    Generate an opening & one paragraph of response to the customer message at {{priority}}:
    {{gen 'email'}} 
    {{~/block~}}

    {{email}}

    {{#if priority=='high priority'}}Would love to setup a call this/next week, here is the calendly link: https://calendly.com/janik-dotzel{{/if}}

    Best regards

    Janik
    ''')

    email_response = email_generator(email=email, priorities=priorities)
    print(email_response)

    return email_response

def generate_image(query):
        
    image_generator = guidance('''
    {{#block hidden=True~}}
    You are a world class AI artiest who are great at generating text to image prompts for the story; 
    Your goal is to generate a good text to image prompt and put it in a url that can generate image from the prompt;

    Story: You find yourself standing on the deck of a pirate ship in the middle of the ocean. You are checking if there are still people on the ship
    Image url: https://image.pollinations.ai/prompt/a%203d%20render%20of%20a%20man%20standing%20on%20the%20deck%20of%20a%20pirate%20ship%20in%20the%20middle%20of%20the%20ocean
                    
    Story: {{query}}
    Image url: {{gen 'image_url' temperature=0 max_tokens=500}})
    {{~/block~}}

    {{image_url}}
    ''')

    image_url = str(image_generator(query=query))    
    print(image_url)
    
    return image_url

def generate_tweet(query):        
    tweet_generator = guidance(
    '''    
    {{#block hidden=True~}}
    You are a world class marketing genius. You excel through your twitter markeing. You write tweets that drive engagement.
    Your goal is to generate a tweet based on a query.
    Here is the query: {{query}}
    Generate a text for a tweet. Make sure it follows the rules of a tweet like max characters, etc.:
    {{gen 'tweet'}} 
    {{~/block~}}
                
    {{tweet}}
    ''')

    ratings = ["bad tweet", "medium tweet", "great tweet"]

    tweet_reviewer = guidance(
    '''
    {{#block hidden=True~}}
    You are a world class marketing genius. You excel through your twitter markeing. You write tweets that drive engagement.
    Here is the tweet that we received and you should rate: {{tweet}}
    Please rate it.  
    Rating: {{select "rating" options=ratings}}
    
    {{rating}}
    {{~/block~}}
    ''')

    # Generate multiple drafts of a tweet based on the query and rate each draft using tweet_reviewer.
    # If a draft receives a "great tweet" rating, select it as the final tweet.
    # If none of the drafts receive a "great tweet" rating, select the last draft as the final tweet.
    for _ in range(5):
        draft = tweet_generator(query=query)
        rating = tweet_reviewer(tweet=draft, ratings=ratings)
        if rating == "great tweet":
            tweet = draft
            break
    else:
        tweet = draft

    print(tweet)
    return tweet


def main():
    st.set_page_config(page_title="Content Generator", page_icon="☁️")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Prompt Generator", "Chart Generator", "Story Generator", "Email Generator", "Image Generator", "Tweet Generator"])  

    with tab1:
        st.header("Prompt Generator")
        prompt = st.text_input("Enter a query of the prompt you want to generate.")
        if prompt:
            genereated_prompt = generate_prompt(prompt)            
            st.markdown(genereated_prompt)

    with tab2:
        st.header("Chart Generator")
        prompt = st.text_input("Enter a query of the chart you want to generate.")
        if prompt:
            chart = generate_chart(prompt)            
            st.markdown(chart)
    
    with tab3:
        st.header("Story Generator")
        prompt = st.text_input("Enter a story idea to generate a story.")
        if prompt:
            story = generate_story(prompt)            
            st.markdown(story)

    with tab4:
        st.header("Email Generator")
        prompt = st.text_area("Enter a customer email to generate a response.")
        if prompt:
            email_response = generate_email(prompt)            
            st.write(email_response)  

    with tab5:
        st.header("Image Generator")
        prompt = st.text_area("Enter a prompt to generate an image.")
        if prompt:
            image = generate_image(prompt)            
            st.image(image)  

    with tab6:
        st.header("Tweet Generator")
        prompt = st.text_input("Enter a query of the tweet you want to generate.")
        if prompt:
            tweet = generate_tweet(prompt)            
            st.markdown(tweet)

if __name__ == '__main__':
    main()
