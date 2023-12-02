from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from agent.models import advanced_summary_model
from utils.template import TEST_TWEET_THREAD_TEMPLATE, TWEET_THREAD_TEMPLATE, REWRITE_TWEET_TEMPLATE


def generate_tweets(video_info):

    # video_link = video['url']
    # video_title = video['title']
    # video_summary = video['summary']
    
    # Display video info we are about to send
    basic_info = video_info.split("\n\n")[0]
    print(f"This is the video about to be converted to tweets: {basic_info}")
    
    template = TWEET_THREAD_TEMPLATE

    prompt = ChatPromptTemplate.from_template(template=template)
    
    chain = prompt | advanced_summary_model | StrOutputParser()

    q = {"video_info": video_info}

    twitter_thread = chain.invoke(q)

    return twitter_thread


def rewrite_tweets(tweet):
    template = REWRITE_TWEET_TEMPLATE

    prompt = ChatPromptTemplate.from_template(template=template)
    
    chain = prompt | advanced_summary_model | StrOutputParser()

    q = {"tweet": tweet}

    rewritten_tweet = chain.invoke(q)

    return rewritten_tweet
    

if __name__ == "__main__":

    pass
    
    