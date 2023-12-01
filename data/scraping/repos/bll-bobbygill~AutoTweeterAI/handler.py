import json
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.chains import SimpleSequentialChain, LLMChain, SequentialChain
from langchain import PromptTemplate
from langchain.document_loaders import CSVLoader
from langchain.memory.simple import SimpleMemory
from langchain.document_loaders import RedditPostsLoader
from langchain.llms import PromptLayerOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch

import promptlayer
import random
import os

from config import REDDIT_CLIENT_ID,REDDIT_CLIENT_SECRET,REDDIT_USER_AGENT,PROMPT_LAYER_API_KEY,PROMPT_LAYER_API_KEY
from constants import PROMPTLAYER_TAG,NUM_TWEETS_TO_GENERATE,NUM_TWEETS_TO_SAMPLE,SAMPLE_TWEETS_FILENAME,REDDIT_SAMPLE_SIZE,OPEN_AI_MODEL
from prompttemplates import *
from utilities import convertStringToDataArray
from gsheet import append_to_google_sheet


def getRedditPosts()-> str:
      # load using 'subreddit' mode
    loader = RedditPostsLoader(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        categories=["hot"],  # List of categories to load posts from
        mode="subreddit",
        search_queries=[
            "worldnews"
        ],  # List of subreddits to load posts from
        number_posts=20,  # Default value is 10
    )

    reddit_posts = loader.load()
    reddit_posts_objs = [{"id":post.metadata["post_id"],"title":post.metadata["post_title"],"url":post.metadata["post_url"]} for post in reddit_posts]

    #now we select N topics from the array
    number_to_sample=min(REDDIT_SAMPLE_SIZE,len(reddit_posts_objs))

    selected_reddit_posts=random.sample(reddit_posts_objs,number_to_sample)

    selected_reddit_posts_as_text=""
    for value in selected_reddit_posts:
        selected_reddit_posts_as_text += "\n\n"+value["title"]
    return selected_reddit_posts_as_text

def getSampleTweetsUsingRetrievalQA(selected_reddit_posts_as_text)->str:
    file = SAMPLE_TWEETS_FILENAME
    loader = CSVLoader(file_path=file,encoding="utf8", csv_args={'delimiter':',','quotechar': '"','fieldnames':['Tweet']})
    tweets = loader.load()
    print("Loaded tweets history from file: "+SAMPLE_TWEETS_FILENAME)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_tweets = text_splitter.split_documents(tweets)
    print("Completed splitting of sample tweets file into chunks")

    embeddings = OpenAIEmbeddings()
    db = DocArrayInMemorySearch.from_documents(
        split_tweets, 
        embeddings)
    print("Loaded split, chunked tweets into DocArrayInMemory")
    query = TWEET_CSV_QA_TEMPLATE + selected_reddit_posts_as_text
    print("Executing following similarity query on doc array in memory")
    docs = db.similarity_search(query)
    reference_tweets = "\n".join([docs[i].page_content for i in range(len(docs))])
    print("Query returned following reference tweets as result of query")
    return reference_tweets




def getSampleTweets()->str:
    file = SAMPLE_TWEETS_FILENAME
    loader = CSVLoader(file_path=file,encoding="utf8", csv_args={'delimiter':',','quotechar': '"','fieldnames':['Tweet']})
    tweets = loader.load()

    number_of_tweets_to_use = NUM_TWEETS_TO_SAMPLE
    #Now we need to pull X tweets out of the entire CSV

    random_sample_of_tweets = random.sample(tweets,number_of_tweets_to_use)
    list_of_tweets = [x.page_content for x in random_sample_of_tweets][:number_of_tweets_to_use]

    #now we concatenate this into a single string
    tweets_as_text = "\n\n".join(list_of_tweets)
    return tweets_as_text

def getGeneratedTweets(sample_tweets,reddit_posts)->str:
    #create prompt templates for generating new tweets
    template = GENERATE_TWEETS_TEMPLATE

    promptlayer.api_key = PROMPT_LAYER_API_KEY
    openai = promptlayer.openai
    gpt_model = OPEN_AI_MODEL
    openai.api_key = os.environ.get('OPENAI_API_KEY');

    llm = PromptLayerChatOpenAI(model=OPEN_AI_MODEL, pl_tags=[PROMPTLAYER_TAG])
    prompt_template = PromptTemplate(input_variables=["number_of_tweets_to_generate", "example_tweets", "topics"],template=template)
    llm_chain = LLMChain(llm=llm,prompt=prompt_template)
    response = llm_chain.predict(number_of_tweets_to_generate=NUM_TWEETS_TO_GENERATE,example_tweets=sample_tweets,topics=reddit_posts)
    return response

def getGeneratedAndModeratedTweets(sample_tweets,reddit_posts)->str:
    promptlayer.api_key = PROMPT_LAYER_API_KEY
    openai = promptlayer.openai
    gpt_model = OPEN_AI_MODEL
    openai.api_key = os.environ.get('OPENAI_API_KEY');

    
    #Used for Chat Models
    llm = PromptLayerChatOpenAI(model=OPEN_AI_MODEL, pl_tags=[PROMPTLAYER_TAG])
    print("Using Open AI Model: "+OPEN_AI_MODEL+", using key: "+openai.api_key)
    #Used for Completion Models like 'davinci'
    #llm = PromptLayerOpenAI(model=OPEN_AI_MODEL, pl_tags=[PROMPTLAYER_TAG])

    
    #we setup our first chain
    prompt_template = PromptTemplate(input_variables=["number_of_tweets_to_generate", "example_tweets", "topics"],template=GENERATE_TWEETS_TEMPLATE)
    generator_chain = LLMChain(llm=llm,prompt=prompt_template,output_key="proposed_tweets")
    print ("Completed setting up of Generator Chain.")

    #we setup our 'make sure its funny chain'
    funny_chain_prompt_template = PromptTemplate(input_variables=["proposed_tweets"], template=MAKE_IT_FUNNY_TEMPLATE)
    funny_chain = LLMChain(llm=llm,prompt=funny_chain_prompt_template,output_key="final_tweets")
    print ("Completed setting up of Funny Chain.")

    #we setup our moderation chain
    prompt_template2 = PromptTemplate(input_variables=["final_tweets"], template=MODERATE_TWEETS_TEMPLATE)
    moderation_chain = LLMChain(llm=llm, prompt=prompt_template2,output_key="moderated_tweets")
    print ("Completed setting up of Moderation Chain.")


    overall_chain = SequentialChain(
        chains=[generator_chain,funny_chain,moderation_chain],
        input_variables=["number_of_tweets_to_generate", "example_tweets", "topics"],
        output_variables=["moderated_tweets"])
    
    print ("Beginning execution over Overall Chain containing 3 sub-chains.")
    return overall_chain.run({"number_of_tweets_to_generate":NUM_TWEETS_TO_GENERATE, "example_tweets":sample_tweets, "topics":reddit_posts})


def generateTweets(event, context):
    print('Beginning execution of Generated Tweets function...')
    reddit_posts = getRedditPosts()   
    print('Samples Reddit Posts:\n\n '+reddit_posts)
    
    #load tweets from a csv file
    sample_tweets = getSampleTweetsUsingRetrievalQA(reddit_posts)

    print('Sampled Tweets:\n\n'+sample_tweets)

    #we make our call to the OpenAI to generate new tweet content
    print('Beginning generation of new tweets...')
    new_tweets = getGeneratedAndModeratedTweets(sample_tweets,reddit_posts)
    print('Newly generated Tweets:\n\n'+new_tweets)
    
    #now we convert the returned string into a data list
    new_tweets_data = convertStringToDataArray(new_tweets)

    #now we append them to the Google Sheet
    append_to_google_sheet(new_tweets_data)

    print('Generated Tweets completed successfully')


    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response


