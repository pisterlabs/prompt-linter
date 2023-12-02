from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
import tweepy

# Project imports
from utils.publishers.IPublisher import IPublisher

class TwitterPublisher(IPublisher):
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret, temperature: float = 0.5):
        super().__init__()
        
        self.temperature = temperature
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api  = tweepy.API(self.auth)

    def build(self, text: str, link: str):
        tweet = None
        docs = None
        llm = ChatOpenAI(model='gpt-3.5-turbo-1106', temperature=self.temperature)

        template_prompt = """
        {text}

        Please suggest in only one line what does above text do. 
        Respond in first person. The response must be it catchy, engaging and suitable for a single tweet.
        Do include link {link} in the response.
        Please sparingly use phrase 'Dive into the', instead use similar catchy and appealing phrases
        """

        prompt_template = PromptTemplate(
            template=template_prompt
            , input_variables=['text', 'link']
        )

        tweet_prompt = prompt_template.format(text=text, link=link)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=10
        )

        # we split the data into multiple chunks
        try:
            docs = text_splitter.create_documents([text, link])
            combine_chain = load_summarize_chain(
                llm=llm, 
                chain_type='stuff'
            )
            tweet = combine_chain.run(docs)
        except Exception as e:
            print("Exception Occurred: ", e)
            exit(2)

        print(tweet.strip())

        return tweet
    
    def publish(self, content):
        message = content.message
        image = content.image

        self._tweet(message)
        self._tweet_with_image(message=message, image=image) if content.publish_image else None

    def _tweet(self, message):
        self.api.update_status(message)

    def _tweet_with_image(self, message, image):
        self.api.update_with_media(image, message)