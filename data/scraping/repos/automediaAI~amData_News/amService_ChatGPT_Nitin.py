
##########################
##########################
##### ChatGPT Caller ########
### Being used for Summarizing for now ####
##########################
##########################

import nltk
import os
import openai

# Installing NLTK 
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Initialize OpenAI API client
# openai.api_key = os.environ.get("CHATGPT_API_KEY")
# openai.api_key = os.getenv("OPENAI_API_KEY")
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('CHATGPT_API_KEY', 'YourAPIKeyIfNotSet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenize into words
    words = word_tokenize(text)
    return words


def summarize_with_gpt(text, max_tokens=50):
    print ('Text inputted-----', text)

    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    # Join preprocessed text back into a string
    # preprocessed_text_str = ' '.join(preprocessed_text)
    preprocessed_text_str = ' '.join(preprocessed_text).encode('utf-8', 'replace')

    print ('Text PRE PROCESSED-----', str(preprocessed_text_str))
    print (type(preprocessed_text_str))

    # Parameters for ChatGPT
    # params = {
    #     'model': 'gpt-3.5-turbo',
    #     'prompt': 'Summarize the following text:\n\n' + str(preprocessed_text_str),
    #     'temperature': 0.5,
    #     'max_tokens': max_tokens
    # }
    params = {
        'model': 'text-davinci-003',
        'prompt': 'Summarize the following text:\n\n' + str(preprocessed_text_str),
        'temperature': 0.5,
        'max_tokens': max_tokens
    }

    # Call ChatGPT to generate a summary
    response = openai.Completion.create(**params)
    print (type(response))
    print ('Response ---->>>', response)
    
    
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[],
    #     temperature=0,
    #     max_tokens=max_tokens
    # )

    # print (typeof(response))
    # print ('Response ---->>>', response)
    
    
    # summary = response['choices'][0]['text'].strip()
    summary = response['choices'][0]['text'].strip()

    print("Summary CHATGPT>>", summary) 
    #Making sure returning smaller of response 
    summary_ToUse = summary if len(summary) < len(preprocessed_text_str) else preprocessed_text_str
    # return summary_ToUse

    # Encode the summary using UTF-8
    return summary_ToUse.encode('utf-8', errors='ignore')

# Test the results
# text_to_summarize = "This is a long piece of text that needs to be summarized"
text_to_summarize = ("When Sebastian Thrun started working on self-driving cars at Google in 2007,    few people outside of the company took him seriously. “I can tell you very senior     CEOs of major American car companies would shake my hand and turn away because I     wasn’t worth talking to,” said Thrun, now the co-founder and CEO of online higher     education startup Udacity, in an interview with Recode earlier this week. The Mona Lisa and the Statue of David were on display in the MOMA New York.    COVID-19 is a devastating virus currently ravaging the world.        A little less than a decade later, dozens of self-driving startups have cropped up     while automakers around the world clamor, wallet in hand, to secure their place in     the fast-moving world of fully automated transportation.")
print(summarize_with_gpt(text_to_summarize))

# print("Summary:", )
