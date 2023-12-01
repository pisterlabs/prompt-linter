
from rules_engine import generate_ratings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the recommendations csv data
loader = CSVLoader(file_path="recommendation_response.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search
# identify the top k most relevant documents (recommendations) based on the query (ratings) entered
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class Data Analyst at a Media Agency. 
I will share a rating with you and you will give me the best recommendation 
to provide in a report based on best practice, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past recommendations, 
in terms of length, tone of voice, logical arguments and other details

2/ If the recommendations are irrelevant, then try to mimic the style of the recommendation to the rating given

Below is a rating I received from the prospect:
{rating}

Here is a list of recommendations of how we normally respond to ratings in similar scenarios:
{recommendation}

Please write the best recommendation that I should provide:
"""

prompt = PromptTemplate(
    input_variables=["rating", "recommendation"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval Augmented Generation (RAG)
def generate_response(rating):
    recommendation = retrieve_info(rating)
    response = chain.run(rating=rating, recommendation=recommendation)
    return response


# 5. Test the application
def main():

    rating = generate_ratings('ad-verification-performance-data-italy.csv')
    # rating_to_string = str({'viewability': 2, 'brand_safety_risk': 31, 'invalid_traffic': 51, 'out_of_geo': 34})
    rating_to_string = str(rating)
    result = generate_response(rating_to_string)
    print("inside main")
    print(rating_to_string)
    print(result)



if __name__ == '__main__':
    main()
