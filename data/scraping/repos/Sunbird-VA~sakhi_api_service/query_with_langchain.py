import logging
import openai
from gpt_index import SimpleDirectoryReader
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, OpenAI, LLMChain
from cloud_storage import *
import shutil
import json
import csv
from io import StringIO
import time
from dotenv import load_dotenv


log_format = '%(asctime)s - %(thread)d - %(threadName)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('jugalbandi_api')

promptsInMemoryDomainQues = []
promptsInMemoryTechQues = []

score_language_mapping =  {
    'en': 0.41,
    'hi': 0.35,
    'kn': 0.26
}
default_language = 'en'
source_default_msg = {
    'en': "Here are some references links that you may enjoy:",
    'hi': "यहां कुछ संदर्भ लिंक दिए गए हैं जिनका आप आनंद ले सकते हैं:",
    'kn': "ನೀವು ಆನಂದಿಸಬಹುದಾದ ಕೆಲವು ಉಲ್ಲೇಖ ಲಿಂಕ್‌ಗಳು ಇಲ್ಲಿವೆ:"
}
search_default_msg  = {
    'en': "I'm sorry, but I don't have enough information to provide a specific answer for your question. Please provide more information or context about what you are referring to.",
    'hi': "मुझे खेद है, लेकिन आपके प्रश्न का विशिष्ट उत्तर देने के लिए मेरे पास पर्याप्त जानकारी नहीं है। आप जिस चीज़ का उल्लेख कर रहे हैं उसके बारे में कृपया अधिक जानकारी या संदर्भ प्रदान करें।",
    'kn': "ನನ್ನನ್ನು ಕ್ಷಮಿಸಿ, ಆದರೆ ನಿಮ್ಮ ಪ್ರಶ್ನೆಗೆ ನಿರ್ದಿಷ್ಟ ಉತ್ತರವನ್ನು ಒದಗಿಸಲು ನನ್ನ ಬಳಿ ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ. ದಯವಿಟ್ಟು ನೀವು ಯಾವುದನ್ನು ಉಲ್ಲೇಖಿಸುತ್ತಿದ್ದೀರಿ ಎಂಬುದರ ಕುರಿತು ಹೆಚ್ಚಿನ ಮಾಹಿತಿ ಅಥವಾ ಸಂದರ್ಭವನ್ನು ಒದಗಿಸಿ."
}

def langchain_indexing(uuid_number):
    sources = SimpleDirectoryReader(uuid_number, recursive=True).load_data()
    source_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=4 * 1024, chunk_overlap=200)
    counter = 0
    for source in sources:
        for chunk in splitter.split_text(source.text):
            new_metadata = {"source": str(counter)}
            source_chunks.append(Document(page_content=chunk, metadata=new_metadata))
            counter += 1
    try:
        search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())
        search_index.save_local("")
        error_message = None
        status_code = 200
    except openai.error.RateLimitError as e:
        error_message = f"OpenAI API request exceeded rate limit: {e}"
        status_code = 500
    except (openai.error.APIError, openai.error.ServiceUnavailableError):
        error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
        status_code = 503
    except Exception as e:
        error_message = str(e.__context__) + " and " + e.__str__()
        status_code = 500
    return error_message, status_code


def rephrased_question(user_query):
    template = """
    Write the same question as user input and make it more descriptive without adding new information and without making the facts incorrect.

    User: {question}
    Rephrased User input:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=False)
    response = llm_chain.predict(question=user_query)
    return response.strip()


def querying_with_langchain(uuid_number, query):
    files_count = read_langchain_index_files(uuid_number)
    if files_count == 2:
        try:
            search_index = FAISS.load_local(uuid_number, OpenAIEmbeddings())
            chain = load_qa_with_sources_chain(
                OpenAI(temperature=0), chain_type="map_reduce"
            )
            paraphrased_query = rephrased_question(query)
            documents = search_index.similarity_search(paraphrased_query, k=5)
            answer = chain(
                {"input_documents": documents, "question": paraphrased_query}
            )
            answer_list = answer["output_text"].split("\nSOURCES:")
            final_answer = answer_list[0].strip()
            source_ids = answer_list[1]
            source_ids = source_ids.replace(" ", "")
            source_ids = source_ids.replace(".", "")
            source_ids = source_ids.split(",")
            final_source_text = ""
            for document in documents:
                if document.metadata["source"] in source_ids:
                    final_source_text += document.page_content + "\n\n"
            shutil.rmtree(uuid_number)
            return final_answer, final_source_text, paraphrased_query, None, 200
        except openai.error.RateLimitError as e:
            error_message = f"OpenAI API request exceeded rate limit: {e}"
            status_code = 500
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
            status_code = 503
        except Exception as e:
            error_message = str(e.__context__) + " and " + e.__str__()
            status_code = 500
    else:
        error_message = "The UUID number is incorrect"
        status_code = 422
    return None, None, None, error_message, status_code


def querying_with_langchain_gpt4(uuid_number, query):
    if uuid_number.lower() == "storybot":
        try:
            system_rules = "I want you to act as an Indian story teller. You will come up with entertaining stories that are engaging, imaginative and captivating for children in India. It can be fairy tales, educational stories or any other type of stories which has the potential to capture children’s attention and imagination. A story should not be more than 200 words. The audience for the stories do not speak English natively. So use very simple English with short and simple sentences, no complex or compound sentences. Extra points if the story ends with an unexpected twist."
            res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": query},
                ],
            )
            return res["choices"][0]["message"]["content"], "", "", None, 200
        except openai.error.RateLimitError as e:
            error_message = f"OpenAI API request exceeded rate limit: {e}"
            status_code = 500
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
            status_code = 503
        except Exception as e:
            error_message = str(e.__context__) + " and " + e.__str__()
            status_code = 500
        return None, None, None, error_message, status_code
    else:
        files_count = read_langchain_index_files(uuid_number)
        if files_count == 2:
            try:
                search_index = FAISS.load_local(uuid_number, OpenAIEmbeddings())
                documents = search_index.similarity_search(query, k=5)
                contexts = [document.page_content for document in documents]
                augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query
                system_rules = "You are a helpful assistant who helps with answering questions based on the provided information. If the information cannot be found in the text provided, you admit that I don't know"

                res = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_rules},
                        {"role": "user", "content": augmented_query},
                    ],
                )
                return res["choices"][0]["message"]["content"], "", "", None, 200

            except openai.error.RateLimitError as e:
                error_message = f"OpenAI API request exceeded rate limit: {e}"
                status_code = 500
            except (openai.error.APIError, openai.error.ServiceUnavailableError):
                error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
                status_code = 503
            except Exception as e:
                error_message = str(e.__context__) + " and " + e.__str__()
                status_code = 500
        else:
            error_message = "The UUID number is incorrect"
            status_code = 422
        return None, None, None, error_message, status_code

def querying_with_langchain_gpt4_streaming(uuid_number, query):
    files_count = read_langchain_index_files(uuid_number)
    if files_count == 2:
        try:
            search_index = FAISS.load_local(uuid_number, OpenAIEmbeddings())
            documents = search_index.similarity_search(query, k=5)
            contexts = [document.page_content for document in documents]
            augmented_query = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n" + query

            system_rules = "You are a helpful assistant who helps with answering questions based on the provided information. If the information cannot be found in the text provided, you admit that I don't know"
        
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": augmented_query}
                ],
                stream=True
            )

            # Define a generator function to yield each chunk of the response
            async def generate_messages():
                for chunk in response:
                    print(chunk)
                    # chunk_message = chunk['choices'][0]['delta']['content']
                    # chunk_message = chunk["choices"][0].get("delta", {}).get("content", '')
                    chunk_message = chunk["choices"][0].get("delta", {}).get("content", '')
                    yield chunk_message

            # Return a StreamingResponse with the generated messages
            return EventSourceResponse(generate_messages(), headers={"Content-Type":"text/plain"})
            # application/json

        except openai.error.RateLimitError as e:
            error_message = f"OpenAI API request exceeded rate limit: {e}"
            status_code = 500
            logger.exception("RateLimitError occurred: %s", e)
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
            status_code = 503
            logger.exception("APIError or ServiceUnavailableError occurred")
        except Exception as e:
            error_message = str(e.__context__) + " and " + e.__str__()
            status_code = 500
            logger.exception("An exception occurred: %s", e)
    else:
        error_message = "The UUID number is incorrect"
        status_code = 422

    # return None, None, None, error_message, status_codewss
    # If there's an error, return a plain text response with the error message
    return Response(content=error_message, media_type="text/plain", status_code=status_code)

def querying_with_langchain_gpt4_mcq(uuid_number, query, doCache):
    if uuid_number.lower() == "tech":
        try:
            logger.info('************** Technology Specific **************')
            system_rules = getSystemRulesForTechQuestions() 
            prompts = getPromptsForGCP(doCache, query, system_rules, promptsInMemoryTechQues)
            logger.info(prompts)  
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages = promptsInMemoryTechQues if doCache else prompts,
            )
            respMsg = res["choices"][0]["message"]["content"]
            logger.info(respMsg)
            if doCache:
                promptsInMemoryTechQues.append({"role":"assistant", "content":respMsg})
            logger.info('************** Questions **************')
            logger.info(respMsg)  
            return respMsg, "", "", None, 200
        except openai.error.RateLimitError as e:
            error_message = f"OpenAI API request exceeded rate limit: {e}"
            status_code = 500
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
            status_code = 503
        except Exception as e:
            error_message = str(e.__context__) + " and " + e.__str__()
            status_code = 500
        return None, None, None, error_message, status_code
    else:
        logger.info('************** Domain Specific **************')
        files_count = read_langchain_index_files(uuid_number)
        if files_count == 2:
            try:
                search_index = FAISS.load_local(uuid_number, OpenAIEmbeddings())
                documents = search_index.similarity_search(query, k=5)
                contexts = [document.page_content for document in documents]

                system_rules = getSystemRulesForDomainSpecificQuestions()
                context = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"
                system_rules = system_rules.format(Context=context)

                prompts = getPromptsForGCP(doCache, query, system_rules,  promptsInMemoryDomainQues)
                logger.info(prompts)
                start_time = time.time()
                res = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-16k",
                    messages = promptsInMemoryDomainQues if doCache else prompts,
                )
                end_time = time.time() - start_time
                logger.info(f"********* TOTAL TIME TOOK **********>>>>> {end_time}")
                respMsg = res["choices"][0]["message"]["content"]
                logger.info('************** Questions **************')
                logger.info(respMsg)    
                if doCache:
                    promptsInMemoryDomainQues.append({"role":"assistant", "content":respMsg})

                csvOutout = jsnoDifferenceData(uuid_number, respMsg) # JSON based duplication solution
                # csvOutout = csvDifferenceData(uuid_number, respMsg) # CSV based duplication solution
                logger.info('---- Filtered Questions-----')
                logger.info(csvOutout)
                return csvOutout, "", "", None, 200

            except openai.error.RateLimitError as e:
                error_message = f"OpenAI API request exceeded rate limit: {e}"
                status_code = 500
            except (openai.error.APIError, openai.error.ServiceUnavailableError):
                error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
                status_code = 503
            except Exception as e:
                # error_message = str(e.__context__) + " and " + e.__str__()
                error_message = e.__str__()
                status_code = 500
        else:
            error_message = "The UUID number is incorrect"
            status_code = 422
        return None, None, None, error_message, status_code

def querying_with_langchain_gpt3(uuid_number, query, converse: bool, language = default_language):
    print("query ====>", query)
    load_dotenv()
    files_count = read_langchain_index_files(uuid_number)
    if files_count == 2:
        try:
            search_index = FAISS.load_local(uuid_number, OpenAIEmbeddings())
            documents = search_index.similarity_search_with_score(query, k=5)
            # contexts = [document.page_content for document in documents]
            score_threshold = score_language_mapping[language]
            contexts =  [document.page_content for document, search_score in documents if round(search_score, 2) <= score_threshold]
            print(str(documents))
            if not contexts:
                return search_default_msg[language], None, None, None, 200
            
            if not converse:
                return "", documents, None, None, 200

            contexts = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"
            system_rules = """You are embodying "Sakhi for Jaadui Pitara", an simple AI assistant specially programmed to help kids navigate the stories and learning materials from the ages 3 to 8. Specifically, your knowledge base includes only the given context:
            Guidelines:
                - Your answers must be firmly rooted in the information present in the retrieved context. Ensure that your responses are directly based on these resources, not on prior knowledge or assumptions.
                - If no contexts are retrieved, then you should not answer the question.
            
            Given the following contexts:                
            {context}

            All answers should be in MARKDOWN (.md) Format:"""
            

            system_rules = system_rules.format(context=contexts)
            # print("system_rules ====> ",  system_rules)
            openai.api_key = os.environ["OPENAI_API_KEY"]
            res = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": system_rules},
                    {"role": "user", "content": query},
                ],
            )
            response = res["choices"][0]["message"]["content"]
            # print("response ====> ", response)
            # f = open("response.txt", "w")
            # f.write(str(response))
            # f.close()
            return response, documents, None, None, 200

        except openai.error.RateLimitError as e:
            error_message = f"OpenAI API request exceeded rate limit: {e}"
            status_code = 500
        except (openai.error.APIError, openai.error.ServiceUnavailableError):
            error_message = "Server is overloaded or unable to answer your request at the moment. Please try again later"
            status_code = 503
        except Exception as e:
            error_message = str(e.__context__) + " and " + e.__str__()
            status_code = 500
    else:
        error_message = "The UUID number is incorrect"
        status_code = 422
    return "", None, None, error_message, status_code

def get_source_markdown(documents, language = default_language) -> str:
    score_threshold = score_language_mapping[language]
    sources =  [document.metadata for document, search_score in documents if round(search_score, 2) <= score_threshold]
    added_sources = []
    sources_markdown = f'\n\n{source_default_msg[language]} \n\n'
    counter = 1
    for data in sources:   
        if not data["identifier"] in added_sources:
            print(data["name"])
            sources_markdown = sources_markdown + f'''{counter}. [{data["name"]}]({data["artifactUrl"]}) \n\n'''
            added_sources.append(data["identifier"])
            counter += 1

    return sources_markdown

# User feedback
async def record_user_feedback(engine, qa_id, feedback_type):
    try:
       async with engine.acquire() as connection:
            record_exists = await connection.fetchval("SELECT id FROM sb_qa_logs WHERE question_id = $1", qa_id)
            if record_exists is not None:
                if feedback_type.lower() == "up":
                    await connection.execute("UPDATE sb_qa_logs SET upvotes = upvotes + 1 WHERE question_id = $1", qa_id)
                elif feedback_type.lower() == "down":
                    await connection.execute("UPDATE sb_qa_logs SET downvotes = downvotes + 1 WHERE question_id = $1", qa_id)
                return 'OK', None, 200
            else:
                 return None, f"Record with ID {qa_id} not found", 404
    except Exception as e:
        error_message = str(e.__context__) + " and " + e.__str__()
        status_code = 500
        print(f"Error while giving feedback: {e}")
        return None, error_message, status_code

def create_directory_from_filepath(filepath):
    directory_path = os.path.dirname(filepath)
    
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Load existing data from JSON file
def load_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    
# Compare and add unique data
def add_unique_data(existing_data, new_data):
    seen_objects = set(tuple(item.items()) for item in existing_data)
    unique_data = []

    for item in new_data:
        obj = tuple(item.items())
        if obj not in seen_objects:
            unique_data.append(item)
            seen_objects.add(obj)

    return unique_data

# Compare and remove duplicate data
def remove_duplicates(data):
    seen_objects = set(tuple(item.items()) for item in data)
    return list(map(lambda t : dict((key,value) for key, value in t), seen_objects))    

# Save data to JSON file
def save_json_file(filename, data):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Convert data to CSV format
def list_to_csv_string(data):
    output = StringIO()
    csv_writer = csv.DictWriter(output, fieldnames=data[0].keys())

    csv_writer.writeheader()
    csv_writer.writerows(data)

    csv_string = output.getvalue()
    output.close()

    return csv_string

def jsnoDifferenceData(uuid_number: str, questions: str) -> str: 
    output_file_path = f"questions_cache/{uuid_number}.json"
    create_directory_from_filepath(output_file_path)
    try:
        parsed_data = json.loads(questions)
    except:
        raise Exception("Apologies! I couldn't create questions in a format that's easy to read for you. Please try again.")
        
    new_questions = remove_duplicates(parsed_data)
    existing_questions = load_json_file(output_file_path)
    unique_data = []
    if(len(existing_questions) == 0):
        save_json_file(output_file_path, new_questions)
        unique_data = new_questions
    else:
        unique_data = add_unique_data(existing_questions, new_questions)
        existing_questions += unique_data
        save_json_file(output_file_path, existing_questions)
    
    logger.info("Data has been updated in the JSON file and return as CSV format.")
    return list_to_csv_string(unique_data)


def removeWhitespace(text:str) -> list[str]:
    return list(map(lambda l : l.strip(),
                    filter(lambda l : l != '',
                            text.split('\n'))))

def string_compare_diff(text1: list[str], text2: list[str]) -> list[str]:
                        result: list[str] = []
                        for line1 in text1:
                            if line1 not in text2:
                                result.append(line1)
                        return result

def csvDifferenceData(uuid_number: str, respMsg: str) -> str: 
    output_file_path = f"questions_cache/{uuid_number}.csv"
    create_directory_from_filepath(output_file_path)
    new_questions = removeWhitespace(respMsg)[1:]
    new_questions = list(set(new_questions))
    if os.path.exists(output_file_path):
        old_question_file = open(output_file_path, 'r')
        old_questions = removeWhitespace(old_question_file.read())
        output = string_compare_diff(new_questions, old_questions)
        with open(output_file_path, "a") as file:
            file.write("\n")
            for item in output:
                file.write(item + "\n")
        csv_string = 'question, option_a, option_b, option_c, option_d, correct_answer \n'
        if output:
            csv_string += '\n'.join(output)
        return csv_string
    else:
        csv_string = 'question, option_a, option_b, option_c, option_d, correct_answer \n'
        csv_string += '\n'.join(new_questions)
        # Write the strings to the file
        with open(output_file_path, mode='w') as output_file:
                output_file.write(csv_string)            
        return csv_string

def getSystemRulesForTechQuestions():
    system_rules = """
                    You are a technology expert tasked with creating multiple-choice questions for a question bank. Your goal is to provide the question, options, and correct answer. Make sure that questions are not repeated.
    
                    Please generate the questions and encode the responses in CSV format. Use the following headers in lowercase with spaces replaced by underscores: question, option_a, option_b, option_c, option_d, correct_answer. The output should be properly formatted and comma-separated.
                    
                    When generating the questions, list the options without prefixing them with option names like A, B, C, or D. However, specify the correct answer in the "correct_answer" column using the corresponding option letter.
                    
                    Example:
                    Question,Option_A,Option_B,Option_C,Option_D,Correct_Answer
                    What is the purpose of the sleep() method in Java?,To terminate a thread,To start a new thread,To pause the execution of a thread for a specific amount of time,To increase the priority of a thread,C
                    
                    Please generate the questions accordingly and provide the encoded CSV data.
                """
    return system_rules

def getSystemRulesForDomainSpecificQuestions():

    system_rules = """
                    As a domain expert, your task is to generate multiple-choice questions for a question bank based on a given context. 
                    The questions should be unique and not repeated. The correct answers should be shuffled among the answer options randomly for each question.

                    Given the context:

                    "{Context}"

                    Here are the specific instructions that you need to follow:
                        - Do not provide answers or information that is not explicitly mentioned in the given context. Stick only to the facts provided.
                        - The questions and answer options should be encoded in JSON format. The JSON object array will consist of the following fields:
                            - question: The text of the question.
                            - option_a: The first answer option.
                            - option_b: The second answer option.
                            - option_c: The third answer option.
                            - option_d: The fourth answer option.
                            - correct_answer: The correct answer index as A, B, C, or D.
                        
                    Please generate the questions accordingly and Provide the questions only in JSON object array format, without any other responses.
                """

    return system_rules


# def setSystemRules(promptType, contexts):
#     if promptType == "getSystemRulesForDomainSpecificQuestions":
#         system_rules = getSystemRulesForDomainSpecificQuestions()
#         context = "\n\n---\n\n".join(contexts) + "\n\n-----\n\n"
#         system_rules = system_rules.format(Context=context)
#         return system_rules 
#     else:
#         system_rules = getSystemRulesForTechQuestions()
#         return system_rules 

def getPromptsForGCP(doCache, query, system_rules, prompts):
    
    userContent = {"role": "user", "content": query}
    systemContent = {"role": "system", "content": system_rules}
    if doCache:
        if len(prompts) == 0:
            prompts.append(systemContent)
            prompts.append(userContent)
        else:
            prompts.append(userContent)
        return prompts
    else:
        singlePrompt = [
            systemContent,
            userContent
        ]
        return singlePrompt


