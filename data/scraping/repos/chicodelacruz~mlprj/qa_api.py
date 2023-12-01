import os
import openai
import configuration
# add a configuration.py file with the line:
# key = "your api key"


class Answer:
    def __init__(self, answer="", additional_info="", next_question=""):
        answer = answer
        additional = additional_info
        follow_up = next_question


def create_jsonlfile():
    #Paste the API KEY
    #openai.api_key ="Your api key"
    openai.api_key = configuration.key
    # Create the documents file as jsonl file
    document_path = "jsonlfiles/finaldoc.jsonl"
    file = openai.File.create(file=open(document_path), purpose='answers')
    return file


def look_alternative_document(response_object):
    """
    Look for an alternative answer
    :param response_object:
    :return:
    """
    return "Would you like to browse all the handbook?"


def check_scores(user_question, response_object, score_threshold=0, low_threshold=50):
    """

    :param response_object:
    :param score_threshold:
    :param low_threshold: threshold for responses with low confidence
    :return:
    """
    answer_object = Answer()
    # go through response selected documents
    scores = []
    for document in response_object.selected_documents:
        # select max score
        scores.append(document.score)
    max_score = max(scores)
    print("max_score: {0}".format(str(max_score)))
    if max_score > score_threshold:
        # look for low confidence answers, it means gpt-3 generates an answer but the similarity to documents is low
        if max_score <= low_threshold:
            # adjust temperature, so far adjusting temperature still returns low scores
            # response = generateAnswers(user_question, temp=response_object.temperature + 1)
            print("low confidence")
            chatbot_response = look_alternative_document(response_object)
        else:
            # it could be the one with the maximum score but the one with higher score is not always on-point
            answer_object.answer = response_object.answers[0]
            # find document with top score
            answer_object.additional = response_object.selected_documents[0].text
            # but also include the documents text
    else:
        chatbot_response = "I don't understand the question"

    return answer_object


def generateAnswers(user_question,jsonl_file,temp = 0.1,maxtoken = 20):
   
   try:
    # Api for creating answers
    response =openai.Answer.create(
        search_model="ada", 
        model="davinci", 
        question=user_question,       
        
        file=jsonl_file["id"], 
        examples_context="Corruption is dishonest or illegal behavior, especially by people in power, typically involving bribery. It can also include other acts, such as fraud, embezzlement, favoritism, and nepotism. The most common form of corruption is bribery.For further information see Section G1 of the BCG.**Additional Information** : For further information, also about what the term gifts of money covers, see [Compliance Handbook](https://webbooks.siemens.com/public/LC/chen/index.htm?n=Part-1-Activity-Fields,A.-Anti-Corruption", 
        examples=[["Can I take my client on a holiday?", "No, you cannot take your client on a holiday .**Additional Information** For further information, see [Compliance Handbook](https://webbooks.siemens.com/public/LC/chen/index.htm?n=Part-1-Activity-Fields,A.-Anti-Corruption"],["What is corruption?", "Corruption is dishonest or illegal behavior, especially by people in power, typically involving bribery **Additional Information** For further information , see [Compliance Handbook](https://webbooks.siemens.com/public/LC/chen/index.htm?n=Part-1-Activity-Fields,A.-Anti-Corruption"],["What is bribery?","Bribery  is the act of offering, promising, or giving money, gifts, or other benefit to a public official or public or private employee with the aim of receiving improper advantages. Bribery is a criminal offense worldwide. Siemens does not tolerate any form of bribery. **Additional Information** For further information check [BCG](https://compliance.siemens.cloud/bcg/responsibility.html#g)"],["What are the rules for cash payments?","Payments with Cash are specifically regulated in many jurisdictions according to money laundering or other laws. The governance for Anti-Money Laundering lies with Legal & Compliance (LC CO RFC / LC CO SFS) and supports the BizCos by appropriate processes. **Additional Information** More information can be found [Here](https://webbooks.siemens.com/public/LC/chen/index.htm?n=Part-1-Activity-Fields,C.-Anti-Money-Laundering-(AML),5.-Cash-Handling-Rules)"],
        ["Was ist ein Geschenk?", "Ein Geschenk ist eine freiwillige Überweisung von Geld oder anderen Vorteilen an Dritte ohne Gegenleistung. ** Zusätzliche Informationen ** Weitere Informationen finden Sie im [Compliance-Handbuch](https://webbooks.siemens.com/public/LC/chen/index.htm?n=Part-1-Activity-Fields,A.-Anti-Corruption"]],
        max_rerank=10,
        max_tokens=maxtoken,
        temperature=temp,
        stop=["\n"]
    )

    return response
   
   except:
       response ={"answers": ["Apologies, I could not find an answer for your query. Please ask questions related to"
                              " compliance or please rephrase your question"],
                  "file": file}
       return response


print("Creating file !")
file =create_jsonlfile() 
print("File created!! File id: ", file["id"])

user_ques =input("Chatbot - Enter your question :")
response = generateAnswers(user_ques, file)
full_answer = check_scores(user_ques, response)
# print("Chatbot Answer :", response["answers"][0])
print("Chatbot Answer :", full_answer.answer)
if full_answer.additional:
    print("Additionally:\n")
    print(full_answer.additional)
