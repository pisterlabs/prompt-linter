#from selectors import EpollSelector#
#from lib2to3.pytree import _Results
#from selectors import EpollSelector
from PineconeEngine import PineconeSearch
import openai
import json
import re
import time
from Researcher.LinkSearch import LinkSearch  # Google Search
from Researcher.crossRef_Search import CrossRef_Search  # Essay Bank Search
openai.api_key = "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"


def Research(chat, task):    
    Researcher = task # decide if it uses essay database only. 

    msg_topicIdentification = "I have a question and this is my problem: {problem}. I am now going to do an fact check. Design a short, concise and accurate query for me to perform the fact check for more relevant google search result. In your response you should only state the new query and nothing else.".format(problem=task)
    topic, token = chat.Chat(msg_topicIdentification)
    print(">>> topic identified: ", topic)

    if Researcher:
        result = CrossRef_Search(topic, 30, 10000) # topic (query), number of webpages, token budget
    else:
        result = LinkSearch(topic, 30, 30, 10000)
    ## and now you can use your result for what ever reason


def OpenaiHandshake(chain):
    response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=chain,
            temperature=0,
            presence_penalty=1
        )
    response_m = response.choices[0].message
    return (response_m, response_m["content"])

def AccessDatabase(query, depth):
    input_article = []
    results = PineconeSearch(query, depth)
    processed_results_dict = [{"doi":i["metadata"]["doi"], "abstract":i["metadata"]["abstract"]} for i in results["matches"]]
    return processed_results_dict

def article_read(topic, directions, articles, basic_log):
    topic_prompt = "The user conducts research of topic {}, and wants to focus on {} as his research directions. Here is the article {}"
    kick_off_prompt = "Please response"
    for i in range(len(articles)):
        basic_log.append({"role": "user", "content": topic_prompt.format(topic, directions, articles[i]["abstract"])})
        basic_log.append({"role": "user", "content": kick_off_prompt})
        assistant, content = OpenaiHandshake(basic_log)
        articles[i]["comment"] = content
        basic_log.pop()
        basic_log.pop()
    print(articles)
    return articles

def search_semantic_scholar(query, search_depth):
    results = []
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&fields=url,abstract,authors,externalIds&offset=0&limit={str(search_depth)}"
    response = requests.get(url)
    search_results = response.json()
    if "data" in search_results:
        for article in search_results["data"]:
            results.append({"abstract":article["abstract"]})
        return results
    else:
        return "false"
   

#Input searched articles and query and output a dict of format {"doi":, "abstract":, "comment":}#
def query_block(query, query_topic, query_log, search_depth=5):
    while True:
        user_choice = input("Please state your preferred searching tools: database, semantic scholar, or Jason search:\n ")
        if user_choice == "database":
            searched_results = AccessDatabase(query_topic, search_depth)
            break
        elif user_choice == "semantic scholar":
            searched_results = search_semantic_scholar(query_topic, search_depth)
            break
        elif user_choice == "Jason search":
            True #edit#

        else:
            print("Please type correctly!")
    if searched_results == "false":
        print("try different question!")
        return True
    else:
        kick_off_query_prompt = {"role":"user","content":"Please start your response"}
        for i in range(len(searched_results)):
            query_log.append({"role":"user", "content":query})
            query_log.append({"role":"user", "content":searched_results[i]["abstract"]})
            query_log.append(kick_off_query_prompt)
            assistant, answer = OpenaiHandshake(query_log)
            query_log.pop()
            query_log.pop()
            searched_results[i]["comment"] = answer
        return  searched_results

#Output a dict of format {"doi":, "abstract":, "suggestion":}#
def brain_storm_block(brain_storm_log, topic, search_depth=1):
    while True: 
        user_choice = input("Please state your preferred searching tools: database or semantic scholar:\n ")
        if user_choice == "database":
            searched_articles = AccessDatabase(topic, search_depth)
            break 
        elif user_choice == "semantic scholar":
            searched_articles = search_semantic_scholar(topic, search_depth)
            break
        else:
            print("please type correctly!")
    kick_off_brain_storm_prompt = {"role":"user","content":"Please start your response"}
    if searched_articles == "false":
        print("suggest write topic differently")
        return True
    else:
        for i in range(len(searched_articles)):
            brain_storm_log.append({"role":"user", "content":searched_articles[i]["abstract"]})
            brain_storm_log.append(kick_off_brain_storm_prompt)
            assistant, suggestion = OpenaiHandshake(brain_storm_log) 
            brain_storm_log.pop()
            brain_storm_log.pop()
            searched_articles[i]["suggestion"] = suggestion
        return searched_articles

def Improving_Block(returned_feedback, article_structure):
    print("----------------------Start changing your outline!---------------------------")
    print("Your received feedback")
    for i in range(len(article_structure)):
        article_structure[i]["content"] = eval(article_structure[i]["content"])
    num_feedback = len(returned_feedback)
    for i in range(num_feedback):
        print("{}th feedback--------------------------".format(i+1))
        print(returned_feedback[i])
    while True:
        print("Your current outline")
        num_article_structure = len(article_structure)
        for i in range(num_article_structure):
            print("{}th major part-----------------------------".format(i+1))
            print(article_structure[i])
        user_act = input("""Please type ("delete", "add","f"(once complete)): """)
        if user_act=="delete":
            while True:
                target_major_point = int(input("Which major point, just type in the order(1,2,3,etc): "))
                user_act_del = input("""Do you want to delete this whole major point, or just delete one of the keypoint? type in "major point" or "key point":\n """)
                if user_act_del=="major point":
                    article_structure.pop(target_major_point-1)
                elif user_act_del=="key point":
                    while True:
                        user_act_del_name = input("Please input the name of the key point you want to delete, just paste it:\n ")
                        article_structure[target_major_point].pop(user_act_del_name)
                        stop_sign = input("if you want to stop, type in f: ")
                        if stop_sign=="f":
                            break
                else:
                    print("Please type correctly! ")
                stop_sign = input("if you finish deleting, type f: ")
                if stop_sign=="f":
                    break

        elif user_act=="add":
            while True:
                user_act_add = input("""Do you want to add a new major point, or just add some keypoint? type in "major point" or "key point":\n """)
                if user_act_add=="major point":
                    while True:
                        new_major_point = {"role":"user"}
                        new_major_point_name = input("Please type in the new major point name and its about briefly:\n")
                        user_act_add_key = input("Do you want to add key point or just leave it blank? if you want to add type add, or not type f: ")
                        if user_act_add_key=="add":
                            new_key_points = {}
                            while True:
                                new_key_point_name = input("Please type in the new key point name:\n ")
                                new_key_point_info = input("please type in some information about this key point:\n ")
                                new_key_points[new_key_point_name] = new_key_point_info
                            
                        elif user_act_add_key=="f":
                            new_key_points = {}
                        else: 
                            print("Please type in correctly!")
                        new_major_point["content"] = {"major point":new_major_point_name, "key points":new_key_points}
                        article_structure.append(new_major_point)
                        stop_sign = input("if you finish adding major point, type f: ")
                        if stop_sign=="f":
                            break
                elif user_act_add=="key point":
                    while True:
                        target_major_point = int(input("Please type in the order of major point the target key point is under: "))
                        while True:
                            new_key_point_name = input("Please type in the new key point name:\n ")
                            new_key_point_info = input("please type in some information about this key point:\n ")
                            article_structure[target_major_point-1]["content"]["key points"][new_key_point_name] = new_key_point_info
                            stop_sign = input("if you finish adding key points under this major point, type f: ")
                            if stop_sign=="f":
                                break
                        stop_sign = input("if you finish adding all ke points, type f: ")
                        if stop_sign=="f":
                            break
                else: 
                    print("Please type correctly!")

                stop_sign = input("if you finish adding, type f: ")
                if stop_sign=="f":
                    break
        elif user_act=="f":
            break
        else:
            print("Please type correctly!")
        
    for i in range(len(article_structure)):
        article_structure[i]["content"] = str(article_structure[i]["content"])
    return article_structure



brain_storm_prompt = """
                        You are an experienced science researcher and writer. Now a user is now preparing writing an essay about {}. He has written parts
                        of the outline, but there are still some gap inside each part that he can fit. Your job is to suggest some changes in the user's 
                        available outline. 
                        For instance, another user takes a research in explaining electrical conductivity in desolved NaCl, and sends you an outline in below dictionary format:
                        {{"1th major point": "What is the electrical conductivity?", "key points":{{"1th key point":"definition of conductivity", "2th key point":"definition of ionic bond"}}}}
                        {{"2th major point": "explain the mechanism of developing electrical conductivity", "key points":{{"1th key point":"when does NaCl has conductivity"}}}}

                        Later the user sends you an article: "We commonly think about salt as consisting of sodium chloride (NaCl) which is common, or table, salt.  But chemists define 
                        “salts” as any compound that, when dissolved in water, releases a positively charged cation and a negatively charged anion" 
                        Based on user sent outline and article, under each major point, you should evaluate each key point written by user on sent article structure one by one, and give suggestion on how to improve this outline. The suggestion can include adding related information to a key point,
                        delete the key point, or create a new key point. 

                        You must
                        output your answer in the same dictionary format as below 

                        Answer format(You are only allowed to write suggestions within(), everything else shouldn't be changed; also, please give suggestion for all of key points under any major point):
                        "{{"1th major point": "What is the electrical conductivity?", "key points":{{"1th key point":(suggestions about this key point of this major point), "2th key point":(suggestions about this key point of this major point)}}}} {{"2th major point": "explain the mechanism of developing electrical conductivity", "key points":{{"1th key point": (suggestions about this key point of this major point). "2th key point new created":(information or suggestions for created key point) }}}}"
                        
                        Answer example:
                        "{{"1th major point": "What is the electrical conductivity?", "key points":{{"1th key point":"based on this article, and your outline, I would suggest you focus on existence of free electron in writing conductivity definition.", "2th key point":"based on article and outline, I would suggest you delete it because it doesn't help"}}}} {{"2th major point": "explain the mechanism of developing electrical conductivity", "key points":{{"1th key point":"based on article and outline, you can write the ionic bond is broken by h-bond so that Na and Cl are separated. The formation of cations and anions will form electrical cnductivity", "2th key point(new created)":"you may add a key point about chemistry how they break" }}}}"   

                    
                    """

query_prompt =  """
                    You are an experitise in science research. Now a user is conductomg a research about {}. and the user encounters a problem.
                    The user has already gathered some information for his research and write parts of research outlines. Later he will send you 
                    details in his article, question he has, and some related articles for the question. Your job is to answer the researcher's 
                    question only based on information he sends to you.
                    For example, let's say another user conducts a research about studying electrical conductivity associated with NaCl.
                    The user first sends you the research outline of below dictionary format
                    {{"1th major point": "What is the electrical conductivity?", "key points":{{"1th key point":"definition of conductivity", "2th key point":"definition of ionic bond"}}}}
                    {{"2th major point": "explain the mechanism of developing electrical conductivity", "key points":{{"1th key point":"when does NaCl has conductivity"}}}}
                     
                    Later, the user sends you his question: in the 2th major point, I find it difficult to explain why desolved NaCl can conduct electricity.
                    Finally, the user sends you an related article: "We commonly think about salt as consisting of sodium chloride (NaCl) which is common, or table, salt.  But chemists define 
                    “salts” as any compound that, when dissolved in water, releases a positively charged cation and a negatively charged anion". These ions are what can conduct the electricity."

                    Then, based on information and article the user just sends to you, you should this answer in this format: according to the information you have sent to me, I would suggest
                    you in the 2th major point "explain the mechanism of developing electrical conductivity" focuse on the production of free ions. The article you just sent to me mention the role of free ions
                    in formation of electrical conductivity. Then, you can say as NaCl desolves into water, the ionic bond breaks and releases ions and thus lead to conductivity of electricity. 
                    If the article the user sends you dosen't contain any related information, then just say the article cannot help.  
                """


article_structure = []
query_log = []
brain_storm_log = []


print("Initialize article topic---------------------------------------------------------------------------------------------")
research_topic = input("Input your research topic: ")

print("Initializing query and brain storm blocks----------------------------------------------------------------------------")
query_log.append({"role":"system", "content":query_prompt.format(research_topic)})
brain_storm_log.append({"role":"system", "content":brain_storm_prompt.format(research_topic)})

print("Initialize your article structure -----------------------------------------------------------------------------------")
print("Please input your article major points.")
major_point_index = 1
while True:
    user_input_majorpoint = input("Please state main idea in the {}th major point; if you don't have yet, just type none: ".format(str(major_point_index)))
    if user_input_majorpoint == "none":
        break
    print("Create key points under the major part")
    key_points = {}
    info = []
    key_point_index = 1
    while True:
        user_input_keypoint_info = input("Please state detail in the {}th key point; if you don't have one yet, just type none:\n ".format(str(key_point_index)))
        if user_input_keypoint_info == "none":
            break
        key_points["{}th key point".format(str(key_point_index))] = user_input_keypoint_info
        key_point_index += 1
    #update article structure of a dict {"role":"user", "content":{"major point": , "key points":{1th key point:info1, 2th key point:info2}}}#
    article_structure.append({"role":"user", "content":str({"{}th major point".format(str(major_point_index)): user_input_majorpoint, "key points":key_points})})
    major_point_index += 1
    

print("-------------------------------Input initialized article structure into query block and brain storm block------------------------------------")
query_log += article_structure
brain_storm_log += article_structure
while True:
    user_input = input("""Please say what you needs next. You have three options: "ask question", "brain storm", and "done": """)
    if user_input=="done":
        break
    elif user_input=="ask question":
        user_question_query = input("Please write your question: ")
        user_question_location = input("Which major part that the question belongs to? just say first major point, second major point, so on (you can also type the first and sencond major points!):\n ")
        user_question = user_question_query+" "+user_question_location
        returned_answers = query_block(user_question,user_question_query, query_log)
        print("------------------------------------Start Improving Block----------------------------------------------------")
        article_structure = Improving_Block(returned_answers, article_structure)
        
    elif user_input=="brain storm":
        returned_suggestions = brain_storm_block(brain_storm_log, research_topic)
        print("------------------------------------Start Improving Block----------------------------------------------------")
        article_structure = Improving_Block(returned_suggestions, article_structure)
    else:
        print("""---------- Please only type three key phrases: "ask question", "brain storm", and "finish" """)
    print("Update your query log, and brain storm log----------------------------------------------------")
    query_log = []
    brain_storm_log = []
    query_log.append({"role":"system", "content":query_prompt.format(research_topic)})
    brain_storm_log.append({"role":"system", "content":brain_storm_prompt.format(research_topic)})
    query_log += article_structure
    brain_storm_log += article_structure








        
