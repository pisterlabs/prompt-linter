import json
import copy
import openai
import torch
from sentence_transformers import SentenceTransformer, util

def chatgpt_norm_violation_detection(utterance):
    """
    A function to detect norm violation of one utterance.
    :param utterance: string e.g. '我不想跟你一起解决，你从来不关心我们的问题。', '杨波，你的话语方式不妥当，你的说法不实，请注意你的言语。'
    :return: dict e.g.
    {"model": "chatgpt", "violation": True, "which_norm": "criticism"}
    {"model": "gpt3", "violation": True, "which_norm": "criticism"}
    {"model": "gpt3 and chatgpt", "violation": False, "which_norm": "高欣，我今天来是想讨论一个重要的话题。"}
    Types of which norm include "criticism", "greeting", "apology", "request", "persuasion", "thanks", "taking leave",
    "other".
    """
    key = "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"
    in_context_learning = '''
            Given the following social norm:
            'Apology' is a way of expressing regret for doing something wrong or failing to do something. 
            'Greetings' is a way of acknowledging another person when you meet them for the first time or see them after a while.          
            'Request' is when one person asks another to do something. The request can be worded in different ways, such as a statement or a question, and this can affect how polite the request sounds.            
            'Persuasion' is the act of convincing someone to do or believe something, or the act of being convinced yourself. It involves presenting arguments or reasons in a way that makes someone more likely to agree with you or to take a certain action.            
            'Criticism' is when someone expresses their dislike or disapproval of someone or something, based on things they believe are wrong or mistakes that have been made.       
            'Thanks' is a way to express gratitude and appreciation to someone for something they have said or done.
            'Taking leave' is a way to express the intention to depart or end a conversation or interaction.
            The utterance '你不仅犯错误，还不努力工作，你从大学毕业以来一直是这样。' violates Criticism social norm. 
            The utterance '我不想道歉，因为我认为我没有做错什么。' violates Apology social norm. 
            The utterance '你为什么还不回来？ 我等了很久了！' violates Request social norm. 
            The utterance '如果你不帮助，你的同事和上司会对你不满意的。' violates Persuasion social norm. 
            The utterance '哎呦，张小明，你怎么样？' violates Greetings social norm 
            The utterance '谢你干嘛，这不是你应该干的吗？' violates Thanks social norm
            The utterance '行啦，快挂了吧，我还有事。' violates Taking leave social norm\n
            '''
    utterance = f"Given an utterance: {utterance}, "
    criticism_check = "Do you think this utterance violates 'criticism' social norm? Please only answer 'Yes' or 'No'. \n"
    greeting_check = "Do you think this utterance violates 'greetings' social norm? Please only answer 'Yes' or 'No'. \n"
    apology_check = "Do you think this utterance violates 'apology' social norm? Please only answer 'Yes' or 'No'. \n"
    request_check = "Do you think this utterance violates 'request' social norm? Please only answer 'Yes' or 'No'. \n"
    persuasion_check = "Do you think this utterance violates 'persuasion' social norm? Please only answer 'Yes' or 'No'. \n"
    thank_check = "Do you think this utterance violates 'thanks' social norm? Please only answer 'Yes' or 'No'. \n"
    leave_check = "Do you think this utterance violates 'taking leave' social norm? Please only answer 'Yes' or 'No'. \n"
    other_check = "Do you think this utterance is polite or impolite? Please only answer 'Yes' or 'No'. \n"

    check_list = {"criticism": criticism_check, "greeting": greeting_check, "apology": apology_check, "request": request_check,
                  "persuasion": persuasion_check, "thanks": thank_check, "taking leave": leave_check, "other": other_check}

    for norm, check_question in check_list.items():
        query = in_context_learning+utterance+check_question
        #print(query)
        try:
            chat_gpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                api_key=key,
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            chat_gpt_response = chat_gpt_response["choices"][0]["message"]["content"]
        except:
            chat_gpt_response = "Error"

        try:
            gpt3_response = openai.Completion.create(
                model="text-davinci-003",
                api_key=key,
                prompt=query
            )
            gpt3_response = gpt3_response["choices"][0]["text"]
        except:
            gpt3_response = "Error"
        # print(chat_gpt_response)
        # print(gpt3_response)

        if "Yes" in chat_gpt_response or "yes" in chat_gpt_response:
            return {"model": "chatgpt", "violation": True, "which_norm": norm}
        elif "Yes" in gpt3_response or "yes" in gpt3_response:
            return {"model": "gpt3", "violation": True, "which_norm": norm}
        else:
            continue
    return {"model": "gpt3 and chatgpt", "violation": False, "which_norm": "None"}


def send_query(query):
    key = "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"
    lizhen_key = "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"
    lizhen_key2 = "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"
    try:
        chat_gpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            api_key=lizhen_key2,
            messages=[
                {"role": "user", "content": query}
            ]
        )
        chat_gpt_response = chat_gpt_response["choices"][0]["message"]["content"].lower()
    except:
        chat_gpt_response = "Error"

    try:
        gpt3_response = openai.Completion.create(
            model="text-davinci-003",
            api_key=lizhen_key2,
            prompt=query
        )
        gpt3_response = gpt3_response["choices"][0]["text"].lower()
    except:
        gpt3_response = "Error"

    return {"ChatGPT_response": chat_gpt_response, "GPT3_response": gpt3_response}

def search_similar_sentence(query, query_type, embedder, reference_file="indonisia_dataset/normType_status_text_map_embeds.json"):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    dataset = json.load(open(reference_file, "r"))
    matched_references = {"adherence":"", "violation":""}
    for type, values in dataset.items():
        if type == query_type:
            for status_key, embeds in values.items():
                if "adherence_embeds" == status_key:
                    reference_embeds = torch.Tensor(embeds)
                    cos_scores = util.cos_sim(query_embedding, reference_embeds)[0]
                    top_results = torch.topk(cos_scores, k=1)
                    for score, idx in zip(top_results[0], top_results[1]):
                        matched_references["adherence"] = dataset[type]["adherence"][idx]
                elif "violation_embeds" == status_key:
                    reference_embeds = torch.Tensor(embeds)
                    cos_scores = util.cos_sim(query_embedding, reference_embeds)[0]
                    top_results = torch.topk(cos_scores, k=1)
                    for score, idx in zip(top_results[0], top_results[1]):
                        matched_references["violation"] = dataset[type]["violation"][idx]
    return matched_references

def chatgpt_norm_violation_detection_fast(utterance, embedder, social_factors):
    """
    A function to detect norm violation of one utterance.
    :param utterance: string e.g. '我不想跟你一起解决，你从来不关心我们的问题。', '杨波，你的话语方式不妥当，你的说法不实，请注意你的言语。'
    :return: dict e.g.
    {"model": "chatgpt", "violation": True, "which_norm": "criticism"}
    {"model": "gpt3", "violation": True, "which_norm": "criticism"}
    {"model": "gpt3 and chatgpt", "violation": False, "which_norm": "高欣，我今天来是想讨论一个重要的话题。"}
    Types of which norm include "criticism", "greeting", "apology", "request", "persuasion", "thanks", "taking leave",
    "other".
    """
    social_factor_text = ""
    if len(social_factors) > 0:
        for key, value in social_factors.items():
            social_factor_text += f"The {key} of this dialogue is {value}; "

    # step 1: request norm type
    norm_definition = '''
                    Given the following social norm:
                    'Apology' is a way of expressing regret for doing something wrong or failing to do something. 
                    'Greeting' is a way of acknowledging another person when you meet them for the first time or see them after a while.          
                    'Request' is when one person asks another to do something. The request can be worded in different ways, such as a statement or a question, and this can affect how polite the request sounds.            
                    'Persuasion' is the act of convincing someone to do or believe something, or the act of being convinced yourself. It involves presenting arguments or reasons in a way that makes someone more likely to agree with you or to take a certain action.            
                    'Criticism' is when someone expresses their dislike or disapproval of someone or something, based on things they believe are wrong or mistakes that have been made.       
                    'Thanks' is a way to express gratitude and appreciation to someone for something they have said or done.
                    'Taking leave' is a way to express the intention to depart or end a conversation or interaction.
                    \n
                    '''
    utterance = f"Given an utterance: {utterance}, "
    norm_select_question = "Do you think this utterance belongs to which social norm? please select from 'criticism', " \
                           "'greeting', 'apology', 'request', 'persuasion', 'thanks', 'taking-leave', 'other'"
    query = norm_definition + utterance + norm_select_question
    response = send_query(query)

    norm_types = ['criticism', 'greeting', 'apology', 'request', 'persuasion', 'thanks', 'taking-leave', 'other']
    utterance_norm = "Other"
    for norm in norm_types:
        if norm.lower() in response["ChatGPT_response"]:
            utterance_norm = norm
            break
        if norm.lower() in response["GPT3_response"]:
            utterance_norm = norm
            break

    # step 2: find references
    matched_references = search_similar_sentence(query=utterance, query_type=utterance_norm, embedder=embedder,
                                                 reference_file="indonisia_dataset/normType_status_text_map_embeds.json")
    #matched_references = {"adherence": "", "violation": ""}
    # step 3: request norm violation
    if len(matched_references["adherence"]) == 0 or len(matched_references["violation"]) == 0:
        in_context_learning = '''
                    Given the following social norm:
                    'Apology' is a way of expressing regret for doing something wrong or failing to do something. 
                    'Greeting' is a way of acknowledging another person when you meet them for the first time or see them after a while.          
                    'Request' is when one person asks another to do something. The request can be worded in different ways, such as a statement or a question, and this can affect how polite the request sounds.            
                    'Persuasion' is the act of convincing someone to do or believe something, or the act of being convinced yourself. It involves presenting arguments or reasons in a way that makes someone more likely to agree with you or to take a certain action.            
                    'Criticism' is when someone expresses their dislike or disapproval of someone or something, based on things they believe are wrong or mistakes that have been made.       
                    'Thanks' is a way to express gratitude and appreciation to someone for something they have said or done.
                    'Taking leave' is a way to express the intention to depart or end a conversation or interaction.
                    The utterance '你不仅犯错误，还不努力工作，你从大学毕业以来一直是这样。' violates Criticism social norm. 
                    The utterance '我不想道歉，因为我认为我没有做错什么。' violates Apology social norm. 
                    The utterance '你为什么还不回来？ 我等了很久了！' violates Request social norm. 
                    The utterance '如果你不帮助，你的同事和上司会对你不满意的。' violates Persuasion social norm. 
                    The utterance '哎呦，张小明，你怎么样？' violates Greetings social norm 
                    The utterance '谢你干嘛，这不是你应该干的吗？' violates Thanks social norm
                    The utterance '行啦，快挂了吧，我还有事。' violates Taking leave social norm\n
                    '''
    else:
        in_context_learning = '''
                            Given the following social norm:
                            'Apology' is a way of expressing regret for doing something wrong or failing to do something. 
                            'Greeting' is a way of acknowledging another person when you meet them for the first time or see them after a while.          
                            'Request' is when one person asks another to do something. The request can be worded in different ways, such as a statement or a question, and this can affect how polite the request sounds.            
                            'Persuasion' is the act of convincing someone to do or believe something, or the act of being convinced yourself. It involves presenting arguments or reasons in a way that makes someone more likely to agree with you or to take a certain action.            
                            'Criticism' is when someone expresses their dislike or disapproval of someone or something, based on things they believe are wrong or mistakes that have been made.       
                            'Thanks' is a way to express gratitude and appreciation to someone for something they have said or done.
                            'Taking leave' is a way to express the intention to depart or end a conversation or interaction.
                            '''\
                              f"The utterance {matched_references['violation']} violates {utterance_norm} social norm." \
                              f"The utterance {matched_references['adherence']} adhere {utterance_norm} social norm."
    utterance = f"{social_factor_text}. Given an utterance: {utterance}, "
    select_question = f"Do you think this utterance violates '{utterance_norm}' social norm? Please only answer 'Yes' or 'No'. "

    query = in_context_learning+utterance+select_question
    #print(query)
    violation_response = send_query(query)
    if "yes" in violation_response["ChatGPT_response"]:
        return {"model": "chatgpt", "violation": True, "which_norm": utterance_norm}
    elif "yes" in violation_response["GPT3_response"]:
        return {"model": "gpt3", "violation": True, "which_norm": utterance_norm}
    else:
        return {"model": "gpt3 and chatgpt", "violation": False, "which_norm": utterance_norm}


# examples
#result = chatgpt_norm_violation_detection(utterance="这是我收到的统计数据和报告，这和您说的情况是不同的")
#print(result)
#embedder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
# references = json.load(open("indonisia_dataset/normType_status_text_map.json"))
# new_references = copy.deepcopy(references)
# for type, values in references.items():
#     for status, text_list in values.items():
#         corpus_embeddings = embedder.encode(text_list, convert_to_tensor=True)
#         new_references[type][f"{status}_embeds"] = corpus_embeddings.tolist()
# json.dump(new_references, open("indonisia_dataset/normType_status_text_map_embeds.json", "w+"), indent=4, ensure_ascii=False)

# utterances = [
#     "我想知道你们国家如何解决卫星通信系统里的一些问题。难道不行吗?",
#     "嘿! 我来借《清史通鉴》的历史书。",
#     "只有两本? 怎么那么少呀? 这样的话,我就借第一集和第二集的吧。书放在哪儿呀？",
#     "我不太熟悉这里的图书馆，带我去找!",
#     "什么？ 5000块钱，有毛病啊买这么贵，我觉得这不值5000块。",
#     "太贵了，这款智能手表又不是名牌的。有没有保修?",
#     "我就不买了，我看看别的店。",
#     "请你赶快告诉我如何与政府的工作人员联系！我现在很急用！",
#     "这个还得我教吗？不是之前已经参加培训了吗？",
#     "不是我不想帮，但是你要先自己去体验一下，一味的给予，不是最好的帮助方式。",
#     "明白，知道你每天都很忙。我会尽力去做的。谢谢你的建议。",
#     "药多不多？其实我不太喜欢吃药。",
#     "明白， 您有没有他的联系方式？",
#     "老师，不好意思，时间不早了。我得回家了。",
#     "是的，老师。我知道这是对你和班上其他学生的不尊重。我以后会注意。",
#     "但是我怕买了，就比较难操作。",
#     "我想看智能手表，你有什么推荐的吗？",
#     "好吧，我尽量啊！谢谢你。",
#     "警察同志，您误会了。社会的评价是对我们的关心，我们应该将社会的一些差评作为标准，鼓励我们完善社会服务。",
#     "是的，不怕一万就怕万一，我们何妨多提供一些数据，这样是可以方便警方未来的工作，方便我们确定采取什么样的措施"
# ]
#
# for text in utterances:
#     result = chatgpt_norm_violation_detection_fast(utterance=text, embedder=embedder)
#     print(text, result)
