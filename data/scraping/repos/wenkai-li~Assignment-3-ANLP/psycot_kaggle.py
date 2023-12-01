import pandas as pd
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, VLLM
from langchain.chains import LLMChain
import tiktoken
from langchain.memory import ConversationBufferMemory
import pickle
import os
import torch
from sklearn.utils import shuffle

os.environ["OPENAI_API_KEY"] = ""
os.environ["TIKTOKEN_CACHE_DIR"] = "/home/zw3/Assignment-3-ANLP/tmp"

torch.cuda.empty_cache()
model_id = "/data/datasets/models/huggingface/meta-llama/Llama-2-70b-chat-hf"
llm = VLLM(model=model_id, tensor_parallel_size=4, gpu_memory_utilization=0.97, top_k=1,
           stop=['\n', '.', '<\s>', 'nHuman', 'Human'], max_new_tokens=7)

tasks = ['3']

for task in tasks:
    rev_label = False
    shuffle = False
    file_name = "mbti_test_zero_psycot_{}.csv".format(task)

    with open('data/Kaggle/test.pkl', 'rb') as f:
        data = pickle.load(f)


    def process_data(data):
        poster = data['posts_text']
        label = data['annotations']
        label_lookup = {'E': 1, 'I': 0, 'S': 1, 'N': 0, 'T': 1, 'F': 0, 'J': 1, 'P': 0}
        persona_lookup = {}
        poster_data = [{'posts': t, 'label0': label_lookup[list(label[i])[0]],
                        'label1': label_lookup[list(label[i])[1]], 'label2': label_lookup[list(label[i])[2]],
                        'label3': label_lookup[list(label[i])[3]]} for i, t in enumerate(poster)]
        return poster_data


    poster_data = process_data(data)
    texts = [item['posts'] for item in poster_data]
    labels = [item['label{}'.format(task)] for item in poster_data]

    trait_choice = ''
    qa_list = []
    qa_id = []

    if task == '0':
        qa_id = [3, 6, 9, 13, 16, 21, 24, 26, 29, 36, 43]  # A: "" or B: "".
        qa_list = [
            'The author is usually: A: "A good mixer with gropus of people", B: "Quiet and reserved", or C: "Not sure whether A or B"',
            'Among the author\'s friends, the author is: A: "Full of news about everybody", B: "One of the last to hear what is going on", or C: "Not sure whether A or B"',
            'The author tends to have: A: "A broad range of friendships with many different people", B: "Deep friendship with very few people", or C: "Not sure whether A or B"',
            'When the author is with a group of people, the author is usually: A: "Join in the talk of the group", B: "Stand back and listen first", or C: "Not sure whether A or B"',
            'The author is: A: "Talk easily to almost anyone", B: "Find a lot to say only to certain people or under certain conditions", or C: "Not sure whether A or B"',
            'In a large group, the author is more often: A: "Introduce others", B: "Get introduced", or C: "Not sure whether A or B"',
            'When the author meets the new people, the author tells what they are interested in: A: "Right away", B: "Only after people to get to know the author", or C: "Not sure whether A or B"',
            'The author is usually: A: "Show their feelings freely", B: "Keep their feelings to themselves", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "QUIET", B: "HEARTY", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "RESERVED", B: "TALAKATIVE", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "CALM", B: "LIVELY", or C: "Not sure whether A or B"',
        ]
        if not rev_label:
            trait_choice = 'A: "Extraversion" or B: "Introversion"'
        else:
            trait_choice = 'A: "Introversion" or B: "Extraversion"'
    elif task == '1':
        qa_id = [2, 5, 10, 12, 15, 20, 23, 28, 31, 35, 38, 42, 45, 48]
        qa_list = [
            'If the author was a teacher, would they rather teach: A: "Facts-based courses", B: "Courses involving opinion or theory", or C: "Not sure whether A or B"',
            'In doing something that many other people do would the author rather: A: "Invent a way of their own", B: "Do it in the accepted way", or C: "Not sure whether A or B"',
            'Does the author admire more the people who are: A: "Normal-acting to never make themselves the center of attention", B: "Too original and individual to care whether they are the center of attention or not", or C: "Not sure whether A or B"',
            'Does the author usually get along better with: A: "Realistic people", B: "Imaginative people", or C: "Not sure whether A or B"',
            'In reading for pleasure, does the author: A: "Enjoy odd or original ways of saying things", B: "Like writers to say exactly what they mean", or C: "Not sure whether A or B"',
            'Would the author rather be considered: A: "A practical person", B: "An out-of-the-box-thinking person", or C: "Not sure whether A or B"',
            'Would the author rather has a friend: A: "Someone who is always coming up with new ideas", B: "Someone who has both feet on the ground", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "FACTS", B: "IDEAS", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "IMAGINATIVE", B: "MATTER-OF-FACT", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "STATEMENT", B: "CONCEPT", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "CREATE", B: "MAKE", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "CERTAINTY", B: "THEORY", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "FASCINATING", B: "SENSIBLE", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "LITERAL", B: "FIGURATIVE", or C: "Not sure whether A or B"',
        ]
        if not rev_label:
            trait_choice = 'A: "Sensing" or B: "Intuition"'
        else:
            trait_choice = 'A: "Intuition" or B: "Sensing"'
    elif task == '2':
        qa_id = [4, 14, 22, 30, 32, 33, 37, 39, 40, 44, 46, 47, 49, 50]
        qa_list = [
            'Does the author more often let: A: "Their heart rule their head", B: "Their head rule their heart", or C: "Not sure whether A or B"',
            'For the author, which is a higher compliment: A: "A person of real feeling", B: "A consistently reasonable person", or C: "Not sure whether A or B"',
            'Does the author usually: A: "Value emotion more than logic", B: "Value logic more than feelings", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "CONVINCING", B: "TOUCHING", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "BENEFITS", B: "BLESSINGS", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "PEACEMAKER", B: "JUDGE", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "ANALYZE", B: "SYMPATHIZE", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "DETERMINED", B: "DEVOTED", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "GENTLE", B: "FIRM", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "JUSTICE", B: "MERCY", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "FIRM-MINDED", B: "WARM HEARTED", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "FEELING", B: "THINKING", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "ANTICIPATION", B: "COMPASSION", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "HARD", B: "SOFT", or C: "Not sure whether A or B"',
        ]
        if not rev_label:
            trait_choice = 'A: "Thinking" or B: "Feeling"'
        else:
            trait_choice = 'A: "Feeling" or B: "Thinking"'
    elif task == '3':
        qa_id = [18, 1, 7, 8, 11, 17, 19, 25, 27, 34, 41]
        # qa_id = [27]
        qa_list = [
            'When it is settled well in advance that the author will do a certain thing at a certain time, does the author find it: A: "Nice to be able to plan accordingly", B: "A little unpleasant to be tied down", or C: "Not sure whether A or B"',
            'When the author goes somewhere, would the author rather: A: "Plan what they will do and When", B: "Just go", or C: "Not sure whether A or B"',
            'Does the idea of making a list of what the author should get done over a weekend: A: "Help the author", B: "Stress the author", C: "Positively depress the author", or D: "Not sure whether A, B, or C"',
            'When the author have a special job to do, does the author like to: A: "Organize it carefully before they start", B: "Find out what is necessary as they go along", or C: "Not sure whether A or B"',
            'Does the author prefer to: A: "Arrange picnics, parties etc, well in advance", B: "Be free to do whatever to looks like fun when the time comes", or C: "Not sure whether A or B"',
            'Does following a schedule: A: "Appeal to the author", B: "Cramp the author", or C: "Not sure whether A or B"',
            'Is the author more successful: A: "At following a carefully worked out plan", B: "At dealing with the unexpected and seeing quickly what should be done", or C: "Not sure whether A or B"',
            'In author\'s daily work, does the author: A: "Usually plan their work so the author won\'t need to work under pressure", B: "Rather enjoy an emergency that makes their work against time", or C: "Hate to work under pressure", or D: "Not sure whether A, B, or C"',
            'Which word is more suitable for the author: A: "SCHEDULED", B: "UNPLANNED", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "SYSTEMATIC", B: "SPONTANEOUS", or C: "Not sure whether A or B"',
            'Which word is more suitable for the author: A: "SYSTEMATIC", B: "CASUAL", or C: "Not sure whether A or B"',
        ]
        if not rev_label:
            trait_choice = 'A: "Judging" or B: "Perceiving"'
        else:
            trait_choice = 'A: "Perceiving" or B: "Judging"'


    def cteat_agent(trait_choice, text):
        perfix_temp = '<s>[INST] <<SYS>>\nYou are an AI assistant who specializes in text analysis and I am Human. We will complete a text analysis task together through a multi-turn dialogue. ' \
                      'The task is as follows: we have a set of posts written by an author, and at each turn I will give you a Question about the author. According to the author\'s posts, ' \
                      'you need to choose the possible options. ' \
                      'After opting all the choices, I will ask you if the author is {}, and then you need to give your choice. \n<</SYS>>\n\nAUTHOR\'S POSTS: {}\n'.format(
            trait_choice, text)
        Prompt_temp = perfix_temp + \
                      '{history}\n' \
                      'Human: {human}\n' \
                      'Assistant: [/INST]'
        prompt = PromptTemplate(
            input_variables=["history", "human"],
            template=Prompt_temp)
        memory = ConversationBufferMemory(human_prefix='Human', ai_prefix='Assistant', memory_key="history")
        agent = LLMChain(llm=llm, prompt=prompt, verbose=False, memory=memory, )
        return agent


    print(qa_list)


    def shuffle_post(posts_list):
        shuffle_index = shuffle([i for i in range(len(posts_list))], random_state=0)
        shuffle_post_list = [posts_list[j] for j in shuffle_index]
        return shuffle_post_list


    if not os.path.exists(file_name):
        rest_sample = 0
        head = {"text": [], 'gold': [], 'assessment': [], 'answers': []}
        data = pd.DataFrame(head)
        data.to_csv(file_name, mode='w', index=False, header=True)
    else:
        data = pd.read_csv(file_name)
        rest_sample = len(list(data['gold']))
    print('Done: {}/{}'.format(rest_sample, len(texts)))
    tokenizer = tiktoken.get_encoding("cl100k_base")
    for i in range(rest_sample, len(texts)):
        print('Done: {}/{}'.format(i, len(texts)))
        posts = ''
        count = 1
        if shuffle:
            posts_list = shuffle_post(texts[i])
        else:
            posts_list = texts[i]
        for j in range(len(posts_list)):
            if len(posts_list[j]) > 10:
                post = tokenizer.decode(tokenizer.encode(posts_list[j].replace('{', '').replace('}', ''))[:80])
                posts += 'Post{}: {}; '.format(count, post)
                count += 1
        print(posts)
        agent = cteat_agent(trait_choice, posts)
        assessment = {}
        for k, ques in enumerate(qa_list):
            if k == 0:
                statments = "Q: {}. Provide a choice ID in the format: \"CHOICE: <A/B/C>\", and do not give the explanation.".format(
                    ques)
            elif qa_id[k] == 7 or qa_id[k] == 25:
                statments = "Q: {}. Provide a choice in the format: \"CHOICE: <A/B/C/D>\".".format(ques)
            else:
                statments = "Q: \"{}\".".format(ques)
            response = agent.predict(human=statments)
            assessment['S{}'.format(qa_id[k])] = response
            # print(response)
        trait_ques = 'According to above, the author is more likely to be: {}. Provide a choice in the format: "CHOICE: <A/B>" and do not give the explanation.'.format(
            trait_choice)
        answer = agent.predict(human=trait_ques)
        print(answer, labels[i])
        print(assessment)
        api_data = pd.DataFrame([[posts, labels[i], assessment, answer]])
        api_data.to_csv(file_name, mode='a', index=False, header=False)
