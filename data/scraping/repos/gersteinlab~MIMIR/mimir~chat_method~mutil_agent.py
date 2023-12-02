import os
import openai, json, random
import hashlib
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the path of the upper-level directory
upper_level_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(upper_level_dir)
from conf.config import *

def calculate_md5(input_list):
    # 将列表转换为字符串
    str_list = ''.join(map(str, input_list))
    # 创建一个 hashlib 对象
    md5_hash = hashlib.md5()
    # 更新哈希对象的内容
    md5_hash.update(str_list.encode('utf-8'))
    # 计算哈希值并返回
    return md5_hash.hexdigest()


# # 示例使用
# my_list = [1, 2, 3, 4, 5]
# md5_value = calculate_md5(my_list)
# print("MD5 哈希值：", md5_value)

def num_tokens_from_messages(messages, model):
    """Returns the number of tokens used by a list of messages."""
    import tiktoken

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        #         logger.debug("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo" or model == "gpt-35-turbo":
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        #         logger.warn(f"num_tokens_from_messages() is not implemented for model {model}. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


prompt_meta = '''### Instruction:
{}

### Response:'''


def generate(prompt, asure=True, temperature=0.1):
    if asure:
        key_bundles = configure["key_bundles"]
        try:
            output = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
        except openai.error.Timeout:
            print("User Timeout")
            key_bundle = random.choice(key_bundles)
            openai.api_key, openai.api_base = key_bundle
            output = openai.ChatCompletion.create(
                engine="gpt-35-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
        except openai.error.InvalidRequestError:
            print("User InvalidRequestError")
        except Exception as e:
            print(f"User An error occurred: {str(e)}")
    else:
        open_ai_api_key_list = configure["open_ai_api_key"]
        try:
            output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
        except openai.error.Timeout:
            print("User Timeout")
            key_bundle = random.choice(open_ai_api_key_list)
            openai.api_key = key_bundle
            output = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
            )
        except openai.error.InvalidRequestError:
            print("User InvalidRequestError")
        except Exception as e:
            print(f"User An error occurred: {str(e)}")
    out = output.choices[0].message["content"].replace('\n', ' ')
    if '### Response:' in out:
        out = out.split('### Response:')[1]
    if '### Instruction:' in out:
        out = out.split('### Instruction:')[0]
    return out



def mutil_agent(queue, progress, topic_list, index_list, picked_roles, role_prompt, MEMORY_LIMIT=10, asure=True, max_rounds=5,
                max_input_token=3000, user_temperature=0.1, ai_temperature=0.1):
    openai.api_key = None
    openai.api_key = ""
    if asure:
        openai.api_type = "azure"
        openai.api_base = configure["open_ai_api_base"]
        openai.api_version = "2023-05-15"
        openai.api_key = configure["open_ai_api_key"][0]  # get this API key from the resource (its not inside the OpenAI deployment portal)
        key_bundles = configure["key_bundles"]


    chat_content = {}

    def pick_elements_by_index(input_list, indices):
        new_list = [input_list[i] for i in indices]
        return new_list

    query_default = "what is your suggestions for breakfast, no eggs and milk!"
    if len(topic_list) == 0:
        topic_list.append(query_default)
    if len(index_list) != 0 and len(topic_list) > 1:
        topic_list = pick_elements_by_index(topic_list, index_list)

    step = round((100.0 / (len(topic_list) * max_rounds * len(picked_roles))) * 1.0 / 100, 2)
    ## topic 处理
    for query in topic_list:
        ## 轮次对话
        chat_content[query] = []
        memories = {}
        ideas = {}
        compressed_memories_all = {}

        ## 初始化记忆
        for i in picked_roles:
            memories[i] = []
        ## 压缩记忆
        for name in picked_roles:
            compressed_memories_all[name] = []
        ## 观点
        for name in picked_roles:
            ideas[i] = []
        ## 生成观点
        for name in picked_roles:
            prompt = "You are {}. {} You come to a chat room because you want to discuss the topic about {}. " \
                     "The following people are in this chat room: {}. What is your main point? Be brief, " \
                     "and use at most 20 words and answer from your perspective.".format(
                name, role_prompt[name], query,
                ', '.join(picked_roles))
            # print(prompt_meta.format(prompt))
            ideas[name] = generate(prompt_meta.format(prompt), asure, ai_temperature)
            # print("观点", name, ideas[name])

        say_prompts = {}
        ## 生成话术
        #             print("compressed_memories_all",compressed_memories_all)
        for name in picked_roles:
            prompt = "You are {}. {} You have ideas about topic {} is: {}. Your memories are: {}. " \
                     "The following people are in this chat room: {}. " \
                     "You can interact with them.".format(
                name,
                role_prompt[name],
                query,
                ideas[name],
                '\n'.join(compressed_memories_all[name][-5:]),
                ', '.join(picked_roles))
            people_description = []
            for i in picked_roles:
                people_description.append(i + ': ' + role_prompt[i])
            prompt += ' You know the following about people: ' + '. '.join(people_description)
            memory_text = '. '.join(memories[name][-10:])
            prompt += "Who do you want to say something to and what you want to say?"
            say_prompts[name] = prompt
        ## 处理话术为第一人称
        reps_results = {}
        for name in picked_roles:
            reps_results[name] = generate(prompt_meta.format(say_prompts[name]), asure, ai_temperature)
            # Now clean the action
            prompt = """
                        Convert the following paragraph to first person past tense, and make this text more like a conversation:
                        "{}"
                     """.format(reps_results[name])
            reps_results[name] = generate(prompt_meta.format(prompt), asure, ai_temperature).replace('"', '').replace("'", '')
            content = name + ":" + reps_results[name]
            chat_content[query].append(content)
            progress.value += step
            # print("话术", name, reps_results[name])

        ### 保存说话人的所有记忆
        for name in picked_roles:
            for name_two in picked_roles:
                memories[name].append('[Person is {}. Memory is {}]\n'
                                      .format(name_two, reps_results[name_two]))
        #             print("记忆", memories)
        ### 给记忆打分
        import re
        def get_rating(x):
            nums = [int(i) for i in re.findall(r'\d+', x)]
            if len(nums) > 0:
                return min(nums)
            else:
                return None

        memory_ratings = {}
        for name in picked_roles:
            memory_ratings[name] = []
            for i, memory in enumerate(memories[name]):
                prompt = "You are {}. Your ideas are: {}. You are currently in a chat room and you are talk about {}. " \
                         "You observe the following: {}. Give a rating, between 1 and 5, to how much you care about this. Keep the memory that who mentioned you and what he say!".format(
                    name, ideas[name], query, memory)
                res = generate(prompt_meta.format(prompt), asure, ai_temperature)
                rating = get_rating(res)
                max_attempts = 2
                current_attempt = 0
                while rating is None and current_attempt < max_attempts:
                    rating = get_rating(res)
                    current_attempt += 1
                if rating is None:
                    rating = 0
                memory_ratings[name].append((res, rating))
        # print("mR", name, memory_ratings[name])

        ### 记忆压缩
        compressed_memories = {}
        for name in picked_roles:
            memories_sorted = sorted(
                memory_ratings[name],
                key=lambda x: x[1]
            )[::-1]
            relevant_memories = memories_sorted[:MEMORY_LIMIT]
            # print(name, relevant_memories)
            memory_string_to_compress = '.'.join([a[0] for a in relevant_memories])
            prompt = "You are {}. Your ideas are: {}. You are currently in a chat room talking about {}. You observe the following: {}. Summarize these memories in one sentence.".format(
                name, ideas[name], query, memory_string_to_compress)
            res = generate(prompt_meta.format(prompt), asure, ai_temperature)
            compressed_memories[name] = '[Recollection {}: {}]'.format(name, res)
            compressed_memories_all[name].append(compressed_memories[name])

        for repeats in range(max_rounds - 1):
            say_prompts = {}
            ## 生成话术
            #             print("compressed_memories_all",compressed_memories_all)
            for name in picked_roles:
                prompt = "You are {}. {} You with the following guys {} are talking about the topic {}, you have ideas is: {}." \
                         "You have the memories is {}. " \
                         "If someone mentioned you in the before conversation answer him, if no one mentioned you, " \
                         " You can say something relevant to the topic. End the conversation if you have no words to say! So you will say:".format(
                    name,
                    role_prompt[name],
                    ', '.join(picked_roles),
                    query,
                    ideas[name],
                    '\n'.join(compressed_memories_all[name][-5:]))
                #                 people_description = []
                #                 for i in picked_roles:
                #                     people_description.append(i + ' who is a person that ' + role_prompt[i])
                #                 prompt += ' You know the following about people with their description is : ' + '. '.join(people_description)
                #                 memory_text = '. '.join(memories[name][-10:])
                #                 prompt += "Who do you want to say something to and what you want to say?"
                say_prompts[name] = prompt
                # print("say prompt", say_prompts[name])
            ## 处理话术为第一人称
            reps_results = {}
            for name in picked_roles:
                reps_results[name] = generate(prompt_meta.format(say_prompts[name]), asure, ai_temperature)
                # print("话术0 ", name, reps_results[name])
                # Now clean the action
                prompt = """
                            Convert the following paragraph to first person past tense, and make this text more like a conversation:
                            "{}"
                         """.format(reps_results[name])
                reps_results[name] = generate(prompt_meta.format(prompt), asure, ai_temperature).replace('"', '').replace("'", '')
                content = name + ":" + reps_results[name]
                chat_content[query].append(content)
                progress.value += step
                # print("话术", name, reps_results[name])

            ### 保存说话人的所有记忆
            for name in picked_roles:
                for name_two in picked_roles:
                    memories[name].append('[Person {}. Memory is {}]\n'
                                          .format(name_two, reps_results[name_two]))
            #             print("记忆", memories)
            ### 给记忆打分
            import re
            def get_rating(x):
                nums = [int(i) for i in re.findall(r'\d+', x)]
                if len(nums) > 0:
                    return min(nums)
                else:
                    return None

            memory_ratings = {}
            for name in picked_roles:
                memory_ratings[name] = []
                for i, memory in enumerate(memories[name]):
                    prompt = "You are {}. Your ideas are: {}. You are currently in a chat room and you are talk about {}. " \
                             "You observe the following: {}. Give a rating, between 1 and 5, to how much you care about this. ".format(
                        name, ideas[name], query, memory)
                    res = generate(prompt_meta.format(prompt), asure, ai_temperature)
                    rating = get_rating(res)
                    max_attempts = 2
                    current_attempt = 0
                    while rating is None and current_attempt < max_attempts:
                        rating = get_rating(res)
                        current_attempt += 1
                    if rating is None:
                        rating = 0
                    memory_ratings[name].append((res, rating))
            #                 print("mR", name, memory_ratings[name])
            ### 记忆压缩
            compressed_memories = {}
            for name in picked_roles:
                memories_sorted = sorted(
                    memory_ratings[name],
                    key=lambda x: x[1]
                )[::-1]
                relevant_memories = memories_sorted[:MEMORY_LIMIT]
                # print(name, relevant_memories)
                memory_string_to_compress = '.'.join([a[0] for a in relevant_memories])
                prompt = "You are {}. Your ideas are: {}. You are currently in a chat room talking about {}. You observe the following: {}. Summarize these memories in one sentence.".format(
                    name, ideas[name], query, memory_string_to_compress)
                res = generate(prompt_meta.format(prompt), asure, ai_temperature)
                compressed_memories[name] = '[Recollection {} : {}]'.format(name, res)
                compressed_memories_all[name].append(compressed_memories[name])
    # print(chat_content)
    queue.put(chat_content)
    return chat_content
# if total_tokens >= max_tokens:
#     break
# if len(chat_content) % 100 == 0:
#     print("total_tokens: {}, examples: {}".format(total_tokens, len(chat_content)))
#     pkl.dump(
#         chat_content,
#         open("collected_data/{}_chat_{}.pkl".format(data_name, index), "wb"),
#     )

# pkl.dump(
#     chat_content, open("collected_data/{}_chat_{}.pkl".format(data_name, index), "wb")
# )
