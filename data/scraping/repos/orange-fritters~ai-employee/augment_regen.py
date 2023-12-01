"""
Some data is missing from the chatgpt_460.json file,
dut to OpenAI API's server error.
This script is to find and compensate the missing data.
"""

import openai
import os
import json
import tiktoken
import random


def count_tokens(text: str):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def get_response_openai(prompt: str,
                        document: str):
    prompt_tokens, generated_tokens = 0, 0
    SYSTEM1 = prompt
    ASSISTANT1 = document
    USER1 = """
        Generate five different scenarios you recommending which service might fit the counselee based on the situation.
        Situation is in conversational tone.
        Do not mention who is talking.
        """
    prompt_tokens += count_tokens(SYSTEM1)
    prompt_tokens += count_tokens(ASSISTANT1)
    prompt_tokens += count_tokens(USER1)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "assistant",
                 "content": ASSISTANT1},
                {"role": "user",
                 "content": USER1
                 },
                {"role": "system",
                 "content": SYSTEM1},
            ],
            presence_penalty=0.6
        )
    except:
        print("OpenAI Response Error")
        return None, None, None

    generated_tokens += count_tokens(response['choices'][0]['message']['content'])

    SYSTEM2 = """
            You are a translator from English to Korean.Translated tone must be conversational.
            Do not translate the answers.
            Just translate the situations of counsellee only.
            You generate all text in Korean.
            Drop out the answers from counselor.
            You generate all text in Korean.
            Translation only include the situations of counsellee nothing else.

            Form of the output is:
            1. Translated text
            2. Translated text
            3. Translated text
            4. Translated text
            5. Translated text
            """
    ASSISTANT2 = f"{response['choices'][0]['message']['content']}"
    USER2 = """
            Drop out the answer from counselor.
            Extract only the situations from the Counsellee and translate them into Korean.
            Tone must be conversational.
            Number them 1, 2, 3, 4, 5.
            """

    prompt_tokens += count_tokens(SYSTEM2)
    prompt_tokens += count_tokens(ASSISTANT2)
    prompt_tokens += count_tokens(USER2)

    try:
        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "assistant",
                    "content": ASSISTANT2},
                {"role": "user",
                 "content": USER2
                 },
                {"role": "system",
                    "content": SYSTEM2},
            ],
        )
    except:
        return None, None, None

    generated_tokens += count_tokens(second_response['choices'][0]['message']['content'])

    return second_response['choices'][0]['message']['content'], prompt_tokens, generated_tokens


def get_chat_response(doc_dir: str):
    PROMPT = """
        You are a counseler recommending the given service for who is asking for which welfare service might fit him.
        Counselee is ignorant to the given service, you are recommending the service for the first time.
        Situation explained by counsellee MUST NOT include the name of the service.
        Situation explained by counsellee include just the situation of the counselee, MUST NOT include the content of the service.
        Situation explained by counsellee is in conversational tone.
        Situation explained by counsellee ais in very easy word.
        Situation explained by counsellee is very short.
        Situation explained by counsellee is very general, never specific.
        Counsellee's job, gender, age, disability and other attributes are various.
        Question by counsellee is related to the situation of the counsellee, NOT about the service.
        Answer is extremely short, less than 5 words.
        Question's are short.
    """

    with open(doc_dir, 'r', encoding='utf-8') as f:
        document = f.read()

    questions_generated = get_response_openai(PROMPT, document)
    return questions_generated


if __name__ == '__main__':
    with open("server/config.json") as config_file:
        config_data = json.load(config_file)
        openai.api_key = config_data["chatgpt"]["secret"]

    with open("data/embeds/embeddings_notag.json") as file:
        embeddings = json.load(file)

    with open("data/augmented/generated_m.json") as f:
        data = json.load(f)

    # file_to_generate = [
    #     3, 10, 13, 14, 24, 65, 71, 76, 81, 95, 98, 110, 117, 134, 138, 160, 163, 165, 199, 208, 212, 215, 234, 236, 249,
    #     256, 264, 274, 289, 290, 300, 302, 309, 314, 322, 325, 326, 336, 338, 346, 383, 384, 388, 398, 400, 402, 417,
    #     420, 449, 460, 43, 44, 17, 34, 57, 78, 80, 83, 88, 92, 97, 113, 133, 137, 141, 156, 168, 169, 170, 175, 180,
    #     181, 186, 195, 204, 210, 242, 245, 247, 250, 251, 252, 253, 261, 280, 288, 301, 303, 324, 332, 335, 337, 339,
    #     340, 358, 359, 404, 408, 411, 412, 422, 429, 447, 450, 42, 47, 48, 392, 461]

    # file_to_reregenerate = [
    #     14, 137, 141, 186, 208, 242, 274, 388, 398, 402, 408, 420,
    # ]

    # file_to_rereregenerate = [
    #     388
    # ]

    files = [158, 184, 227, 248, 344]

    # 보건의료지원_03.html, 생계지원_28.html, 장애인지원_58.html, 388...

    input_dir = 'data/notags'
    for i, doc_dir in enumerate(sorted(os.listdir('data/notags'))):
        if i in files:
            file_dir = os.path.join(input_dir, doc_dir)
            data_tokens = embeddings[str(i)]['tokens']
            response, p_tok, g_tok = get_chat_response(file_dir)

            data[i] = {
                "filename": embeddings[str(i)]['filename'],
                "title": embeddings[str(i)]['title'],
                "index": i,
                "response": response,
            }
            print(i)
            print(response)
            print()

    with open(f'data/augmented/generated_m.json', 'w') as f:
        json.dump(data, f, indent=2)
