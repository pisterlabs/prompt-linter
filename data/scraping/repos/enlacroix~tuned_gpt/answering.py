import configparser
import traceback
import openai
from openai.embeddings_utils import cosine_similarity
import numpy as np
import pandas as pd

from add_embeddings import EmbeddingConverter
from utils import retry_on_error


config = configparser.ConfigParser()
config.read('config.ini')

openai.api_key = config['passwords']['openAIKey']
file = 'subgraphs_faq._question_embed.csv'
topic = 'Graphs and subgraphs'
# Functional analysis
GPTModel = 'gpt-3.5-turbo-16k'
num_top_qa = 3
start_message = 'Привет'


def handleEmbeds():
    df_qa = pd.read_csv(file)
    df_qa = df_qa[~df_qa['ada_embedding'].isna()]
    df_qa['ada_embedding'] = df_qa.ada_embedding.apply(eval).apply(np.array)
    return df_qa


@retry_on_error
def callChatGPT(*args, **kwargs):
    return openai.ChatCompletion.create(*args, **kwargs)


# def get_embedding(text, GPTModel="text-embedding-ada-002"):
#     text = text.replace("\n", " ")
#     return openai.Embedding.create(input=[text], GPTModel=GPTModel)['data'][0]['embedding']


def search_similar(df, product_description, n=3, pprint=True):
    """

    :param df:
    :param product_description:
    :param n:
    :param pprint:
    :return:
    """
    embedding = EmbeddingConverter.getEmbedding(product_description)
    df['similarities'] = df.ada_embedding.apply(lambda x: cosine_similarity(x, embedding))
    res = df.sort_values('similarities', ascending=False).head(n)
    return res


def collect_text_qa(df):
    text = ''
    for i, row in df.iterrows():
        text += f'Q: <' + row['Question'] + '>\nA: <' + row['Answer'] + '>\n\n'
    print('len qa', len(text.split(' ')))
    return text


def convertTextToTGFormat(text: str) -> list[str]:
    MAXMSGLEN = 4096

    if len(text) > MAXMSGLEN:
        parts = []
        while len(text) > MAXMSGLEN:
            parts.append(text[:MAXMSGLEN])
            text = text[MAXMSGLEN:]
        parts.append(text)
        return parts
    else:
        return [text]


def collectFullPromptForGPT(question, qa_prompt, chat_prompt=None):
    prompt = f'I need to get an answer to the question related to the topic of "{topic}": ' + "{{{" + question + "}}}. "
    prompt += '\n\nPossibly, you might find an answer in these Q&As [use the information only if it is actually relevant and useful for the question answering]: \n\n' \
              + qa_prompt
    # edit if you need to use this also
    if chat_prompt is not None:
        prompt += "---------\nIf you didn't find a clear answer in the Q&As, possibly, these talks from chats might be helpful to answer properly [use the information only if it is actually relevant and useful for the question answering]: \n\n" + chat_prompt
    prompt += f'\nFinally, only if the information above was not enough you can use your knowledge in the topic of "{topic}" to answer the question.'

    return prompt


def mainTask(update, context):
    user = update.effective_user
    context.bot.send_message(chat_id=user.id, text='Ваш запрос очень важен для нас, ждите ⏳...')

    try:
        question = update.message.text.strip()
    except Exception as e:
        context.bot.send_message(chat_id=user.id,
                                 text=f"🤔It seems like you're sending not text to the bot. Currently, the bot can only work with text requests.")
        return

    try:
        qa_found = search_similar(handleEmbeds(), question, n=num_top_qa)
        qa_prompt = collect_text_qa(qa_found)
        full_prompt = collectFullPromptForGPT(question, qa_prompt)
    except:
        print(traceback.format_exc())
        context.bot.send_message(chat_id=user.id,
                                 text=f"Поиск полностью провалился. В консоли выведены подробности возникшей ошибки.")
        return

    try:
        print(full_prompt)
        completion = callChatGPT(
            model=GPTModel,
            n=1,
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}]
        )
        result = completion['choices'][0]['message']['content']
    except:
        print(traceback.format_exc())
        context.bot.send_message(chat_id=user.id,
                                 text=f'Проблема на стороне сервиса OpenAI. Пробуйте ещё раз.')
        return

    parts = convertTextToTGFormat(result + full_prompt)
    for part in parts:
        update.message.reply_text(part, reply_to_message_id=update.message.message_id)
