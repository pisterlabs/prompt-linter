import re
import django
import anthropic
django.setup()
from sefaria.model import *
from sefaria.client.wrapper import get_links
from sefaria.datatype.jagged_array import JaggedTextArray
from util.openai import get_completion_openai, count_tokens_openai
from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage
from langchain.cache import SQLiteCache
import langchain
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def get_prompt(tref, topic_slug, commentary):
    topic_name, topic_description = get_topic_prompt(topic_slug)
    prompt = (
        f"# Input:\n"
        f"1) Commentary: commentary on the verse {tref}.\n"
        f"2) Topic: topic which relates to this verse\n"
        f"3) Topic Description: description of topic\n"
        f"# Task: Summarize the main points discussed by the commentators. Only include points that relate to the"
        f" topic \"{topic_name}\".\n"
        f"# Output: Numbered list of main points, only when relating to the topic \"{topic_name}\".\n"
        f"-----\n"
        f"# Input:\n1) Topic: {topic_name}\n2) Topic Description: {topic_description}\n3) Commentary: {commentary}"
    )
    return prompt


def get_topic_prompt(slug):
    topic = Topic.init(slug)
    return topic.get_primary_title('en'), getattr(topic, 'description', {}).get('en', '')


def get_commentary_for_tref(tref, max_tokens=7000):
    library.rebuild_toc()
    commentary_text = ""

    for link_dict in get_links(tref, with_text=True):
        if link_dict['category'] not in {'Commentary'}:
            continue
        if not link_dict['sourceHasEn']:
            continue
        link_text = JaggedTextArray(link_dict['text']).flatten_to_string()
        link_text = re.sub(r"<[^>]+>", " ", TextChunk.strip_itags(link_text))
        commentary_text += f"Source: {link_dict['sourceRef']}\n{link_text}\n"
        if count_tokens_openai(commentary_text) > max_tokens:
            break
    return commentary_text


def summarize_commentary(tref, topic_slug, company='openai'):
    commentary_text = get_commentary_for_tref(tref)
    prompt = get_prompt(tref, topic_slug, commentary_text)

    if company == 'openai':
        num_tokens = count_tokens_openai(prompt)
        print(f"Number of commentary tokens: {num_tokens}")
        completion = get_completion_openai(prompt)
    elif company == 'anthropic':
        llm = ChatAnthropic(model="claude-instant-1")
        completion = llm([HumanMessage(content=prompt)]).content
    else:
        raise Exception("No valid company passed. Options are 'openai' or 'anthropic'.")
    return completion


def print_summarized_commentary(tref, topic_slug):
    completion = summarize_commentary(tref, topic_slug)
    print(completion)


if __name__ == '__main__':
    print_summarized_commentary('Exodus 10:1-2', 'haggadah')
