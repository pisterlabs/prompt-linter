import django
django.setup()
from sefaria.model.text import Ref, library

import re
import langchain
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

from functools import reduce
from util.general import get_raw_ref_text
import typer
from tqdm import tqdm
import csv


def get_topics_for_title(title: str, lang: str):
    index = library.get_index(title)
    rows = []
    for segment_oref in tqdm(index.all_section_refs()[:20]):
        print('-----')
        print(segment_oref.normal())
        topics = get_topics_for_tref(segment_oref, lang)
        rows += [{"Ref": segment_oref.normal(), "Text": get_raw_ref_text(segment_oref, lang), "Topics": ", ".join(topics)}]
    with open("output/Pri Eitz Chaim Topics.csv", "w") as fout:
        cout = csv.DictWriter(fout, ['Ref', 'Text', "Topics"])
        cout.writeheader()
        cout.writerows(rows)



def get_topics_for_tref(oref: Ref, lang: str):
    text = get_raw_ref_text(oref, lang)
    return get_raw_topics(text, lang)


def get_raw_topics(text, lang):
    short_to_long_lang = {
        "he": "Hebrew", "en": "English"
    }
    examples_by_lang = {
        "he":
            "<topic>תרומה</topic>\n"
            "<topic>פרשת נח</topic>\n"
            "<topic>אברהם</topic>\n"
            "<topic>שבת</topic>\n",

        "en":
            "<topic>Teruma</topic>\n"
            "<topic>Parashat Noach</topic>\n"
            "<topic>Abraham</topic>\n"
            "<topic>Shabbat</topic>\n"

    }
    system_message = SystemMessage(content=
                                   "You are an intelligent Jewish scholar who is knowledgeable in all aspects of the Torah and Jewish texts.\n"
                                    "<task>\n"
                                    "Output list of high-level topics discussed by the input\n"
                                   "Topics should be important enough that they would warrant an entry in the index in the back of a book\n"
                                   "Each topic should be wrapped in <topic> tags\n"
                                   "Topics should be short. They should be written as if they are titles of encyclopedia entries. Therefore, they should be understandable when read independent of the source text.\n"
                                   "Citations are not topics. E.g. Genesis 1:4 is not a topic\n"
                                   "Topics should be written assuming a Torah context. Phrases like \"Torah perspective\", \"in Judaism\", \"in the Torah\" and \"Biblical Narrative\" should not appear in a topic.\n"
                                   f"Topics should be written in {short_to_long_lang[lang]}."
                                   "</task>"
                                   "<examples>\n"
                                   f"{examples_by_lang[lang]}"
                                   "</examples>\n"
                                   "<negative_examples>\n"
                                   "<topic>Dispute between Rabbi Akiva and Rabbi Yehoshua</topic>\n"
                                   "<topic>Opinions on how to shake lulav</topic>\n"
                                   "</negative_examples>"
                                   )
    user_prompt = PromptTemplate.from_template("# Input\n{text}")
    human_message = HumanMessage(content=user_prompt.format(text=text))

    # llm = ChatOpenAI(model="gpt-4", temperature=0)
    llm = ChatAnthropic(model="claude-2", temperature=0)

    response = llm([system_message, human_message])
    # print('---')
    # human_refine = HumanMessage(content="Of the topics above, list the most fundamental topics for understanding the source text. Exclude topics that are very specific.")
    # response2 = llm([system_message, human_message, response, human_refine])
    # human_breakup = HumanMessage(content="Of the topics above, break up complex topics into simpler topics.\n"
    #                                      "<examples>\n"
    #                                      "<topic>הלכות מזוזה בבית כנסת</topic> should become <topic>מזוזה</topic> and <topic>בית כנסה</topic>\n"
    #                                      "<topic>שאלה בדין תקיעת שופר ביום כיפור</topic> should become <topic>תקיעת שופר</topic> and <topic>יום כיפור</topic>\n"
    #                                      "<topic>הלכות עירוב</topic> should remain unchanged."
    #                                      "</examples>")
    #
    # response3 = llm([system_message, human_message, response, human_refine, response2, human_breakup])
    topics = reduce(lambda a, b: a + [b.group(1).strip()], re.finditer(r"<topic>(.+?)</topic>", response.content), [])
    return topics



if __name__ == '__main__':
    typer.run(get_topics_for_title)


