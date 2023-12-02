"""
Given all the sources curated for a topic, determine what is unique about this source
"""
import json
import re
from functools import reduce
import django
django.setup()
from sefaria.model import *
from typing import List
from util.general import get_ref_text_with_fallback
from sheet_interface import get_topic_and_orefs

import langchain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.cache import SQLiteCache
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")


def _get_prompt_inputs(oref: Ref, other_orefs: List[Ref], topic: Topic):
    topic_title = topic.get_primary_title("en")
    topic_description = getattr(topic, "description", {}).get("en", "N/A")
    comparison_sources_list = []
    max_len = 7000
    for other_oref in other_orefs:
        other_text = get_ref_text_with_fallback(other_oref, "en", auto_translate=True)
        curr_len = reduce(lambda a, b: a + len(b), comparison_sources_list, 0)
        if curr_len + len(other_text) < max_len:
            comparison_sources_list += [other_text]
    return {
        "topic_title": topic_title,
        "topic_description": topic_description,
        "input_source": get_ref_text_with_fallback(oref, "en", auto_translate=True),
        "comparison_sources": json.dumps(comparison_sources_list)
    }

def _get_other_orefs_on_topic(oref: Ref, lang: str, topic: Topic) -> List[Ref]:
    ref_topics_links = topic.link_set("refTopic", {f"descriptions.{lang}": {"$exists": True}, "ref": {"$ne": oref.normal()}})
    other_orefs = []
    for link in ref_topics_links:
        try:
            other_orefs += [Ref(link.ref)]
        except:
            continue
    return other_orefs


def get_uniqueness_of_source(oref: Ref, lang: str, topic: Topic) -> str:
    other_orefs = _get_other_orefs_on_topic(oref, lang, topic)
    return _get_uniqueness_of_source_as_compared_to_other_sources(oref, other_orefs, topic)


def summarize_based_on_uniqueness(text: str, uniqueness: str) -> str:

    llm = ChatOpenAI(model="gpt-4", temperature=0)
    system_message = SystemMessage(content=
                                   "You are an intelligent Jewish scholar who is knowledgeable in all aspects of the Torah and Jewish texts.\n"
                                   "# Task\n"
                                   "Given a Jewish text and an idea mentioned in this text, write a summary of the text"
                                   " that focuses on this idea.\n" 
                                   "# Input format\n"
                                   "Input will be in XML format with the following structure:\n"
                                   "<text> text to be summarized according to idea </text>\n"
                                   "<idea> idea mentioned in the text </idea>\n"
                                   "# Output format\n"
                                   "A summary of the text that focuses on the idea, in 50 words or less.\n"
                                   "Wrap the summary in <summary> tags."
                                   "Summary should start with the words \"The text discusses...\""
                                   )
    prompt = PromptTemplate.from_template("<text>{text}</text>\n<idea>{idea}</idea>")
    human_message = HumanMessage(content=prompt.format(text=text, idea=uniqueness))
    response = llm([system_message, human_message])
    return re.search(r"<summary>\s*The text discusses (.+?)</summary>", response.content).group(1)


def _get_uniqueness_of_source_as_compared_to_other_sources(oref: Ref, other_orefs: List[Ref], topic: Topic) -> str:
    uniqueness_preamble = "The input source emphasizes"
    prompt_inputs = _get_prompt_inputs(oref, other_orefs, topic)
    system_message = SystemMessage(content=
                                   "You are an intelligent Jewish scholar who is knowledgeable in all aspects of the Torah and Jewish texts.\n"
                                   "# Task\n"
                                   "Given a list of Jewish texts about a certain topic, output the aspect that differentiates the input source from the other sources.\n"
                                   "# Input format\n"
                                   "Input will be in JSON format with the following structure\n"
                                   '{'
                                   '"topicTitle": "Title of the topic the sources are focusing on",'
                                   '"topicDescription": "Description of the topic",'
                                   '"inputSource": "Text of the source we want to differentiate from `comparisonSources`",'
                                   '"comparisonSources": "List of text of sources to compare `inputSource` to"'
                                   '}\n'
                                   "# Output format\n"
                                   "Output a summary that explains the aspect of `inputSource` that differentiates it "
                                   "from `comparisonSources`.\n"
                                   # "Summary should be no more than 20 words.\n"
                                   "Only mention the `inputSource`. Don't mention the `comparisonSources`.\n"
                                   f'Summary should complete the following sentence: "{uniqueness_preamble}...".'
                                   )
    prompt = PromptTemplate.from_template('{{{{'
                                          '"topicTitle": "{topic_title}", "topicDescription": "{topic_description}",'
                                          '"inputSource": "{input_source}", "comparisonSources": {comparison_sources}'
                                          '}}}}')
    human_message = HumanMessage(content=prompt.format(**prompt_inputs))
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    # llm = ChatAnthropic(model="claude-2", temperature=0)

    response = llm([system_message, human_message])
    uniqueness = re.sub(fr'^"?{uniqueness_preamble}\s*', '', response.content)
    uniqueness = re.sub(r'"$', '', uniqueness)
    return uniqueness


if __name__ == '__main__':
    # sheet_id = 498250
    # topic, orefs = get_topic_and_orefs(sheet_id)
    # for i in range(len(orefs)):
    #     oref = orefs[i]
    #     other_orefs = [r for j, r in enumerate(orefs) if j != i]
    #     print(get_uniqueness_of_source(oref, other_orefs, topic))
    uniqueness = "the enduring presence of the Divine in the Temple, even after its destruction."
    text = """[(Exod. 3:1:) <small>NOW MOSES WAS TENDING &lt;THE FLOCK&gt;</small>.] This text is related (to Ps. 11:4): <small>THE LORD IS IN HIS HOLY TEMPLE</small>….<sup class="footnote-marker">44</sup><i class="footnote">Cf. Hab. 2:20.</i> R. Samuel bar Nahman said: Before the destruction of the Sanctuary, the Divine Presence was situated in the Temple, as stated (Ps. 11:4): <small>THE LORD IS IN HIS HOLY TEMPLE</small>;<sup class="footnote-marker">45</sup><i class="footnote">Exod. R. 2:2; M. Pss. 11:3.</i> but, after the Temple was destroyed, (ibid. cont.:) <small>THE LORD'S THRONE IS IN THE HEAVENS</small>. He had removed his Divine Presence to the heavens. R. Eleazar ben Pedat said: Whether the Temple is destroyed or not destroyed, the Divine Presence has not moved from its place, as stated (in Ps. 11:4): <small>THE LORD IS IN HIS HOLY TEMPLE</small>. And where is it shown? Where it is stated (in I Kings 9:3): <small>MY EYES AND MY HEART SHALL BE THERE FOR ALL TIME</small>. It also says so (in Ps. 3:5 [4]): <small>I RAISE MY VOICE UNTO THE LORD, AND HE ANSWERS ME FROM HIS HOLY HILL. SELAH</small>. For even though it is &lt;only&gt; a hill,<sup class="footnote-marker">46</sup><i class="footnote"><i>Midrash Tanhuma</i> (Jerusalem: Eshkol: n.d.), vol. 1, appendix, p. 90, n. 2, suggests emending <i><small>HR</small></i> (“hill”) to <i><small>HRB</small></i> (“destroyed”) so that the clause would read in agreement with <i>Codex Vaticanus Ebr</i>. 34 and Exod. R. 2:2: “For even though it is destroyed.”</i> here he remains in his holiness. R. Eleazar ben Pedat said: See what is written (in Ezra 1:3): <small>AND LET HIM BUILD THE HOUSE OF THE LORD GOD OF ISRAEL. HE IS THE GOD WHO IS IN JERUSALEM</small>. He has not moved from there. R. Aha said: The Divine Presence has never moved from the West Wall (i.e., the Wailing Wall) of the Sanctuary. Thus it is stated (in Cant. 2:9): <small>THERE HE STANDS BEHIND OUR WALL</small>. Ergo (in Ps. 11:4): <small>THE LORD IS IN HIS HOLY TEMPLE</small>. R. Jannay said: Although they said (in Ps. 11:4): <small>THE LORD IS IN HIS HOLY TEMPLE; THE LORD HAS HIS THRONE IN THE HEAVENS</small>; &lt; nevertheless &gt; (the verse continues), <small>HIS EYES BEHOLD, HIS EYELIDS TEST THE CHILDREN OF ADAM</small>. To what is the matter comparable? To a king who had an orchard<sup class="footnote-marker">47</sup><i class="footnote"><i>Pardes</i>. Cf. the Gk.: <i>paradeisos</i>, i.e., “paradise.”</i> and brought in the workers. Now by the orchard gate there was a certain storehouse full of everything good. The king said: Whoever does his work wholeheartedly will receive his reward from here, but whoever does not do his work wholeheartedly, him I shall return to my palace<sup class="footnote-marker">48</sup><i class="footnote">Lat.: <i>palatium</i>.</i> and judge. Who is this king? This is the Supreme King of Kings, the Holy One. And what is the garden? It is this world. Within it the Holy One has put the children of Adam so that they may observe the Torah. But he has made a stipulation with them and said to them: For everyone who truly observes the Torah, here is paradise &lt; lying &gt; before him; but for everyone who does not truly observe the Torah, here is Gehinnom &lt; lying &gt; before him. The Holy One said: Although I seemed to have removed my Divine Presence from the Sanctuary, still (in Ps. 11:4): <small>MY EYES BEHOLD, &lt;MY EYELIDS TEST THE CHILDREN OF ADAM</small> &gt;.<sup class="footnote-marker">49</sup><i class="footnote">The Masoretic Text of this verse reads “his” for <small>MY</small> in both places.</i> Whom does he test? (According to vs. 5:) <small>THE LORD TESTS THE RIGHTEOUS</small>. And why does he not test the wicked? R. Jannay said: When the flax worker is pounding away and sees that the flax is good, he pounds it a lot; but, when he sees that it is not good, he does not pound on it, lest it be spoiled.<sup class="footnote-marker">50</sup><i class="footnote">Gen. R. 32:3; 34:2; 55:2; Cant. R. 2:16:2.</i> Ergo (in Ps. 11:4:) <small>HIS EYES BEHOLD, HIS EYELIDS TEST THE CHILDREN OF ADAM</small>. [And whom does he test? The righteous, as stated (in vs. 5):] <small>THE LORD TESTS THE RIGHTEOUS</small>."""
    print(summarize_based_on_uniqueness(text, uniqueness))
