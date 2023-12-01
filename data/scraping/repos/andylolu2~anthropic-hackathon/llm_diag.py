import json
import os
import re
import time

import torch
from dotenv import load_dotenv
from langchain.cache import InMemoryCache
from langchain.chains import ConversationChain
from langchain.chat_models import ChatAnthropic
from langchain.globals import set_llm_cache
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.tools import BraveSearch
from langchain.vectorstores import MongoDBAtlasVectorSearch

from bert_embedder import BertEmbeddings
from medwise import query_medwise

load_dotenv()

set_llm_cache(InMemoryCache())


def get_investigate_prompt(knowledge="", conversation=""):
    template = """
Hi Claude, I am a medical expert analysing the quality of history taken by doctors. As part of the simulation, you will be asked to provide your thoughts to compare to mine.

You are having a professional conversation with me, a trained doctor. You should be talkative and provide lots of specific details from its context. If you are not sure of the answer, says it and explain why you are unsure.
You will read a consultation between a GP (me) and a patient that has already happened.
The patient starts by saying their initial story and what they would like help with.
This is never enough to get to a full diagnosis. In fact the role of an excellent GP is to ask a series of very well phrased questions that most effectively and intelligently dissect the diagnostic search space to reach a set of most probable differential diagnoses, and most importantly to rule out differential diagnoses that are potentially life threatening to the patient, even if they are not the most likely.
The conversation takes the form of a series of questions (asked by me) and answers (from the patient).

The full conversation between the doctor and the patient is as follows:
<conversation>
{conversation}
</conversation>

<tasks_if_investigation>
As you read through the conversation, pause and think after each response from the patient. I want you to think of three things given the information you have at each point.
The top differential diagnoses that explain the symptoms the patient is describing.
The most dangerous diagnoses that even if unlikely could potentially explain the cluster of symptoms from the patient and that therefore you need to rule out
And most importantly, given these two types of differentials, what is the most informative next question/set of questions that will allow you to efficiently dissect the diagnostic search space
At each point, you will internally compare your next best question/set of questions with the clinician's actual question/set of questions.  

Finally, showing your reasoning:
1. What are the most probable differential diagnoses including important life-threatening ones that mandate exclusion that the doctor HAS NOT appropriately enquired about and ruled in or out. (For appropriately I mean that the patient's answer does not leave scope for misunderstanding, and if it does that it should be clarified.)
2. What are the most important differential diagnoses that I have not enquired about at all?
3. What about the consultation makes you believe that? Reference relevant parts of the conversation.
4. What are the most efficient questions, physical exam findings and investigations to help rule in or out these differentials?

Make sure that your suggested steps are structured by history - examination - investigations and that you do not repeat what I have already been asked/said.
</tasks_if_investigation>

Here is the additional domain knowledge and context that has been retrieved based on the conversation that you should use to make more informed reasoning:
<knowledge>
{knowledge}
</knowledge>

Below is your conversation with me so far:
<conversation_history>
{{history}}
</conversation_history>

<current_input>
Doctor: {{input}} 
</current_input>

If my current input DOESN'T pertain to performing further investigation, then simply answer my question/query appropriately. You are encouraged to use the knowledge and context provided to help you answer the question.
Otherwise, if my current input DOES pertain to performing further investigation, answer according to the instructions in the <tasks_if_investigation> tag above.
""".strip()

    template = template.format(knowledge=knowledge, conversation=conversation)

    return PromptTemplate(input_variables=["history", "input"], template=template)


def get_keyword_prompt():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """
You are a helpful chatbot with a lot of knowledge about the medical domain.

I want you to look at this conversation between a doctor and a patient. I want you to extract three to five most relevant keywords/phrases that summarize the important medical topics related to this patient. Each keyword/phrase should be enclosed in its own <keyword> </keyword> tag.
    
<conversation>
{conversation}
</conversation>
""".strip(),
            ),
        ]
    )
    return prompt


class DiagnosisLLM:
    def __init__(self):
        self.llm = ChatAnthropic(temperature=0.2, max_tokens=4096, cache=True)
        self.keywords = None
        self.conv_chain = None
        self.keyword_chain = None
        self.memory = None
        self.transcript = None
        self.context = None

    def init_conv_chain(self) -> None:
        self.memory = ConversationSummaryBufferMemory(
            return_messages=True, llm=self.llm
        )
        self.get_context()
        knowledge = self.parse_context()
        investigation_prompt = get_investigate_prompt(knowledge, self.transcript)
        self.conv_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=investigation_prompt,
        )

    def parse_context(self):
        guidelines_knowledge = "<guidelines>\n"
        for item in self.context["guidelines"]:
            text = item["content"]
            guidelines_knowledge += f"<content>\n{text}\n</content>\n\n"
        guidelines_knowledge += "</guidelines>"

        textbook_knowledge = "<textbook>\n"
        for text in self.context["textbook"]:
            textbook_knowledge += f"<content>\n{text}\n</content>\n\n"
        textbook_knowledge += "</textbook>"

        web_knowledge = "<web_search>\n"
        for item in self.context["web"]:
            title = item["title"]
            text = item["snippet"]
            web_knowledge += f"<title>\n{title}\n</title>\n"
            web_knowledge += f"<content>\n{text}\n</content>\n\n"
        web_knowledge += "</web_search>"

        knowledge = (
            guidelines_knowledge + "\n" + textbook_knowledge + "\n" + web_knowledge
        )
        return knowledge

    def get_chat_history(self):
        chat_history = []
        for message in self.memory.chat_memory.messages:
            role = type(message).__name__
            chat_history.append({"role": role, "content": message.content})
        return chat_history

    def get_sources(self):
        links = {"guidelines": [], "web": []}
        for guideline in self.context["guidelines"]:
            url = guideline["url"]
            links["guidelines"].append(url)
        for web_res in self.context["web"]:
            url = web_res["link"]
            links["web"].append(url)
        return links

    def answer_doctor_query(self, query: str):
        """
        Answer the doctor's query backed by the knowledge relevant to the patient's condition
        """
        self.conv_chain.run({"input": query})
        chat_history_so_far = self.get_chat_history()
        links = self.get_sources()
        return {"chat_history": chat_history_so_far, "sources": links}

    def extract_from_transcript(self, transcript):
        self.transcript = transcript
        self.keywords = self.keyword_chain.invoke({"conversation": transcript}).content
        print(self.keywords)
        # Regular expression to find all words enclosed in <keyword> tags
        keywords = re.findall(r"<keyword>(.*?)</keyword>", self.keywords)
        # Join the words into a single string separated by spaces
        self.keywords = ", ".join(keywords)

        time.sleep(2)
        print("==========keywords========")
        print(self.keywords)

    def init_extraction_chains(self) -> None:
        keyword_prompt = get_keyword_prompt()
        self.keyword_chain = keyword_prompt | self.llm

    def get_context_from_brave(self, k=5):
        if k == 0:
            return json.dumps([])
        brave_search_tool = BraveSearch.from_api_key(
            api_key=os.environ["BRAVE_API_KEY"], search_kwargs={"count": k}
        )
        out = brave_search_tool.run(
            f"Medical documents on: {self.keywords}"
        )  # TODO prompt engineer improvement
        return out

    def get_context_from_medwise(self, k=1, render_js: bool = False):
        if k == 0:
            return []
        results = query_medwise(self.keywords, k=k, render_js=render_js)
        return results

    def get_context_from_textbook(self, k=5):
        if k == 0:
            return []
        embed = BertEmbeddings(
            model_name="michiyasunaga/BioLinkBERT-large",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # MONGODB_ATLAS_CLUSTER_URI = "mongodb+srv://evanrex:c1UgqaM0U2Ay72Es@cluster0.ebrorq5.mongodb.net/?retryWrites=true&w=majority"
        MONGODB_ATLAS_CLUSTER_URI = os.environ["MONGODB_ATLAS_CLUSTER_URI"]
        ATLAS_VECTOR_SEARCH_INDEX_NAME = "embedding"

        vector_search = MongoDBAtlasVectorSearch.from_connection_string(
            MONGODB_ATLAS_CLUSTER_URI,
            "macleod_textbook.paragraphs",
            embed,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )

        results = vector_search.similarity_search_with_score(
            query=self.keywords,
            k=k,
        )  # TODO use paragraph.next and paragraph.prev to get window around returned documents

        results_list = []
        for i, result in enumerate(results):
            doc, score = result
            results_list.append(doc.page_content)
        return results_list

    def get_context(self, k_brave=1, k_medwise=5, k_textbook=1):
        brave = self.get_context_from_brave(k=k_brave)
        print("==============brave=================")
        new_brave = json.loads(brave)
        print(new_brave)
        time.sleep(2)
        medwise = self.get_context_from_medwise(k=k_medwise)
        print("==============medwise=================")
        print(medwise)
        textbook = self.get_context_from_textbook(k=k_textbook)
        print("==============textbook=================")
        print(textbook)
        self.context = {"guidelines": medwise, "textbook": textbook, "web": new_brave}
