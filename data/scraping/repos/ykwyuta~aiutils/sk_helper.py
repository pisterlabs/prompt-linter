import os
import re
from typing import Dict, List
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from langchain.text_splitter import CharacterTextSplitter
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper

deployment = "gpt-35-turbo"
endpoint = os.environ["OPENAI_API_BASE"]
api_key = os.environ["OPENAI_API_KEY"]

class SkHelper:

    def __init__(self) -> None:
        self.kernel = sk.Kernel()
        self.kernel.add_chat_service("chat_completion", AzureChatCompletion(deployment, endpoint, api_key))

    def summarize_chunk(self, chunk: str) -> str:
        prompt="""
        Take a deep breath, then carefully, summarize the CONTENT within 200 characters.
        content
        ----------------
        {{$content}}
        """
        chat = self.kernel.create_semantic_function(prompt, "summarizer", max_tokens=2000, temperature=0.2, top_p=0.5)
        context = self.kernel.create_new_context()
        context["content"] = chunk
        result = chat.invoke(context=context)
        return result.result

    def summarize_chunks(self,chunks: List[str], max_length=1000, loop=0) -> str:
        text_splitter = CharacterTextSplitter(
            separator = "\n\n",
            chunk_size = max_length,
            chunk_overlap = 0,
            length_function = len,
        )
        summary = ""
        for chunk in chunks:
            summary += self.summarize_chunk(chunk)
        if len(summary) < max_length:
            return summary
        if loop > 3:
            return summary
        chunks = text_splitter.split_text(summary)
        return self.summarize_chunks(chunks, max_length, loop+1)

    def translate(self, text: str) -> str:
        prompt="""
        textを日本語にしてください。丁寧な日本語でお願いします。登録や参加方法、フィードバック方法について言及している箇所は削除してください。
        text
        ----------------
        {{$text}}
        """
        chat = self.kernel.create_semantic_function(prompt, "translator", max_tokens=2000, temperature=0.2, top_p=0.5)
        context = self.kernel.create_new_context()
        context["text"] = text
        result = chat.invoke(context=context)
        return result.result

    def extract_keywords(self, text: str) -> List[Dict[str, str]]:
        prompt="""
        Identify and describe the keywords in the following text: TEXT. Please format your response as follows: 
        "Keyword:<keyword>; Description:<description>;"
        TEXT
        ----------------
        {{$text}}
        ----------------

        EXAMPLE:
        TEXT
        ----------------
        The cat (Felis catus) is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae and is often referred to as the domestic cat to distinguish it from the wild members of the family.
        ----------------
        Keyword: Cat (Felis catus) ; Description: A domestic species of small carnivorous mammal. ;
        Keyword: Family Felidae ; Description: The family to which the domestic cat belongs, distinguishing it from wild members of the family. ;
        """
        chat = self.kernel.create_semantic_function(prompt, "keywords", max_tokens=2000, temperature=0.2, top_p=0.5)
        context = self.kernel.create_new_context()
        context["text"] = text
        result = chat.invoke(context=context)
        answer_text = result.result
        keywords_descriptions = []
        for line in answer_text.split("\n"):
            item = line.strip().split(";")
            if len(item) < 2:
                continue
            keywords_descriptions.append({
                "keyword": item[0].replace("Keyword", "").replace(":", "").strip(),
                "description": item[1].replace("Description", "").replace(":", "").strip()
            })
        return keywords_descriptions
    
    def extract_keywords_with_search(self, text: str, domain: str = None) -> List[Dict[str, str]]:
        keywords_descriptions = self.extract_keywords(text)
        for keyword_description in keywords_descriptions:
            try:
                keyword = keyword_description["keyword"]
                query = keyword
                snippets = self.get_snippets(query, domain)
                keyword_description["search"] = "".join([snippet.replace("\n", "") for snippet in snippets])
            except Exception as e:
                pass
        return keywords_descriptions

    def extract_keywords_with_wikipedia(self, text: str) -> List[Dict[str, str]]:
        keywords_descriptions = self.extract_keywords(text)
        for keyword_description in keywords_descriptions:
            try:
                keyword = keyword_description["keyword"]
                query = keyword
                summary = self.get_wikipedia_summary(query)
                keyword_description["wikipedia"] = summary[:500]
            except Exception as e:
                pass
        return keywords_descriptions

    def get_wikipedia_summary(self, query: str) -> str:
        wikipedia = WikipediaAPIWrapper()
        result = wikipedia.run(query)
        return result

    def get_snippets(self, query: str, domain:str =  None) -> List[str]:
        search = DuckDuckGoSearchAPIWrapper()
        if domain is not None:
            query = "{0} site:{1}".format(query, self.domain)
        result=search.get_snippets(query)
        return result

    def question_and_answer(self, text: str) -> List[Dict[str, str]]:
        prompt="""
Based on the following TEXT, please generate 10 questions to measure understanding, along with their answers and explanations for the answers.
Please format your response as follows: 
Question: <question>;Answer: <answer>; Explanation: <explanation>;

TEXT
----------------
{{$text}}
----------------

EXAMPLE:
TEXT
----------------
The sun is a star located at the center of our solar system. It provides light and heat that sustain life on Earth. The sun is composed primarily of hydrogen and helium. Its gravitational force keeps the planets, including Earth, in their orbits.
----------------
Question: Where is the sun located in our solar system?; Answer: At the center; Explanation: The text mentions that the sun is located at the center of our solar system.;
Question: What does the sun provide that sustains life on Earth?; Answer: Light and heat; Explanation: The text states that the sun provides light and heat that sustain life on Earth.;

        """
        chat = self.kernel.create_semantic_function(prompt, "question_and_answer", max_tokens=2000, temperature=0.2, top_p=0.5)
        context = self.kernel.create_new_context()
        context["text"] = text
        result = chat.invoke(context=context)
        question_and_answer = result.result
        question_and_answer_list = []
        for line in question_and_answer.split("\n"):
            item = line.strip().split(";")
            if len(item) < 3:
                continue
            question_and_answer_list.append({
                "question": item[0].replace("Question", "").replace(":", "").strip(),
                "answer": item[1].replace("Answer", "").replace(":", "").strip(),
                "explanation": item[2].replace("Explanation", "").replace(":", "").strip()
            })
        return question_and_answer_list