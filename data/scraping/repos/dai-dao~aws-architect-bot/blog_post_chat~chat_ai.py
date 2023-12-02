import re
import guidance
from knowledge_base import KnowledgeBase
import json
from typing import Tuple, List
import tiktoken
from termcolor import colored, cprint
from memory import ChatMemory
from uuid import uuid4
import logging


guidance.llm = guidance.llms.OpenAI("text-davinci-003") 


def print_debug(debug_msg):
    text = colored("DEBUG: " + debug_msg, "blue", attrs=["bold"])
    print(text)


# Load few-shot examples
summarize_knowledge_examples = []
with open("knowledge-data/summarize_knowledge_questions.txt", "r") as f:
    for l in f.readlines():
        summarize_knowledge_examples.append(l.strip())        


def get_token_length(input : str) -> int:
    encoder = tiktoken.encoding_for_model("text-davinci-003")
    encoded_content = encoder.encode(input)
    return len(encoded_content)


class ChatAI:
    def __init__(self, debug=False) -> None:
        self.intent_prompt = guidance(
        '''
        Given a user question, please classify the intent of the question. 
        Please answer with a single word, either "summarize_knowledge", "specific_question".
        ----

        {{~! display the few-shot summarize_knowledge_examples ~}}
        {{~#each summarize_knowledge_examples}}
        Sentence: {{this}}
        Intent: summarize_knowledge
        ---
        {{~/each}}


        {{~! Question }}
        Question: {{input}}
        Intent: {{#select "intent" logprobs='logprobs'}} summarize_knowledge{{or}} specific_question{{/select}}
        '''.strip())
        self.select_sections_prompt = guidance("""
        Given the blog post title and the section names, please select the most relevant sections that can be used to answer the user question.

        Question:
        {{question}}

        Title:
        {{title}}
        Section names:
        {{~#each section_names}}
        Section: {{this}}
        {{~/each}}

        Relevant section 1: {{select "section_1" logprobs='logprobs' options=section_names}}
        Relevant section 2: {{select "section_2" logprobs='logprobs' options=section_names}}
        Relevant section 3: {{select "section_3" logprobs='logprobs' options=section_names}}
        """.strip())
        self.summarize_prompt = guidance("""
        Given the blog post title and the section names, please answer the user question.

        Question:
        {{question}}

        Title:
        {{title}}
        Section names:
        {{~#each section_names}}
        Section: {{this}}
        {{~/each}}

        answer: {{gen "answer"}}
        """.strip())
        self.detail_answer_prompt = guidance("""
        Given the blog post title and the content in markdown, please generate an answer for the user question 
        with step by step instructions with code examples and reference URLs from content if provided. You need to 
        provide URL sources for where the information is coming from. If the question is asking for a very specific command 
        or a small piece of information then just provide that along with the references.

        Question:
        {{question}}

        Title:
        {{title}}
        Markdown Content:
        {{content}}

        Answer:
        {{gen "answer" max_tokens=700}}
        """.strip())
        self.answer_from_memory_prompt = guidance("""
        Given the conversation history, is it possible to answer the user question? If yes then please 
        provide the answer, if not then respond with "I need more information to answer this question".
        
        {{history}}
        
        Question: {{question}}
        Answer: {{gen "answer"}}                                   
        """)
        
        self.debug = debug
        self.knowledge_base = KnowledgeBase()
        self.memory = ChatMemory(str(uuid4()))
        self.section_titles = list(self.knowledge_base.content_map.keys())


    def summarize_knowledge(self, input : str) -> str:
        relevant_titles = self.section_titles.copy()
        relevant_titles.remove("About the Authors")
        relevant_titles.remove("Conclusion")
        out = self.summarize_prompt(
            title = self.knowledge_base.title,
            section_names = relevant_titles,
            question = input
        )
        return out["answer"].strip()


    def get_intent(self, input : str) -> str:
        response = self.intent_prompt(
            summarize_knowledge_examples = summarize_knowledge_examples,
            input = input
        )
        out = response["intent"].strip()
        if self.debug:
            print_debug("Identified intent " + out)
        return out


    def extract_related_sections(self, input) -> Tuple[str, str, str]:
        out = self.select_sections_prompt(
            title = self.knowledge_base.title,
            section_names = self.section_titles,
            question = input
        )
        if self.debug:
            print_debug(f'Selected sections {out["section_1"]}, {out["section_2"]}, {out["section_3"]}')
        return out["section_1"], out["section_2"], out["section_3"]


    def generate_detail_output(self, input : str, content : str) -> str:
        out = self.detail_answer_prompt(
            question = input,
            title = self.knowledge_base.title,
            content = content,
        )
        return out["answer"].strip()


    def clarify_answer(self, input : str, history : str, content : str) -> str:
        out = self.clarify_answer_prompt(
            question = input,
            history = history,
            title = self.knowledge_base.title,
            content = content
        )
        return out["answer"]


    def get_content_from_sections(self, section_1 : str, section_2 : str, section_3 : str ) -> str:
        section_1_content = self.knowledge_base.content_map[section_1].strip()
        section_2_content = self.knowledge_base.content_map[section_2].strip()
        section_3_content = self.knowledge_base.content_map[section_3].strip()
        content = f"""
            {section_1}
            {section_1_content}
            {section_2}
            {section_2_content}
            {section_3}
            {section_3_content}
        """.strip()
        return content


    def process_input(self, input : str) -> str:
        if input.strip() == "":
            return "No output"
        
        memories = self.memory.retrieve_relevant_memory(input)
        history = ""
        for i in range(min(3, len(memories))):
            history += memories[i]['content'] + "\n"
            
        if history.strip() != "":
            out = self.answer_from_memory_prompt(history=history, question=input)
            if self.debug:
                print_debug("History " +  history)
                print_debug("Answer from history " + out["answer"])
                
            if "I need more information to answer this question".lower() not in out["answer"].lower():
                return out["answer"]

        intent = self.get_intent(input)
        if intent == "summarize_knowledge":
            return self.summarize_knowledge(input)
        elif intent == "specific_question":
            section_1, section_2, section_3 = self.extract_related_sections(input)
            content = self.get_content_from_sections(section_1, section_2, section_3)
            if self.debug:
                print_debug(f"Content length {get_token_length(content)}")
            
            out = ""
            if get_token_length(content) > 3200:
                out = content
            else:
                out = self.generate_detail_output(input, content)
                
            content = f"""
            Q: {input}
            A: {out}
            """.strip()
            self.memory.add(is_human=False, content = content)
            return out