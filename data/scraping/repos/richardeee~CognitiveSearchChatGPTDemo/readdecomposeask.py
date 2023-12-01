import openai
from approaches.approach import Approach
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from langchain.chat_models  import AzureChatOpenAI
from langchain.llms.openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate, BasePromptTemplate
from langchain.schema import AgentAction, AgentFinish, HumanMessage
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain.agents.react.base import ReActDocstoreAgent
from langchainadapters import HtmlCallbackHandler
from langchain import LLMChain
from text import nonewlines
from typing import List, Union
import requests
import re
import json
import tiktoken
import math

class ReadDecomposeAsk(Approach):
    def __init__(self, search_client: SearchClient, openai_deployment: str, sourcepage_field: str, content_field: str, azure_openai_key: str, azure_openai_base: str, bing_search_subscriptin_key: str, bing_search_endpoint: str, sourcepage_path_field: str):
        self.search_client = search_client
        self.openai_deployment = openai_deployment
        self.sourcepage_field = sourcepage_field
        self.content_field = content_field
        self.azure_openai_key = azure_openai_key
        self.azure_openai_base = azure_openai_base
        self.bing_search_subscriptin_key = bing_search_subscriptin_key
        self.bing_search_endpoint = bing_search_endpoint
        self.sourcepage_path_field = sourcepage_path_field

    def search(self, q: str, overrides: dict) -> str:
        use_semantic_captions = True if overrides.get("semantic_captions") else False
        top = overrides.get("top") or 3
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None

        if overrides.get("semantic_ranker"):
            r = self.search_client.search(q,
                                          filter=filter,
                                          query_type=QueryType.SEMANTIC, 
                                          query_language="zh-CN", 
                                        #   query_speller="lexicon", 
                                          semantic_configuration_name="default", 
                                          top = top,
                                          query_caption="extractive|highlight-false" if use_semantic_captions else None)
        else:
            r = self.search_client.search(q, filter=filter, top=top)
        if use_semantic_captions:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(" . ".join([c.text for c in doc['@search.captions'] ])) for doc in r]
        else:
            self.results = [doc[self.sourcepage_field] + ":" + nonewlines(doc[self.content_field][:500]) for doc in r]

        if use_semantic_captions:
            self.json_results = [{"sourcepage": doc[self.sourcepage_field], "content": nonewlines(" . ".join([c.text for c in doc['@search.captions']])), "sourcepage_path": doc[self.sourcepage_path_field]} for doc in r]     
        else:
            self.json_results = [{"sourcepage": doc[self.sourcepage_field], "content": nonewlines(doc[self.content_field]), "sourcepage_path": doc[self.sourcepage_path_field]} for doc in r if doc['@search.score']]
        # Add Bing Search
        # search_result = self.get_bing_search_result(q, top)
        # self.results = self.results + search_result
        result = json.dumps(self.results, ensure_ascii=False)
        return result

    def lookup(self, q: str, overrides: dict) -> str:
        exclude_category = overrides.get("exclude_category") or None
        filter = "category ne '{}'".format(exclude_category.replace("'", "''")) if exclude_category else None
        
        if(overrides.get("semantic_ranker")):
            r = self.search_client.search(q,
                                      top = 1,
                                      include_total_count=True,
                                      query_type=QueryType.SEMANTIC, 
                                      query_language="zh-CN", 
                                      semantic_configuration_name="default",
                                      query_answer="extractive|count-1",
                                      query_caption="extractive|highlight-false")
        else:
            r = self.search_client.search(q, filter=filter, top=1, include_total_count=True)
            
        answers = r.get_answers()
        result = ''
        if answers and len(answers) > 0:
            result =  answers[0].text
            print(answers[0])
            self.json_results = [{"sourcepage": answers[0]['sourcepage'], "content": nonewlines(answers[0].text), "sourcepage_path": answers[0]['sourcepage_path']}]
        if r.get_count() > 0:
            result =  "\n".join(d['content'] for d in r)
            print(result)
            self.json_results = [{"sourcepage": d['sourcepage'], "content": nonewlines(d['content']), "sourcepage_path": d['sourcepage_path']} for d in r]
        
        # search_result = self.lookup_bing_result(q,1)
        # result = result + search_result
        return result        

    def run(self, q: str, overrides: dict) -> any:
        # Not great to keep this as instance state, won't work with interleaving (e.g. if using async), but keeps the example simple
        self.results = None

        # Use to capture thought process during iterations
        cb_handler = HtmlCallbackHandler()
        # cb_manager = CallbackManager(handlers=[cb_handler])

        # llm = AzureOpenAI(deployment_name=self.openai_deployment, temperature=overrides.get("temperature") or 0.3, openai_api_key=openai.api_key)
        
        llm = AzureChatOpenAI(
            openai_api_base=self.azure_openai_base,
            openai_api_version="2023-03-15-preview",
            deployment_name="gpt-4",
            openai_api_key=self.azure_openai_key,
            openai_api_type = "azure",
            temperature=0.0
        )
        
        tools = [
            Tool(name="Search", description="Search in document store", func=lambda q: self.search(q, overrides)),
            Tool(name="Lookup", description="Lookup in document store", func=lambda q: self.lookup(q,overrides))
        ]

        # Like results above, not great to keep this as a global, will interfere with interleaving
        global prompt
        prompt = CustomPromptTemplate(
            template=TEMPLATES,
            tools=tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=["input", "intermediate_steps"]
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        tool_names = [tool.name for tool in tools]
        output_parser = CustomOutputParser()
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain, 
            output_parser=output_parser,
            stop=["\nObservation:"], 
            allowed_tools=tool_names
        )
        agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,callbacks=[cb_handler])
        result = agent_executor.run(q)

        # agent = ReAct.from_llm_and_tools(llm, tools)
        # chain = AgentExecutor.from_agent_and_tools(agent, tools, verbose=True, callbacks=[cb_handler])
        # result = chain.run(q)

        # Fix up references to they look like what the frontend expects ([] instead of ()), need a better citation format since parentheses are so common
        # result = result.replace("(", "[").replace(")", "]")

        data_points = []
        if hasattr(self, 'json_results'):
            data_points = [{"sourcepage": d['sourcepage'], "content": nonewlines(d['content']), "sourcepage_path": d['sourcepage_path']} for d in self.json_results]
        return {"data_points": data_points, "answer": result, "thoughts": cb_handler.get_and_reset_log()}
    
    def get_bing_search_result(self, question, top):
        print("-------------------------searching: " + question + "-------------------------")
        mkt = 'zh-CN'
        params = { 'q': question, 'mkt': mkt , 'answerCount': top}
        headers = { 'Ocp-Apim-Subscription-Key': self.bing_search_subscriptin_key }
        r = requests.get(self.bing_search_endpoint, headers=headers, params=params)
        json_response = json.loads(r.text)
        # print(json_response)
        result = [page['name'] + ": " + nonewlines(page['snippet']) + " <" + page['url'] + ">" for page in list(json_response['webPages']['value'])[:top]]   
        return result
    
    def lookup_bing_result(self, question, top=1):
        print("-------------------------lookup: " + question + "-------------------------")
        mkt = 'zh-CN'
        params = { 'q': question, 'mkt': mkt , 'answerCount': top}
        headers = { 'Ocp-Apim-Subscription-Key': self.bing_search_subscriptin_key }
        r = requests.get(self.bing_search_endpoint, headers=headers, params=params)
        json_response = json.loads(r.text)
        # print(json_response)
        url = list(json_response['webPages']['value'])[0]['url']
        r = requests.get(url)
        #Remove html tags using regex
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', r.text)
        print("text length: " + str(len(text)))
        num_of_tokens = num_tokens_from_messages([text])
        if(num_of_tokens > 4000):
            #Split text into 4000 tokens and summarize each part
            num_of_parts = math.ceil(num_of_tokens / 4000)
            print("num_of_parts: " + str(num_of_parts))
            part_length = math.ceil(len(text) / num_of_parts)
            print("part_length: " + str(part_length))
            parts = [text[i:i+part_length] for i in range(0, len(text), part_length)]
            
            #Summarize each part
            summaries = []
            for part in parts:
                summaries.append(self.summarize(part))

            #Combine summaries
            summary = ""
            for s in summaries:
                summary += s
            return summary
        return text

    def summarize(self, text):
        summarize_prompt = """
        [SUMMARIZATION RULES]
            DONT WASTE WORDS
            USE SHORT, CLEAR, COMPLETE SENTENCES.
            DO NOT USE BULLET POINTS OR DASHES.
            USE ACTIVE VOICE.
            MAXIMIZE DETAIL, MEANING
            FOCUS ON THE CONTENT

            [BANNED PHRASES]
            This article
            This document
            This page
            This material
            [END LIST]

            Summarize:
            Hello how are you?
            +++++
            Hello

            Summarize this
            {{$input}}
            +++++"""
        message = [{
            "role": "user",
            "content": summarize_prompt.format(input=text)
        }]
        completion = openai.ChatCompletion.create(
            engine=self.openai_deployment,
            messages=message,
            temprature=0.0
        )
        wrap_upped_answer = completion['choices'][0]['message']['content']
        return wrap_upped_answer

def num_tokens_from_messages(messages, model="gpt-4"):
    """Returns the number of tokens used by a list of messages."""
    print("Counting tokens..." + messages)
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(messages))

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
       
TEMPLATES = """
Assistant is a large language model trained by OpenAI.
Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
1. You can provide additional relevant details to responde thoroughly and comprehensively to cover multiple aspects in depth.
2. You should always answer user questions based on the context provided.
3. If the context does not provide enough information, you answer ```I don't know``` or ```I don't understand```.
4. You should not answer questions that are not related to the context.
5. You should explain the reasons behind your answers.
6. Answer in HTML format.
7. Answer in Simplified Chinese.
8. If there's images in the context, you should display them in your answer.
9. Use HTML table format to display tabular data.

Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
Don't use reference, ALWAYS keep source page pth in (), e.g. (http://www.somedomain1.com/info1.txt)(http://www.somedomain2.com/info2.pdf).

You have access to the following tools:{tools}
To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```
When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
Action Input:
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Examples:
Question: What is the elevation range for the area that the eastern sector of the
Colorado orogeny extends into?
Thought: I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the
area.
Action: Search
Action Input: Colorado orogeny
Observation: [info1.pdf](http://www.example1.com/info1.pdf) The Colorado orogeny was an episode of mountain building (an orogeny) in
Colorado and surrounding areas.
Thought: It does not mention the eastern sector. So I need to look up eastern
sector.
Action: Lookup
Action Input: eastern sector
Observation: [info2.txt](http://www.example1.com/info2.pdf) The eastern sector extends into the High Plains and is called
the Central Plains orogeny.
Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I
need to search High Plains and find its elevation range.
Action: Search
Action Input: High Plains
Observation: [some_file.pdf](http://www.exampl1.com/some_file.pdf) High Plains refers to one of two distinct land regions
Thought: I need to instead search High Plains (United States).
Action: Search
Action Input: High Plains (United States)
Observation: [filea.pdf](http://www.example1.com/filea.pdf) The High Plains are a subregion of the Great Plains. [another-ref.docx](http://www.example1.com/another-ref.docx) From east to west, the
High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130
m).
Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer
is 1,800 to 7,000 ft.
Final Answer: 1,800 to 7,000 ft [filea.pdf](http://www.example1.com/filea.pdf)

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons"
character Milhouse, who Matt Groening named after who?
Thought: The question simplifies to "The Simpsons" character Milhouse is named after
who. I only need to search Milhouse and find who it is named after.
Action: Search
Action Input: Milhouse
Observation: [info7.pdf](http://www.example2.com/info7.pdf) Milhouse Mussolini Van Houten is a recurring character in the Fox animated
television series The Simpsons voiced by Pamela Hayden and created by Matt
Groening.
Thought: The paragraph does not tell who Milhouse is named after, maybe I can look up
"named after".
Action: Lookup
Action Input: named after
Observation: [historyref2.txt](http://www.example2.com/historyref2.txt) Milhouse was named after U.S. president Richard Nixon, whose
middle name was Milhous.
Thought: Milhouse was named after U.S. president Richard Nixon, so the answer is
Richard Nixon.
Final Answer: Richard Nixon [historyref2.txt](http://www.example2.com/historyref2.txt)

Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The
Saimaa Gesture?
Thought: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which
documentary is about Finnish rock groups.
Action: Search
Action Input: Adam Clayton Powell
Observation: [info9.pdf](http://www.example3.com/info9.pdf) Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell
III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office
Building', 'Isabel Washington Powell', 'Adam Powell', vAdam Clayton Powell
(film)', 'Giancarlo Esposito'].
Thought: To find the documentary, I can search Adam Clayton Powell (film).
Action: Search
Action Input: Adam Clayton Powell (film)
Observation: [data123.txt](http://www.example3.com/data123.txt) Adam Clayton Powell is a 1989 American documentary film directed by
Richard Kilberg. The film is about the rise and fall of influential
African-American politician Adam Clayton Powell Jr.[3][4] It was later aired
as part of the PBS series The American Experience.
Thought: Adam Clayton Powell (film) is a documentary about an African-American
politician, not Finnish rock groups. So the documentary about Finnish rock
groups must instead be The Saimaa Gesture.
Final Answer: The Saimaa Gesture [data123.txt](http://www.example3.com/data123.txt)

Question: What profession does Nicholas Ray and Elia Kazan have in common?
Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then
find the profession they have in common.
Action: Search
Action Input: Nicholas Ray
Observation: [files-987.png](http://www.example4.com/files-987.png) Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 - June 16,
1979) was an American film director, screenwriter, and actor best known for
the 1955 film Rebel Without a Cause.
Thought: Professions of Nicholas Ray are director, screenwriter, and actor. I need
to search Elia Kazan next and find his professions.
Action: Search
Action Input: Elia Kazan
Observation: [files-654.txt](http://www.example4.com/files-654.txt) Elia Kazan was an American film and theatre director, producer, screenwriter
and actor.
Thought: Professions of Elia Kazan are director, producer, screenwriter, and actor.
So profession Nicholas Ray and Elia Kazan have in common is director,
screenwriter, and actor.
Final Answer: director, screenwriter, actor [files-987.png](http://www.example4.com/files-987.png)[files-654.txt](http://www.example4.com/files-654.txt)

Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought: I need to search Arthur's Magazine and First for Women, and find which was
started first.
Action: Search
Action Input: Arthur's Magazine
Observation: [magazines-1850.pdf](http://www.example5.com/magazines-1850.pdf) Arthur's Magazine (1844-1846) was an American literary periodical published
in Philadelphia in the 19th century.
Thought: Arthur's Magazine was started in 1844. I need to search First for Women
next.
Action: Search
Action Input: First for Women
Observation 2: [magazines-1900.pdf](http://www.example5.com/magazines-1900.pdf) First for Women is a woman's magazine published by Bauer Media Group in the
USA.[1] The magazine was started in 1989.
Thought: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First
for Women), so Arthur's Magazine was started first.
Final Answer: Arthur's Magazine [magazines-1850.pdf](http://www.example5.com/magazines-1850.pdf)[magazines-1900.pdf](http://www.example5.com/magazines-1900.pdf)

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought: I need to search Pavel Urysohn and Leonid Levin, find their types of work,
then find if they are the same.
Action: Search
Action Input: Pavel Urysohn
Observation: [info4444.pdf](http://www.example6.com/info4444.pdf) Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet
mathematician who is best known for his contributions in dimension theory.
Thought: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and
find its type of work.
Action: Search
Action Input: Leonid Levin
Observation: [datapoints_aaa.txt](http://www.example6.com/datapoints_aaa.txt) Leonid Anatolievich Levin is a Soviet-American mathematician and computer
scientist.
Thought: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn
and Leonid Levin have the same type of work.
Final Answer: yes [info4444.pdf](http://www.example6.com/info4444.pdf)[datapoints_aaa.txt](http://www.example6.com/datapoints_aaa.txt)


Question: {input}
{agent_scratchpad}
"""
