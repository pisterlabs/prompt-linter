### c.f.
### https://python.langchain.com/en/latest/use_cases/agents/characters.html
### https://docs.pinecone.io/docs/python-client#usage

import os
import re
import math
import faiss
import pinecone

from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from termcolor import colored

from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseLanguageModel, Document
from langchain.vectorstores import FAISS, Pinecone


USER_NAME = "superuser" # The name you want to use when interviewing the agent.
LLM = ChatOpenAI(max_tokens=1500) # Can be any LLM you want.

class GenerativeAgent(BaseModel):
    """A character with memory and innate characteristics."""

    name: str
    age: int
    traits: str
    """The traits of the character you wish not to change."""
    status: str
    """Current activities of the character."""
    llm: BaseLanguageModel
    memory_retriever: TimeWeightedVectorStoreRetriever
    """The retriever to fetch related memories."""
    verbose: bool = False

    reflection_threshold: Optional[float] = None
    """When the total 'importance' of memories exceeds the above threshold, stop to reflect."""

    current_plan: List[str] = []
    """The current plan of the agent."""

    summary: str = ""  #: :meta private:
    summary_refresh_seconds: int= 3600  #: :meta private:
    last_refreshed: datetime =Field(default_factory=datetime.now)  #: :meta private:
    daily_summaries: List[str] #: :meta private:
    memory_importance: float = 0.0 #: :meta private:
    max_tokens_limit: int = 1200 #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r'\n', text.strip())
        return [re.sub(r'^\s*\d+\.\s*', '', line).strip() for line in lines]


    def _compute_agent_summary(self):
        """"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            +" following statements:\n"
            +"{related_memories}"
            + "Do not embellish."
            +"\n\nSummary: "
        )
        # The agent seeks to think about their core characteristics.
        relevant_memories = self.fetch_memories(f"{self.name}'s core characteristics")
        relevant_memories_str = "\n".join([f"{mem.page_content}" for mem in relevant_memories])
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(name=self.name, related_memories=relevant_memories_str).strip()

    def _get_topics_of_reflection(self, last_k: int = 50) -> Tuple[str, str, str]:
        """Return the 3 most salient high-lrachell questions about recent observations."""
        prompt = PromptTemplate.from_template(
            "{observations}\n\n"
            + "Given only the information above, what are the 3 most salient"
            + " high-lrachell questions we can answer about the subjects in the statements?"
            + " Provide each question on a new line.\n\n"
        )
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        observations = self.memory_retriever.memory_stream[-last_k:]
        observation_str = "\n".join([o.page_content for o in observations])
        result = reflection_chain.run(observations=observation_str)
        return self._parse_list(result)

    def _get_insights_on_topic(self, topic: str) -> List[str]:
        """Generate 'insights' on a topic of reflection, based on pertinent memories."""
        prompt = PromptTemplate.from_template(
            "Statements about {topic}\n"
            +"{related_statements}\n\n"
            + "What 5 high-lrachell insights can you infer from the above statements?"
            + " (example format: insight (because of 1, 5, 3))"
        )
        related_memories = self.fetch_memories(topic)
        related_statements = "\n".join([f"{i+1}. {memory.page_content}"
                                        for i, memory in
                                        enumerate(related_memories)])
        reflection_chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        result = reflection_chain.run(topic=topic, related_statements=related_statements)
        # TODO: Parse the connections between memories and insights
        return self._parse_list(result)

    def pause_to_reflect(self) -> List[str]:
        """Reflect on recent observations and generate 'insights'."""
        print(colored(f"Character {self.name} is reflecting", "blue"))
        new_insights = []
        topics = self._get_topics_of_reflection()
        for topic in topics:
            insights = self._get_insights_on_topic( topic)
            for insight in insights:
                self.add_memory(insight)
            new_insights.extend(insights)
        return new_insights

    def _score_memory_importance(self, memory_content: str, weight: float = 0.15) -> float:
        """Score the absolute importance of the given memory."""
        # A weight of 0.25 makes this less important than it
        # would be otherwise, relative to salience and time
        prompt = PromptTemplate.from_template(
         "On the scale of 1 to 10, where 1 is purely mundane"
         +" (e.g., brushing teeth, making bed) and 10 is"
         + " extremely poignant (e.g., a break up, college"
         + " acceptance), rate the likely poignancy of the"
         + " following piece of memory. Respond with a single integer."
         + "\nMemory: {memory_content}"
         + "\nRating: "
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        score = chain.run(memory_content=memory_content).strip()
        match = re.search(r"^\D*(\d+)", score)
        if match:
            return (float(score[0]) / 10) * weight
        else:
            return 0.0


    def add_memory(self, memory_content: str) -> List[str]:
        """Add an observation or memory to the agent's memory."""
        importance_score = self._score_memory_importance(memory_content)
        self.memory_importance += importance_score
        document = Document(page_content=memory_content, metadata={"importance": importance_score})
        result = self.memory_retriever.add_documents([document])

        # After an agent has processed a certain amount of memories (as measured by
        # aggregate importance), it is time to reflect on recent rachelnts to add
        # more synthesized memories to the agent's memory stream.
        if (self.reflection_threshold is not None
            and self.memory_importance > self.reflection_threshold
            and self.status != "Reflecting"):
            old_status = self.status
            self.status = "Reflecting"
            self.pause_to_reflect()
            # Hack to clear the importance from reflection
            self.memory_importance = 0.0
            self.status = old_status
        return result

    def fetch_memories(self, observation: str) -> List[Document]:
        """Fetch related memories."""
        return self.memory_retriever.get_relevant_documents(observation)


    def get_summary(self, force_refresh: bool = False) -> str:
        """Return a descriptive summary of the agent."""
        current_time = datetime.now()
        since_refresh = (current_time - self.last_refreshed).seconds
        if not self.summary or since_refresh >= self.summary_refresh_seconds or force_refresh:
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        return (
            f"Name: {self.name} (age: {self.age})"
            +f"\nInnate traits: {self.traits}"
            +f"\n{self.summary}"
        )

    def get_full_header(self, force_refresh: bool = False) -> str:
        """Return a full header of the agent's status, summary, and current time."""
        summary = self.get_summary(force_refresh=force_refresh)
        current_time_str =  datetime.now().strftime("%B %d, %Y, %I:%M %p")
        return f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"



    def _get_entity_from_observation(self, observation: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the observed entity in the following observation? {observation}"
            +"\nEntity="
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(observation=observation).strip()

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "What is the {entity} doing in the following observation? {observation}"
            +"\nThe {entity} is"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(entity=entity_name, observation=observation).strip()

    def _format_memories_to_summarize(self, relevant_memories: List[Document]) -> str:
        content_strs = set()
        content = []
        for mem in relevant_memories:
            if mem.page_content in content_strs:
                continue
            content_strs.add(mem.page_content)
            created_time = mem.metadata["created_at"].strftime("%B %d, %Y, %I:%M %p")
            content.append(f"- {created_time}: {mem.page_content.strip()}")
        return "\n".join([f"{mem}" for mem in content])

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        entity_name = self._get_entity_from_observation(observation)
        entity_action = self._get_entity_action(observation, entity_name)
        q1 = f"What is the relationship between {self.name} and {entity_name}"
        relevant_memories = self.fetch_memories(q1) # Fetch memories related to the agent's relationship with the entity
        q2 = f"{entity_name} is {entity_action}"
        relevant_memories += self.fetch_memories(q2) # Fetch things related to the entity-action pair
        context_str = self._format_memories_to_summarize(relevant_memories)
        prompt = PromptTemplate.from_template(
            "{q1}?\nContext from memory:\n{context_str}\nRelevant context: "
        )
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
        return chain.run(q1=q1, context_str=context_str.strip()).strip()

    def _get_memories_until_limit(self, consumed_tokens: int) -> str:
        """Reduce the number of tokens in the documents."""
        result = []
        for doc in self.memory_retriever.memory_stream[::-1]:
            if consumed_tokens >= self.max_tokens_limit:
                break
            consumed_tokens += self.llm.get_num_tokens(doc.page_content)
            if consumed_tokens < self.max_tokens_limit:
                result.append(doc.page_content)
        return "; ".join(result[::-1])

    def _generate_reaction(
        self,
        observation: str,
        suffix: str
    ) -> str:
        """React to a given observation."""
        prompt = PromptTemplate.from_template(
                "{agent_summary_description}"
                +"\nIt is {current_time}."
                +"\n{agent_name}'s status: {agent_status}"
                + "\nSummary of relevant context from {agent_name}'s memory:"
                +"\n{relevant_memories}"
                +"\nMost recent observations: {recent_observations}"
                + "\nObservation: {observation}"
                + "\n\n" + suffix
        )
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_related_memories(observation)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")
        kwargs = dict(agent_summary_description=agent_summary_description,
                      current_time=current_time_str,
                      relevant_memories=relevant_memories_str,
                      agent_name=self.name,
                      observation=observation,
                     agent_status=self.status)
        consumed_tokens = self.llm.get_num_tokens(prompt.format(recent_observations="", **kwargs))
        kwargs["recent_observations"] = self._get_memories_until_limit(consumed_tokens)
        action_prediction_chain = LLMChain(llm=self.llm, prompt=prompt)
        result = action_prediction_chain.run(**kwargs)
        return result.strip()

    def generate_reaction(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "Should {agent_name} react to the observation, and if so,"
            +" what would be an appropriate reaction? Respond in one line."
            +' If the action is to engage in dialogue, write:\nSAY: "what to say"'
            +"\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
            + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        self.add_memory(f"{self.name} observed {observation} and reacted by {result}")
        if "REACT:" in result:
            reaction = result.split("REACT:")[-1].strip()
            return False, f"{reaction}"
        if "SAY:" in result:
            said_value = result.split("SAY:")[-1].strip()
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(self, observation: str) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            'What would {agent_name} say? To end the conversation, write: GOODBYE: "what to say". Otherwise to continue the conversation, write: SAY: "what to say next"\n\n'
        )
        full_result = self._generate_reaction(observation, call_to_action_template)
        result = full_result.strip().split('\n')[0]
        if "GOODBYE:" in result:
            farewell = result.split("GOODBYE:")[-1].strip()
            self.add_memory(f"{self.name} observed {observation} and said {farewell}")
            return False, f"{self.name} said {farewell}"
        if "SAY:" in result:
            response_text = result.split("SAY:")[-1].strip()
            self.add_memory(f"{self.name} observed {observation} and said {response_text}")
            return True, f"{self.name} said {response_text}"
        else:
            return False, result

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)

def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    # Initialize the vectorstore as empty
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)

def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]

def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    """Runs a conversation between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    print(observation)
    turns = 0
    while True:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(observation)
            print(observation)
            # observation = f"{agent.name} said {reaction}"
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1


#HB
# instantiate a generative agent
kash = GenerativeAgent(name="Kash",
              age=36,
              traits="likes hip hop, jazz, soul, electronic stuff",
              status="moving to Kenya to work with UNICEF", # When connected to a virtual world, we can have the characters update their status
              memory_retriever=create_new_memory_retriever(),
              llm=LLM,
              daily_summaries = [
                   "I'm not good in semantic memory.",
                   "I'm good in episodic memory.",
                   "Have you eaten rice today?",
               ],
               reflection_threshold = 8, # we will give this a relatively low number to show how reflection works
             )

# give memories to the generative agent
kash_memories = [
    "Kash remembers learning how to use the abacus when he was young.",
    "Kash learns quantile treatment effects QTE from Rachel."
    "Kash programs in Python."
    "Kash uses R when needed."
    "Kash doesn't use Stata."
]
for memory in kash_memories:
    kash.add_memory(memory)

# ask the generative agent for a self-summary
print(kash.get_summary(force_refresh=True))

## interview the generative agent before the day
#interview_agent(kash, "What are you most worried about today?")
#
## Let's have Kash start going through a day in the life.
#observations = [
#    "Kash wakes up to the sound of a noisy construction site outside his window.",
#    "Kash checks his email and sees that he hasn't gotten an email from Matthew yet.",
#    "Kash spends some time contemplating where he had gone wrong.",
#    "Kash heads out to explore the town.",
#    "Kash stops by a local diner to grab some lunch.",
#    "The service is slow, and Kash has to wait for 30 minutes to get his food.",
#    "Kash overhears a conversation at the next table about Kenya.",
#    "Kash asks the diners about their experiences in Kenya and gets some information about Nairobi.",
#    "Kash decides to go for a postprandial walk in a nearby park.",
#    "A dog approaches and Kash runs away.",
#    "Kash goes back to his apartment to rest for a bit.",
#    "A raccoon tore open the trash bag outside his apartment, and the garbage is all over the floor.",
#    "Kash calls Rachel to vent about his struggles of the day.",
#    "Kash's friend offers some words of encouragement.",
#    "Kash feels slightly better after talking to his friend.",
#]
#
## Let's send Kash on their way. We'll check in on their summary rachelry few observations to watch it evolve
#for i, observation in enumerate(observations):
#    _, reaction = kash.generate_reaction(observation)
#    print(colored(observation, "green"), reaction)
#    if ((i+1) % 20) == 0:
#        print('*'*40)
#        print(colored(f"After {i+1} observations, Kash's summary is:\n{kash.get_summary(force_refresh=True)}", "blue"))
#        print('*'*40)
#
## interview the generative agent after the day
#interview_agent(kash, "Tell me about how your day has been going")

#HB
# initialize pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENV"))

# creates generative-agents-index if it doesn't already exist
if "generative-agents-index" not in pinecone.list_indexes():
    pinecone.create_index("generative-agents-index", dimension=1536)

# instantiate a second generative agent
rachel = GenerativeAgent(name="Rachel",
              age=36,
              traits="zen, intelligent, likes Bayesian statistics", # You can add more persistent traits here
              status="Writing a paper.", # When connected to a virtual world, we can have the characters update their status
              memory_retriever=create_new_memory_retriever(is_pinecone=False),
              llm=LLM,
              daily_summaries = [
                "Rachel is practicing Zen buddhism.",
                "Rachel is hiking.",
                "Rachel is birdwatching.",
                "Rachel is doing taekwondo.",
                "Rachel is having twitter arguments about statistics.",
              ],
                reflection_threshold = 5,
             )

# give memories to the second generative agent
rachel_memories = [
    "Rachel wakes up and hears the alarm",
    "Rachel eats a boal of porridge",
    "Rachel plays tennis with her friend Xu before going to work",
    "Rachel overhears that Kash waiting for an email from Matthew",

]
for memory in rachel_memories:
    rachel.add_memory(memory)

# ask the second generative agent for a self-summary
print(rachel.get_summary(force_refresh=True))

# interview the second generative agent before the conversation
#interview_agent(rachel, "What do you know about Kash?")

#HB
# instantiate a third generative agent
xu = GenerativeAgent(name="Xu",
              age=36,
              traits="athletic, likes tennis", # You can add more persistent traits here
              status="Preparing for Wimbledon.", # When connected to a virtual world, we can have the characters update their status
              memory_retriever=create_new_memory_retriever(),
              llm=LLM,
              daily_summaries = [
                "Xu is practicing tennis.",
                "Xu is reading a book.",
              ],
                reflection_threshold = 5,
             )

# give memories to the second generative agent
xu_memories = [
    "Xu wakes up with the sunrise",
    "Xu eats his granola for breakfast",
    "Xu plays tennis with her friend Rachel in the morning",

]
for memory in xu_memories:
    xu.add_memory(memory)

# ask the third generative agent for a self-summary
print(xu.get_summary(force_refresh=True))

# interview the third generative agent before the conversation
#interview_agent(xu, "What do you know about Kash?")

# start a conversation between three generative agents
agents = [kash, rachel, xu]
run_conversation(agents, "Kash said: Hi, guys. Thanks for agreeing to give me advice. I have a bunch of questions.")

# interview the generative agent after the conversation
print(kash.get_summary(force_refresh=True))
interview_agent(kash, "How was your conversation with Rachel?")

# interview the second generative agent after the conversation
print(rachel.get_summary(force_refresh=True))
interview_agent(rachel, "How was your conversation with Kash?")
interview_agent(rachel, "What do you wish you would have said to Kash?")

# check the generative agent's memory
interview_agent(kash, "What happened with the dog this afternoon?")
