import datetime
import json
import markdown
import time
import vecs
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from enum import Enum
from loguru import logger
from pathlib import Path

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from sentence_transformers import SentenceTransformer

from .prompts.segment_template import SEGMENT_SYSTEM_TEMPLATE, SEGMENT_HUMAN_TEMPLATE
from .prompts.final_answer_template import FINAL_ANSWER_SYSTEM_TEMPLATE, FINAL_ANSWER_HUMAN_TEMPLATE
from ...db.models import (
    QuestionsHubermanLab,
    AnswersHubermanLab,
    ResourcesHubermanLab,
    RecommendedResourcesHubermanLab
)

css = """
<style>
body {
    font-family: 'Helvetica', sans-serif;
    margin: 0 auto;
    max-width: 800px;
    padding: 2em;
    color: #444444;
    line-height: 1.6;
    background-color: #F9F9F9;
    box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
}

h1, h2 {
    color: #383838;
}

h1 {
    text-align: center;
    border-bottom: 2px solid #3F51B5;
    margin-bottom: 1em;
    padding-bottom: 0.5em;
}

iframe {
    display: block;
    margin: 2em auto;
    border: 1px solid #D3D3D3;
    box-shadow: 2px 2px 15px rgba(0,0,0,0.1);
}

ol {
    padding-left: 1em;
}

li {
    margin-bottom: 0.5em;
}

li:last-child {
    margin-bottom: 0;
}

a {
    color: #3F51B5;
    text-decoration: none;
}

a:hover {
    color: #303F9F;
}

</style>
"""


class EmbeddingModel(Enum):
    SBERT = "sbert"
    OPENAI = "openai"


class PostgresQAEngine:
    def __init__(self,
                 embedding_model: EmbeddingModel,
                 sql_session: Session,
                 vecs_client: vecs.Client,
                 vecs_collection_name: str,
                 n_search: int = 20,
                 n_relevant_segments: int = 3,
                 llm_model: str = 'gpt-3.5-turbo',
                 temperature: float = 0) -> None:

        if embedding_model.value == EmbeddingModel.SBERT.value:
            self.embedding_model = SentenceTransformer(
                'app/models/all-MiniLM-L6-v2')
            self.embedding_ndim = self.embedding_model.get_sentence_embedding_dimension()
        else:
            raise NotImplementedError

        self.session = sql_session
        self.vx = vecs_client
        self.docs = self.vx.get_or_create_collection(
            name=vecs_collection_name, dimension=self.embedding_ndim)

        self.n_search = n_search
        self.n_relevant_segments = n_relevant_segments
        self.llm_model = llm_model
        self.temperature = temperature

    def segment_check_and_answer(self, question: str, context: str) -> str:
        chat = ChatOpenAI(temperature=self.temperature,
                          model_name=self.llm_model)

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            SEGMENT_SYSTEM_TEMPLATE)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            SEGMENT_HUMAN_TEMPLATE)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])

        chain = LLMChain(llm=chat, prompt=chat_prompt)
        output = chain.run(question=question, context=context)

        return output

    def embed_question(self, question) -> list:
        if isinstance(self.embedding_model, SentenceTransformer):
            return self.embedding_model.encode(question).tolist()

        elif isinstance(self.embedding_model, OpenAIEmbeddings):
            return self.embedding_model.embed_query(question)

    def search_segments(self, embedded_question: list, n: int) -> dict:
        """
        Returns segment ids with cosine similarity, e.g.
        {
            65: 0.82,
            189: 0.78,
            921: 0.77
        }
        The values are sorted in a descending order.
        """
        query_results = self.docs.query(
            data=embedded_question,
            limit=n,
            filters={},
            measure="cosine_distance",
            include_value=True,
            include_metadata=False
        )
        indices = list(map(int, [el[0] for el in query_results]))
        cos_similarity = [1-el[1] for el in query_results]
        indices_dict = dict(zip(indices, cos_similarity))
        return indices_dict

    def process_found_segments(self, question: str, indices: dict):
        n_relevant_segments = 0
        n_non_relevant_segments = 0
        relevant_summaries = {}

        for ind in indices:
            if n_relevant_segments < self.n_relevant_segments:
                # Query context via SQL using SQLAlchemy ORM
                resource = self.session.query(ResourcesHubermanLab).get(ind)

                context = resource.summary
                answer = self.segment_check_and_answer(
                    question=question, context=context)

                if answer.startswith("Not relevant"):
                    n_non_relevant_segments += 1
                else:
                    n_relevant_segments += 1
                    relevant_summaries[int(ind)] = {"answer": answer,
                                                    "summary": resource.summary,
                                                    "episode_name": resource.episode_name,
                                                    "segment_title": resource.segment_title,
                                                    "url": resource.url,
                                                    "topic": resource.topic
                                                    }

            else:
                return relevant_summaries, n_relevant_segments, n_non_relevant_segments

        return relevant_summaries, n_relevant_segments, n_non_relevant_segments

    def get_final_answer(self, question: str, answers: dict):
        chat = ChatOpenAI(temperature=self.temperature,
                          model_name=self.llm_model)

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            FINAL_ANSWER_SYSTEM_TEMPLATE)
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            FINAL_ANSWER_HUMAN_TEMPLATE)
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt])

        prompt_context = "\n".join(
            [f"ANSWER {i+1}:\n{answer['answer']}\n" for i, answer in enumerate(answers.values())])

        chain = LLMChain(llm=chat, prompt=chat_prompt)
        output = chain.run(question=question, context=prompt_context)
        return output

    def answer_full_flow(self, user_id, question):
        start_time = time.time()
        # Create a new question instance
        new_question = QuestionsHubermanLab(
            user_id=user_id, created_at=datetime.datetime.now(), question=question, mode="answer")

        # Add the question instance to the session and commit
        self.session.add(new_question)
        self.session.commit()

        question_id = new_question.id

        # Encoding question to embedding space
        embedded_question = self.embed_question(question)

        # Finding relevant segments
        indices = self.search_segments(embedded_question, 20)

        # Create new recommended resource instances and add them to the session
        for resource_id, similarity_score in indices.items():
            new_recommended_resource = RecommendedResourcesHubermanLab(
                question_id=question_id, resource_id=resource_id, similarity_score=similarity_score)
            self.session.add(new_recommended_resource)
        self.session.commit()

        # Getting answers from segments
        relevant_segments, n_relevant, n_non_relevant = self.process_found_segments(
            question, indices)

        # Getting the final answer
        answer = self.get_final_answer(question, relevant_segments)
        answer += "\n\n## Related Videos\n\n"
        for i, value in enumerate(relevant_segments.values()):
            answer += f'\n{i+1}. {value["segment_title"]}\n <iframe width="770" height="400" src="{value["url"].replace("watch?v=", "embed/").replace("&t=", "?start=")[:-1]}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\n'

        # Adding the answer to the AnswersHubermanLab table
        new_answer = AnswersHubermanLab(
            user_id=user_id, question_id=question_id, answer=answer, n_relevant=n_relevant, n_non_relevant=n_non_relevant)
        self.session.add(new_answer)
        self.session.commit()

        html_raw = markdown.markdown(answer, extensions=['extra'])

        end_time = time.time()

        logger.info(f"Full QA flow time: {round(end_time-start_time, 2)}")
        return answer, html_raw

    def resource_full_flow(self, user_id, question):
        start_time = time.time()

        # Create a new question instance
        new_question = QuestionsHubermanLab(
            user_id=user_id, created_at=datetime.datetime.now(), question=question, mode="resources")

        # Add the question instance to the session and commit
        self.session.add(new_question)
        self.session.commit()

        question_id = new_question.id

        # Encoding question to embedding space
        embedded_question = self.embed_question(question)

        # Finding relevant segments
        indices = self.search_segments(embedded_question, 7)

        # Create new recommended resource instances and add them to the session
        for resource_id, similarity_score in indices.items():
            new_recommended_resource = RecommendedResourcesHubermanLab(
                question_id=question_id, resource_id=resource_id, similarity_score=similarity_score)
            self.session.add(new_recommended_resource)
        self.session.commit()

        output = "\n\n## Related Videos\n\n"
        for i, ind in enumerate(indices, 1):
            resource = self.session.query(ResourcesHubermanLab).get(ind)
            output += f'\n{i+1}. {resource.segment_title}\n <iframe width="770" height="400" src="{resource.url.replace("watch?v=", "embed/").replace("&t=", "?start=")[:-1]}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>\n'

        html_raw = markdown.markdown(output, extensions=['extra'])
        end_time = time.time()

        logger.info(f"Resource flow time: {round(end_time-start_time, 2)}")
        return html_raw
