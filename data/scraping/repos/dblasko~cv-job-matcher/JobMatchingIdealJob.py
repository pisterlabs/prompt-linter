import os
import json
import hashlib
import re

from json.decoder import JSONDecodeError
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models.base import BaseChatModel
from langchain import PromptTemplate
import numpy as np

from job_description_embedding.JobMatchingBaseline import JobMatchingBaseline
from job_description_embedding.printer import eprint


# from dotenv import load_dotenv


class JobMatchingIdealJob(JobMatchingBaseline):
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: HuggingFaceEmbeddings = None,
        cache_dir: str = ".query_cache",
        ideal_job_fields={
            "title": None,
            "company": "appropriate company or industry",
            "body": "brief job summary",
            "city": None,
            "state": None,
            "country": None,
            "location": "from candidate's preference or CV",
            "function": None,
            "jobtype": None,
            "education": None,
            "experience": None,
            "salary": None,
            "requiredlanguages": "from CV",
            "requiredskills": "from CV",
        },
        job_fields=[
            "title",
            "company",
            "posted_date",
            "job_reference",
            "req_number",
            "url",
            "body",
            "city",
            "state",
            "country",
            "location",
            "function",
            "logo",
            "jobtype",
            "education",
            "experience",
            "salary",
            "requiredlanguages",
            "requiredskills",
        ],
    ):
        super().__init__(embeddings=embeddings)
        self.llm = llm
        self.cache_dir = cache_dir
        self.sha256 = hashlib.sha256
        self.job_fields = job_fields

        self.prompt = PromptTemplate.from_template(
            "Analyze the following CV and transform the extracted information into an ideal job description for the candidate,"
            + " assuming they are seeking to switch jobs or secure a new position. The output should be a valid JSON object that could be parsed by json.loads()."
            + " Include: "
            + ", ".join(f"{k} ({v})" if v else k for k, v in ideal_job_fields.items())
            + "."
            + " Remember to use the information available in the CV, along with your general knowledge about the world, job markets, and companies, to make informed choices for each field."
            + " If a field cannot be filled based on the information, set it to null. Please respond with just the JSON object. CV content: {cv}"
        )

    def match_jobs(self, query, openai_key, k=5):
        print("HERE")
        query_d = self._get_ideal_job(query=query)
        if query_d is None:
            return (None, [])

        query_d = dict(
            {k: None for k in self.job_fields if k not in query_d}, **query_d
        )
        query_result = self.embedder.embed_query(json.dumps(query_d))
        query_result = np.array(query_result)
        distances, neighbors = self.index.search(
            query_result.reshape(1, -1).astype(np.float32), k
        )

        scores = [distance for distance in distances[0]]
        # Normalize scores to be between 0 and 100
        scores = [100 * (1 - score / max(scores)) for score in scores]

        return (scores, [self.strings[neighbor] for neighbor in neighbors[0]])

    def _parse_json(self, response) -> dict | None:
        try:
            return json.loads(
                re.sub(r"(?<=\w)\n(?=\w)", "\\\\n", response.generations[0][0].text)
            )
        except JSONDecodeError:
            eprint("Couldn't parse:", response.generations[0][0].text)
            return None

    def _get_ideal_job(self, query: str) -> dict | None:
        directory = os.path.join(os.getcwd(), self.cache_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
        query_hash = self.sha256(query.encode("utf-8")).hexdigest()
        file_path = os.path.join(directory, f"ideal_job_cv-{query_hash}" + ".json")

        if not os.path.exists(file_path):
            try:
                prompt = self.prompt.format_prompt(cv=query)
                ideal_job = self._parse_json(
                    self.llm.generate(messages=[prompt.to_messages()])
                )
                if ideal_job is not None:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(ideal_job, f)
            except Exception as err:
                print("got exception:", err)
                return None
        if os.path.exists(file_path):
            print("la")
            with open(file_path, "r", encoding="utf-8") as j:
                return json.load(j)

        return None
