import os
import json
import xml.etree.ElementTree as ET
import faiss
import pickle
import numpy as np
import collections
from langchain.embeddings import HuggingFaceEmbeddings

import cv_parsing.ResumeParser as ResumeParser


class JobMatchingFineGrained:
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.embedder = HuggingFaceEmbeddings()
        self.indexes = None
        self.strings = None

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        jobs_list = []
        for job in root.findall("job"):
            job_dict = {
                "title": job.find("title").text,
                "company": job.find("company").text,
                "posted_date": job.find("posted_date").text,
                "job_reference": job.find("job_reference").text,
                "req_number": job.find("req_number").text,
                "url": job.find("url").text,
                "body": job.find("body").text,
                "city": job.find("city").text,
                "state": job.find("state").text,
                "country": job.find("country").text,
                "location": job.find("location").text,
                "function": job.find("function").text,
                "logo": job.find("logo").text,
                "jobtype": job.find("jobtype").text,
                "education": job.find("education").text,
                "experience": job.find("experience").text,
                "salary": job.find("salary").text,
                "requiredlanguages": job.find("requiredlanguages").text,
                "requiredskills": job.find("requiredskills").text,
            }
            jobs_list.append(job_dict)

        return jobs_list

    def xml_to_json(self, xml_file, json_output_file):
        jobs_list = self.parse_xml(xml_file)
        json_output = json.dumps(jobs_list, indent=4)

        with open(json_output_file, "w") as json_file:
            json_file.write(json_output)

    def create_embeddings(self, json_file, save=True):
        with open(json_file, "r") as f:
            data = json.load(f)

        with open("job_description_embedding/job_openings_completed.json", "r") as f:
            strings = json.load(f)
            self.strings = strings

        new_data = []
        for obj in data:
            # Useful fields: title, company, body, city, state, country, location, function, jobtype, education, experience, requiredlanguages, requiredskills
            new_obj = {
                "job": " ".join(
                    [
                        i
                        for i in [obj["jobtype"], obj["function"], obj["function"]]
                        if i is not None
                    ]
                ),
                "location": " ".join(
                    [
                        i
                        for i in [
                            obj["location"],
                            obj["city"],
                            obj["state"],
                            obj["country"],
                        ]
                        if i is not None
                    ]
                ),
                "company": obj["company"] if obj["company"] is not None else "",
                "body": obj["body"] if obj["body"] is not None else "",
                "education": " ".join(
                    [str(obj["education"])]
                    if not isinstance(obj["education"], list)
                    else obj["education"]
                )
                if obj["education"] is not None
                else "",
                "experience": " ".join(
                    [str(obj["experience"])]
                    if not isinstance(obj["experience"], list)
                    else obj["experience"]
                )
                if obj["experience"] is not None
                else "",
                "requiredLanguages": " ".join(
                    [str(obj["requiredLanguages"])]
                    if not isinstance(obj["requiredLanguages"], list)
                    else obj["requiredLanguages"]
                )
                if "requiredLanguages" in obj and obj["requiredLanguages"] is not None
                else "",
                "requiredSkills": " ".join(
                    [str(obj["requiredskills"])]
                    if not isinstance(obj["requiredSkills"], list)
                    else obj["requiredSkills"]
                )
                if "requiredSkills" in obj and obj["requiredSkills"] is not None
                else "",
            }
            new_data.append(new_obj)

        jobs = self.embedder.embed_documents([obj["job"] for obj in new_data])
        locations = self.embedder.embed_documents([obj["location"] for obj in new_data])
        companies = self.embedder.embed_documents([obj["company"] for obj in new_data])
        bodies = self.embedder.embed_documents([obj["body"] for obj in new_data])
        educations = self.embedder.embed_documents(
            [obj["education"] for obj in new_data]
        )
        experiences = self.embedder.embed_documents(
            [obj["experience"] for obj in new_data]
        )
        requiredlanguages = self.embedder.embed_documents(
            [obj["requiredLanguages"] for obj in new_data]
        )
        requiredskills = self.embedder.embed_documents(
            [obj["requiredSkills"] for obj in new_data]
        )

        # Save embeddings
        keys = list(new_data[0].keys())
        if save:
            directory = os.path.join(
                os.getcwd(), "job_description_embedding/embeddings/fine_grained"
            )
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i, embeds in enumerate(
                [
                    jobs,
                    locations,
                    companies,
                    bodies,
                    educations,
                    experiences,
                    requiredlanguages,
                    requiredskills,
                ]
            ):
                file_path = os.path.join(directory, keys[i] + ".pkl")
                # Save embeddings to binary file
                with open(file_path, "wb") as f:
                    pickle.dump(embeds, f)

        # Create indexes
        print("Creating indexes...")
        indexes = {}
        for i, embeds in enumerate(
            [
                jobs,
                locations,
                companies,
                bodies,
                educations,
                experiences,
                requiredlanguages,
            ]
        ):
            index = faiss.index_factory(len(embeds[0]), "Flat")
            index.train(np.array(embeds))
            index.add(np.array(embeds))
            indexes[keys[i]] = index

        self.indexes = indexes

        return indexes, strings

    def load_embeddings(self):
        indexes = {}
        for key in [
            "job",
            "location",
            "company",
            "body",
            "education",
            "experience",
            "requiredLanguages",
            "requiredSkills",
        ]:
            directory = os.path.join(
                os.getcwd(), "job_description_embedding/embeddings/fine_grained"
            )
            file_path = os.path.join(directory, key + ".pkl")
            with open(file_path, "rb") as f:
                embeddings = pickle.load(f)

            index = faiss.index_factory(len(embeddings[0]), "Flat")
            index.train(np.array(embeddings))
            index.add(np.array(embeddings))
            indexes[key] = index
        self.indexes = indexes

        with open("job_description_embedding/job_openings_completed.json", "r") as f:
            strings = json.load(f)
        self.strings = strings

    def match_jobs(self, query, openai_key, k=5):
        p = ResumeParser.ResumeParser(openai_key)
        cv_json = p.query_resume(query)
        # cv_json = self.__cv_to_json(query)
        cv_meta_json = {
            "job": " ".join(
                [
                    ("" if el["job_title"] is None else (el["job_title"] + " "))
                    + ("" if el["job_summary"] is None else el["job_summary"])
                    for el in cv_json["work_experience"]
                ]
            ),
            "location": ""
            if cv_json["basic_info"]["location"] is None
            else cv_json["basic_info"]["location"],
            "company": " ".join([el["company"] for el in cv_json["work_experience"]]),
            "body": query,
            "education": ""
            if cv_json["basic_info"]["university"] is None
            and cv_json["basic_info"]["education_level"] is None
            and cv_json["basic_info"]["majors"] is None
            else " ".join(
                [
                    cv_json["basic_info"]["university"],
                    cv_json["basic_info"]["education_level"],
                    " ".join(cv_json["basic_info"]["majors"])
                    if cv_json["basic_info"]["majors"] is not None
                    else "",
                ]
            ),
            "experience": " ".join(
                [
                    ("" if el["job_title"] is None else (el["job_title"] + " "))
                    + ("" if el["job_summary"] is None else el["job_summary"])
                    for el in cv_json["work_experience"]
                ]
            ),
            "requiredLanguages": " ".join(
                ""
                if "languages" not in cv_json["basic_info"]
                or cv_json["basic_info"]["languages"] is None
                or len(cv_json["basic_info"]["languages"]) == 0
                else cv_json["basic_info"]["languages"]
            ),
            "requiredSkills": "".join(
                [
                    "" if el["job_summary"] is None else el["job_summary"]
                    for el in cv_json["work_experience"]
                ]
                if "skills" not in cv_json["basic_info"]
                or cv_json["basic_info"]["skills"] is None
                or len(cv_json["basic_info"]["skills"]) == 0
                else " ".join(cv_json["basic_info"]["skills"])
            ),
        }

        print(cv_meta_json)

        posting_scores = {}
        for key in cv_meta_json.keys():
            index = self.indexes[key]
            query = np.array(self.embedder.embed_query(cv_meta_json[key]))
            dists, neighbors = index.search(
                query.reshape(1, -1).astype(np.float32), 1000
            )
            scores = [distance for distance in dists[0]]
            # Normalize scores to be between 0 and 100
            if max(scores) == 0:
                scores = [0 for score in scores]
            else:
                scores = [
                    100 * (1 - score / max(scores)) if score != 0 else 0
                    for score in scores
                ]
            for ind, neighbor_id in enumerate(neighbors[0]):
                if neighbor_id not in posting_scores:
                    posting_scores[neighbor_id] = {}
                posting_scores[neighbor_id][key] = scores[ind]

        weighted_scores = {}
        for key, scores in posting_scores.items():
            weighted_scores[key] = (
                scores["job"] * 0.1
                + scores["location"] * 0.05
                + scores["company"] * 0.05
                + scores["body"] * 0.1  # used to be 0.1
                + scores["education"] * 0.2
                + scores["experience"] * 0.2
                + scores["requiredLanguages"] * 0.1
                + scores["requiredSkills"] * 0.2
            )

        sorted_keys = sorted(weighted_scores, key=weighted_scores.get, reverse=True)
        sorted_scores = [weighted_scores[key] for key in sorted_keys]
        return (
            sorted_scores[:k],
            [self.strings[neighbor] for neighbor in sorted_keys][:k],
        )


if __name__ == "__main__":
    engine = JobMatchingFineGrained(None)
    embeddings = engine.create_embeddings(
        "job_description_embedding/job_openings_completed.json"
    )
    engine.load_embeddings()
