import os
import json
import xml.etree.ElementTree as ET
import faiss
import pickle
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings

# from dotenv import load_dotenv


class JobMatchingBaseline:
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.embedder = HuggingFaceEmbeddings()
        # load_dotenv()  # Load environment variables from .env file
        self.embeddings = None
        self.index = None
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

    def create_embeddings(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        strings = []
        for obj in data:
            string = json.dumps(obj)
            strings.append(string)

        doc_result = self.embedder.embed_documents(strings)
        self.embeddings = doc_result

        index = faiss.index_factory(len(doc_result[0]), "Flat")
        index.train(doc_result)
        index.add(doc_result)
        self.index = index

        return index, strings

    def create_embedding_index(self):
        index = faiss.index_factory(len(self.embeddings[0]), "Flat")
        index.train(self.embeddings)
        index.add(self.embeddings)
        self.index = index

    def match_jobs(self, query, openai_key, k=5):
        query_result = self.embedder.embed_query(query)
        query_result = np.array(query_result)
        distances, neighbors = self.index.search(
            query_result.reshape(1, -1).astype(np.float32), k
        )

        scores = [distance for distance in distances[0]]
        # Normalize scores to be between 0 and 100
        scores = [100 * (1 - score / max(scores)) for score in scores]

        return (scores, [self.strings[neighbor] for neighbor in neighbors[0]])

    def save_embeddings(
        self,
        saving_embeddings_file_name: str,
        saving_embeddings_directory: str,
    ) -> None:
        directory = os.path.join(os.getcwd(), saving_embeddings_directory)
        print(directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, saving_embeddings_file_name + ".pkl")

        # Save embeddings to binary file
        with open(file_path, "wb") as f:
            pickle.dump(self.embeddings, f)

    def load_embeddings(self, embeddings_path) -> HuggingFaceEmbeddings:
        print("CALLED")
        with open(embeddings_path, "rb") as f:
            embeddings: HuggingFaceEmbeddings = pickle.load(f)

        print(type(embeddings))
        self.embeddings = embeddings

        with open("job_description_embedding/job_openings.json", "r") as f:
            strings = json.load(f)
        self.strings = strings


if __name__ == "__main__":
    engine = JobMatchingBaseline(None)
    engine.create_embeddings("job_description_embedding/job_openings.json")
    engine.save_embeddings(
        "saved_embeddings",
        "job_description_embedding/embeddings",
    )
