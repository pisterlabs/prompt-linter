import kuzu
import os
import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http import models
from langchain.embeddings import HuggingFaceEmbeddings
load_dotenv()


def log_to_file(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as log_file:
        log_file.write(f"{timestamp} - {message}\n")


qdrant_api_key = os.environ.get('QDRANT_API_KEY')
cohere_api_key = os.environ.get('COHERE_API_KEY')


class database():
    def __init__(self) -> None:
        db = kuzu.Database('./classDB')
        self.conn = kuzu.Connection(db)
        self.qdrant_client = QdrantClient(
            url="https://b4de9d01-9729-4a94-b1f6-d13df2fa6fd0.us-east4-0.gcp.cloud.qdrant.io:6333",
            api_key=qdrant_api_key,
        )
        self.qdrant_client.recreate_collection(
            collection_name="kuzuEmbeddings",
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE))
        self.embedding_client = HuggingFaceEmbeddings()
        # self.cohere_client = CohereEmbeddings(cohere_api_key=cohere_api_key)

    def createTables(self) -> None:
        self.conn.execute(
            "CREATE NODE TABLE Class(id STRING, course_id STRING, course_code STRING, title STRING, grade INT64, associatedAcademicGroupCode STRING, associatedAcademicCareer STRING, description STRING, PRIMARY KEY (id))")
        self.conn.execute(
            "CREATE REL TABLE sibling(FROM Class TO Class, weight STRING)")
        self.conn.execute("CREATE REL TABLE RelatedTo (FROM Class TO Class)")
        self.conn.execute(
            "CREATE REL TABLE prerequisite (FROM Class TO Class)")

    def addClass(self, class_object) -> None:
        grade = int(class_object['catalogNumber'][0]
                    ) if class_object['catalogNumber'][0].isdigit() else -1
        params = {
            "id": f"{class_object['courseId']}{class_object['subjectCode']}{class_object['catalogNumber']}",
            "course_id": class_object['courseId'],
            "course_code": f"{class_object['subjectCode']}{class_object['catalogNumber']}",
            "title": class_object['title'],
            "grade": grade,
            "associatedAcademicGroupCode": class_object['associatedAcademicGroupCode'],
            "associatedAcademicCareer": class_object['associatedAcademicCareer'],
            "description": class_object['description']
        }
        query = """
        CREATE(n:Class {
            id: $id,
            course_id: $course_id,
            title: $title,
            course_code: $course_code,
            grade: $grade,
            associatedAcademicGroupCode: $associatedAcademicGroupCode,
            associatedAcademicCareer: $associatedAcademicCareer,
            description: $description
        })
        RETURN n;
        """
        # try:
        self.conn.execute(query, params)
        print(f"Added class {class_object['title']} to database")
        # except:
        #     print(f"Error adding class {class_object['title']} to database")
        #     log_to_file(str(class_object))
        #     return

    def addPoints(self, points) -> None:
        operation_info = self.qdrant_client.upsert(
            collection_name="kuzuEmbeddings",
            wait=True,
            points=points
        )
