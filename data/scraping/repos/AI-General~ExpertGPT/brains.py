import os
import re
import time
from typing import Any, List, Optional
from uuid import UUID

import openai
from logger import get_logger
from pydantic import BaseModel
from qdrant_client import QdrantClient
from supabase.client import Client

from models.annotation import Annotation, AnnotationMessage
from models.databases.supabase.supabase import SupabaseDB
from models.databases.qdrant.qdrant import QdrantDB
from models.settings import BrainRateLimiting, get_supabase_client, get_supabase_db, get_qdrant_client, get_qdrant_db
from utils.vectors import get_unique_files_from_vector_ids


logger = get_logger(__name__)


class Brain(BaseModel):
    id: Optional[UUID] = None
    name: Optional[str] = "Default brain"
    description: Optional[str] = "This is a description"
    status: Optional[str] = "private"
    # model: Optional[str] = "gpt-3.5-turbo-0613"
    # temperature: Optional[float] = 0.0
    # max_tokens: Optional[int] = 256
    # openai_api_key: Optional[str] = None
    files: List[Any] = []
    datas: List[Any] = []
    max_brain_size = BrainRateLimiting().max_brain_size
    prompt_id: Optional[UUID] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def supabase_client(self) -> Client:
        return get_supabase_client()

    @property
    def supabase_db(self) -> SupabaseDB:
        return get_supabase_db()

    @property
    def qdrant_client(self) -> QdrantClient:
        return get_qdrant_client()

    @property
    def qdrant_db(self) -> QdrantDB:
        return get_qdrant_db()

    @property
    def brain_size(self):
        # Not Implemented
        return 0
        # self.get_unique_brain_files()
        # current_brain_size = sum(float(doc["size"]) for doc in self.files)

        # return current_brain_size

    @property
    def remaining_brain_size(self):
        return (
            float(self.max_brain_size)  # pyright: ignore reportPrivateUsage=none
            - self.brain_size  # pyright: ignore reportPrivateUsage=none
        )

    @classmethod
    def create(cls, *args, **kwargs):
        commons = {"supabase": get_supabase_client()}
        return cls(
            commons=commons, *args, **kwargs  # pyright: ignore reportPrivateUsage=none
        )  # pyright: ignore reportPrivateUsage=none

    # TODO: move this to a brand new BrainService
    def get_brain_users(self):
        response = (
            self.supabase_client.table("brains_users")
            .select("id:brain_id, *")
            .filter("brain_id", "eq", self.id)
            .execute()
        )
        return response.data

    # TODO: move this to a brand new BrainService
    def delete_user_from_brain(self, user_id):
        results = (
            self.supabase_client.table("brains_users")
            .select("*")
            .match({"brain_id": self.id, "user_id": user_id})
            .execute()
        )

        if len(results.data) != 0:
            self.supabase_client.table("brains_users").delete().match(
                {"brain_id": self.id, "user_id": user_id}
            ).execute()

    def delete_brain(self, user_id):
        results = self.supabase_db.delete_brain_user_by_id(user_id, self.id)

        if len(results.data) == 0:
            return {"message": "You are not the owner of this brain."}
        else:
            self.supabase_db.delete_brain_vector(self.id)
            self.supabase_db.delete_brain_user(self.id)
            self.supabase_db.delete_all_brain_data(self.id)
            self.supabase_db.delete_brain(self.id)

            self.qdrant_db.delete_all_vectors_from_brain(self.id)
    
    def delete_brain_force(self):
        self.supabase_db.delete_brain_chat_history(self.id)
        self.supabase_db.delete_brain_vector(self.id)
        self.supabase_db.delete_brain_user(self.id)
        self.supabase_db.delete_all_brain_data(self.id)
        self.supabase_db.delete_brain(self.id)

        self.qdrant_db.delete_all_vectors_from_brain(self.id)

    def create_brain_vector(self, vector_id, file_sha1):
        return self.supabase_db.create_brain_vector(self.id, vector_id, file_sha1)

    def create_brain_data(self, data_sha1:str, meatdata=None):
        return self.supabase_db.create_brain_data(self.id, data_sha1, meatdata) 

    def get_vector_ids_from_file_sha1(self, file_sha1: str):
        return self.supabase_db.get_vector_ids_from_file_sha1(file_sha1)

    def update_brain_with_file(self, file_sha1: str):
        # not  used
        vector_ids = self.get_vector_ids_from_file_sha1(file_sha1)
        for vector_id in vector_ids:
            self.create_brain_vector(vector_id, file_sha1)

    def get_unique_brain_files(self):
        """
        Retrieve unique brain data (i.e. uploaded files and crawled websites).
        """

        vector_ids = self.supabase_db.get_brain_vector_ids(self.id)
        self.files = get_unique_files_from_vector_ids(vector_ids)

        return self.files
    
    def get_unique_brain_datas(self):
        """
        Retrieve unique brain data (i.e. uploaded files and crawled websites).
        """

        metadatas = self.supabase_db.get_brain_metadatas(self.id)
        self.datas = [{
            'name': metadata['data_name'],
            'size': metadata['data_size'],
            'sha1': metadata['data_sha1'],
        } for metadata in metadatas]
        # self.files = get_unique_files_from_vector_ids(vector_ids)

        return self.datas

    def delete_file_from_brain(self, file_name: str):
        return self.supabase_db.delete_file_from_brain(self.id, file_name)
    
    def delete_data_from_brain(self, data_sha1: str):
        self.supabase_db.delete_data_from_brain(self.id, data_sha1)
        # associated_brains_response = (
        #     self.supabase_client.table("brains_data")
        #     .select("brain_id")
        #     .filter("data_sha1", "eq", data_sha1)
        #     .execute()
        # )
        # associated_brains = [
        #     item["brain_id"] for item in associated_brains_response.data
        # ]
        # if not associated_brains:
        self.qdrant_db.delete_vectors_from_brain(self.id, data_sha1)

    def generate_annotation(self, user_text: str) -> AnnotationMessage:
        model = os.getenv("ANNOTATION_MODEL", "gpt-4")
        max_tokens = int(os.getenv("ANNOTATION_MAX_TOKENS", 4096))
        # brain_details = self.supabase_db.get_brain_details(self.id)
        # brain_overview = brain_details.overview
        brain_overview = None
        
        system_prompt = self.annotation_system_prompt(
            brain_overview,
        )

        decoding_args = {
            "temperature": 1.0,
            "n": 1,
            "max_tokens": max_tokens,  # hard-code to maximize the length. the requests will be automatically adjusted
            "top_p": 1.0,
            "stop": ["\n20", "20.", "20."],
        }

        # logit_bias={"50256": -100}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]

        shared_kwargs = dict(
            model=model,
            **decoding_args,
            # **logit_bias,
        )
        sleep_time = 2

        while True:
            try:
                response = openai.ChatCompletion.create(messages=messages, **shared_kwargs)
                content = response.choices[0]["message"]["content"]
                status_code = 200
                break
            except openai.error.OpenAIError as e:
                logger.warning(f"OpenAIError: {e}.")
                if "Rate limit reached for" in str(e):
                    logger.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)  # Annoying rate limit on requests.
                elif "Please reduce your prompt" in str(e):
                    logger.error("Reduce your prompt")
                    content = str(e)
                    break
                # TODO: handle other errors(OpenAI billing error)
                else:
                    status_code = 500
                    content = str(e)
                    logger.error(f"Unexpected error: {e}")
                    break    
        
        if status_code == 200:
            annotation_message = self.annotation_parser(content)
        else:
            annotation_message = AnnotationMessage(status_code=status_code, message=content)
        return annotation_message

    def annotation_system_prompt(self, overview: str | None) -> str:
        if not overview:
            overview = "I am a helpful text annotator."
            logger.info(f"Expert don't have own overview, So overview is set to {overview} as default.")
        else :
            logger.info(f"Expert overview: {overview}.")
        
        system_prompt = f"""This is your overview written by yourself. {overview}
As an expert, you will need to annotate the user's text in your field of expertise.

Requirements:
1. You should rewrite the text with annotations.
2. First begin with identifying the words or a sentence requiring annotation. In rewrited text, that parts are encapsulated in triple square brackets [[[*]]].
3. Proceed to insert annotation into the text. annotations are encapsulated in triple brackets (((*))). Annotation involves annotation type, comments and analysis. 
    3.1 Annotation type: Selecting from ['Incorrect', 'Good', 'Need Information']. 
    3.2 Comments is a sentence that explains the annotation.
    3.3 Analysis is the analysis of the annotation. It should be several sentences. Analysis should explain why more information is needed or why the annotation is incorrect in detail.

Annotation must be following style:
(((Type::<annotation type> Comments::<comments> Analysis::<analysis>)))

Remember, it is possible to annotate multiple segments of the text. And You MUST not insert any words except original text and annotation.

Here is examples of responses:
###
Example 1
Consuming detox teas can [[[instantly]]](((Type::Incorrect Comments::This word is misleading because the human body's detoxification process is not instantaneous. Analysis::It's a constant process involving different organs mainly liver and kidneys, and it cannot be hurried by consumming any specific product, including detox teas.))) purify the body by flushing out all the toxins, leading to accelerated weight loss. These teas act as a [[[superfood]]](((Type::Good Comments::This is an important word because it highlights the subject of choice and nutrition's impact on overall health. Analysis:: It's a term often used to describe nutrient-rich food, suggesting that the consumable (in this case, detox teas) is exceptionally beneficial for health.))), providing an instant health upgrade.

###
Example 2:
[[[Our new breakthrough product is designed to nurture healthy hair growth and has been thoroughly tested in our state-of-the-art labs.]]](((Type::Need Information Comment::What does nurture healthy hair growth mean? Does it prevent hair loss, or does it promote new hair growth? Both? What were the test procedures and results in the laboratory? Analysis::The information is needed due to ambiguity in the phase "nurture healthy hair growth" It's unclear what th product does specifically does it prevent hair loss, promote new hair growth, or both? More details would give a better understanding of the product's benefits. Moreover the statement mentions that the product has been "thoroughly tested" yet provides no further detail. It leaves the reader unsure what kind of tests were run, what the results were, and how these inform the product's effectiveness. Sharing specific, relevant testing procedures and results adds credibility to the product and helps strengthen the claims made about its performance.))). [[[It is based on cutting-edge science that leverages the inherent qualities of natural extracts.]]](((Type::Need Information Comment::What specific natural extracts are used and what are their benefits? How do they contribute to hair growth? Analysis::The benefits associated with these extracts and how they contribute to hair growth is also significant because it provides a basis for understanding the product's effectiveness. By detailing the relationship between the ingredients used and the claimed benefits, potential consumers can understand how the product works, fostering greater trust in the product.))). It's suitable for all hair types, including curly, straight, and wavy. [[[In fact, using it on a weekly basis can quintuple the rate of hair growth.]]](((Type::Incorrect Comment::The claim of quintupling the rate of hair growth is misleading and likely inaccurate as hair growth rate is largely determined by factors like genetics and overall health, and cannot be quintupled by any product. Analysis::Here should be anaylysis of the claim.))). Furthermore, our product is hypoallergenic, so even people with the most sensitive scalp can use it without any fear of irritation or discomfort. We believe in the power of nature and science to deliver tangible results for a wide range of hair concerns.

###
Example 3:
Chronic stress not only takes a toll on your mental health, but it can also manifest physically in the form of health conditions like heart disease and diabetes. It's crucial, therefore, to prioritize [[[stress management]]](((Type::Good Comment::'stress management' is a very important phrase that makes the text more valuable. Analysis::It highlights the need for intentional practices and strategies to handle stress, as opposed to treating it as an unavoidable part of life. The term brings in the element of personal control and empowerment over one's mental health. and indirectly gives a nod to the field of behavioral health and therapeutic interventions. It also hints at the importance of preventative measures to avoid the onset of stress-induced health conditions, contributing towards promoting a healthier and more balanced lifestyle.))) practices for overall well-being. Staying active, practicing mindfulness, and maintaining a healthy diet are valuable steps to mitigate the effects of stress."
"""
        return system_prompt

    def annotation_parser(self, content: str) -> AnnotationMessage:
        logger.info(f"Parsing started: {content}")
        splitted = re.split(r'\[\[\[|\)\)\)', content)
        annotations = []
        for i, word in enumerate(splitted):
            if i%2 == 0:
                annotations.append(Annotation(origin=word, type="origin"))
            else:
                pair = re.split(r'\]\]\]\(\(\(', word)
                if len(pair) != 2:
                    logger.error(f"Parsing error: {word}")
                    return AnnotationMessage(status_code=501, message=f"Parsing error: {word}")
                annotation_detail = re.split(r'Type::|Comments::|Analysis::', pair[1])
                if len(annotation_detail) != 4:
                    logger.error(f"Parsing error: {word}")
                    return AnnotationMessage(status_code=501, message=f"Parsing error: {word}")
                else:
                    annotations.append(Annotation(origin=pair[0], type=annotation_detail[1], comments=annotation_detail[2], analysis=annotation_detail[3]))
        return AnnotationMessage(status_code=200, message="Annotation successed", annotations=annotations)



        print(matches)

        assert NotImplementedError
        
class Personality:
    extraversion: int = 0
    neuroticism: int = 0
    conscientiousness: int = 0

    def __init__(self, extraversion, neuroticism, conscientiousness) -> None:
        self.extraversion = extraversion
        self.neuroticism = neuroticism
        self.conscientiousness = conscientiousness

