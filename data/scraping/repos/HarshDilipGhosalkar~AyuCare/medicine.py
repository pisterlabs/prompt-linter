import json
import os
import pymongo
from flask_restful import Resource, reqparse
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    LangchainEmbedding,
    ServiceContext,
    StorageContext,
    load_index_from_storage,
    set_global_service_context,
)
from llama_index.memory import ChatMemoryBuffer

os.environ["TOKENIZERS_PARALLELISM"] = "False"
os.environ["OPENAI_API_KEY"] = os.getenv("FLASK_OPENAI_API_KEY")

try:
    client = pymongo.MongoClient(os.getenv("FLASK_MONGODB_URI"))
    db = client["hackathon"]
    print("Connected to the dev database successfully.")
except pymongo.errors.ConnectionFailure as e:
    print("Failed to connect to the dev database: %s", e)

medicineRef = db["medicine"]

model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
service_context = ServiceContext.from_defaults(embed_model=model)
set_global_service_context(service_context)

storage_context = StorageContext.from_defaults(persist_dir="./AI/medical_composition_index")
medical_index = load_index_from_storage(storage_context)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

ayurvedicMedicineRecommenderChatEngine = medical_index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="""
        I will provide you a medicine name and its chemical composition. 
        You should tell me its ayurvedic alternative from your custom knowledge.
        Format of your response should be strictly in json format and should strictly contain nothing extra before and after that:
        {"ayurvedic_alternative": "XYZ",
        "information_about_alternative": "XYZ"}
        If you are not able to find the disease, then send empty values but in the same format.
    """ ,
)

class Medicine(Resource):

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument("name", type=str, required=True, help="name is required")
        args = parser.parse_args()

        medicineName = args["name"]

        medicine = medicineRef.find_one({"name": medicineName})

        if medicine:
            medicine['_id'] = str(medicine['_id'])
            alternateMedicine = getAlternateMedicine(medicineName)
            medicine['ayurvedic_alternative'] = alternateMedicine['ayurvedic_alternative']
            medicine['information_about_alternative'] = alternateMedicine['information_about_alternative']
            return {"error": False, "data": medicine}
        else:
            return {"error": True, "message": "Medicine not found"}, 404
        

def getAlternateMedicine(medicineName):
    try:
        response = ayurvedicMedicineRecommenderChatEngine.chat(medicineName)
        print(response)
        json_response = json.loads(response.response)
        print(json_response)
        return json_response
    except Exception as e:
        return {"error": True, "message": str(e)}, 500