from flask_restful import Resource
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.user import User as UserModel
import os
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

model = LangchainEmbedding(HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
service_context = ServiceContext.from_defaults(embed_model=model)
set_global_service_context(service_context)

storage_context = StorageContext.from_defaults(persist_dir="./AI/balance_dosh_index")
medical_index = load_index_from_storage(storage_context)

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

balance_dosha_engine = medical_index.as_chat_engine(
    chat_mode="context",
    memory=memory,
    system_prompt="""
        You are an expert of Ayurveda who is telling the patient effects of the ayurvedic dosha (vata, pitta, kapha).
        I will provide you the percentage of vata, pitta, kapha and you should analyse my data and if found any threats then only provide me with potential health threats and available treatments to balance the dosha.
    """ ,
)

class AyurvedicDoshas(Resource):
    @jwt_required()
    def post(self):
        mobile_number = get_jwt_identity()

        response = UserModel.get_user_by_mobile_number(mobile_number)
        if response["error"]:
            return response, 500
        
        user = response["data"]

        query = f"""
            vata: {int(user.vata) * 10}%,
            pitta: {int(user.pitta) * 10}%,
            kapha: {int(user.kapha) * 10}%,
        """

        response = balance_dosha_engine.chat(query)

        return {"error": False, "data": response.response}, 200
        