# from integrating.dynmaic-dataset import *

# main one without dataset required
# payload =  question, domain, enterprise and target_language


import os
from modules import *
from dotenv import load_dotenv
from pydantic import BaseModel
import openai
load_dotenv()

os.environ["OPENAI_API_KEY"] = ""

app = Flask(__name__)

openai.api_key = ""

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def translate_to_target_language(text, target_language):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Translate the following text to {target_language}: '{text}'",
        max_tokens=100,
        temperature=0.0,
        n=1,
        stop=None
    )

    translated_text = response.choices[0].text.strip()
    return translated_text

def calculate_similarity_with_embedding(query, response):
    query_embedding = model.encode(query)
    response_embedding = model.encode(response)
    similarity = cosine_similarity([query_embedding], [response_embedding])[0][0]
    return similarity

def is_subsequence(subseq, sequence):
    it = iter(sequence)
    return all(char in it for char in subseq)

grpc_server_address = '112.196.81.250'  # Update with the correct gRPC server address
grpc_server_port = 50051  


def call_grpc_server(question):
    channel = grpc.insecure_channel(f"{grpc_server_address}:{grpc_server_port}")
    stub = service_pb2_grpc.YourServiceStub(channel)
   
    grpc_request = service_pb2.Question(text=question)
    grpc_response = stub.Ask(grpc_request)

    return grpc_response.answer


class YourService(service_pb2_grpc.YourServiceServicer):
    def Ask(self, request, context):
            query = request.text
            question = request.text
            target_language = request.target_language
            domain = request.domain
            enterprise = request.enterprise

            # Engineer prompt
            engineer_prompt = f"you are an assistant for a {domain} {enterprise}. provide me anything that i ask regarding the {domain} or {enterprise} field`"
            query = engineer_prompt + question + f" answer me assuming i was asking a {domain} {enterprise}"

            subsequence_responses = {
                "hello": "Hi, How may I assist you today!",
                "goodbye": "Goodbye! Have a great day!",
                # Add more subsequences and responses as needed
            }

            for subsequence, response in subsequence_responses.items():
                if is_subsequence(subsequence, question.lower()):
                    return jsonify({
                        "answer": response,
                        'status': 200,
                        'similarity': 1.0
                    })

            translated_query = translate_to_target_language(query, target_language)
            print(translated_query)
            return service_pb2.Answer(answer=response,translated_answer=translated_query)

def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_YourServiceServicer_to_server(YourService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve_grpc()
