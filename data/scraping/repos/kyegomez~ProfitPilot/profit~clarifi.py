from langchain.llms import Clarifai


# class ClarifiLLM:
#     def __init__(
#             self, 
#             clarifai_pat: str = "890cdb0cb5aa4795ba51af9670120a1e", 
#             user_id="meta", 
#             app_id="Llama-2", 
#             model_id="llama2-70b-chat"
#         ):
#         self.CLARIFAI_PAT = clarifai_pat
#         self.USER_ID = user_id
#         self.APP_ID = app_id
#         self.MODEL_ID = model_id
#         self.clarifai_llm = Clarifai(
#             pat=self.CLARIFAI_PAT, 
#             user_id=self.USER_ID, 
#             app_id=self.APP_ID, 
#             model_id=self.MODEL_ID
#         )
#     def generate(self, question):
#         return self.clarifai_llm(question)
    
#     def __call__(self, question):
#         return self.clarifai_llm(question)



