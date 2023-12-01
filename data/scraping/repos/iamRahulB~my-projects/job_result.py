import pdfplumber
import requests
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

class JobSuggestionResult:
	def __init__(self,content=None):
		self.content = content

		
	def openai_text(self,users_name,page_text,user_interest):
		
		send=f"hey users name is {users_name} and he is interested in {user_interest} and here is his resume text: {page_text}, so in result he wants in given format: hello, (his name), below that tell him if he is ready for the job or not. and if not ready then tell him what more he should. include his skills in pointwise "
		completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content":send}
  ]
)
		return completion.choices[0].message["content"]


	def job_suggestion(self,user, interest, resume_path):
		self.user=user
		self.interest=interest
		self.resume_path=resume_path

		with pdfplumber.open(self.resume_path) as pdf:
			first_page = pdf.pages[0]
			page_text = first_page.extract_text()

		
		openai_result_final=self.openai_text(self.user,page_text,self.interest)
		
		
		return openai_result_final
		
		