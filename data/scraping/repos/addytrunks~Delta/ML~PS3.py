from langchain.llms import OpenAI

llm = OpenAI(openai_api_key="KEY_HERE")

llm.predict('''You are a admission counsellor where you have to rank the candidates based on the following criterias, namely marks(integer) and extracurricular(boolean). I will be giving you a list of candidates as a list of dictionaries, where each candidate is defined like this: {name:'Bob',extracurricular:true}, i want you to rank the candidates based off these criterias, give the first preference to marks, and if two candidates have same marks, go for the extracurricular and return me the ranked candidates as a list with only their names. Here is the list of candidates: 

candidates =[{"name": "John", "marks": 98,"extracurricular": False},{"name": "Alice", "marks": 95, "extracurricular": False},{"name": "Bob", "marks": 98, "extracurricular": True},]

. Don't give me instructions on how to go about it. Give me the expected output ONLY.''')