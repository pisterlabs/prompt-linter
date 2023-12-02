import os
import shutil
import json
import SubjectContext as SubjectContext

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import constants as constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

class CareerAgentService:

  def __init__(self):
    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    self.PERSONAL_DOCS_FOLDER = "personal_docs"
    self.PERSIST_FOLDER = "personal_docs_persist"
    self.GPT_4K_MODEL = "gpt-3.5-turbo"
    self.GPT_16K_MODEL = "gpt-3.5-turbo-16k"
    
    #define a dictionary to store the chat_history for each user_context
    self.store_embedding_index = {}
    self.store_chat_history = {}
    self.store_small_chain = {}
    self.store_large_chain = {}
    self.store_subject_context = {}

    # TODO: The name of the candidate will be set dynamically when multiple subjects are supported
    subject_context = SubjectContext.SubjectContext(applicant_name = "Alex Worden", subject_id = "AlexWorden")
    self.save_subject_context(subject_context)
    self.initialize_subject(subject_context.id, rebuild_index=False)

# ====================================================================================================
# These 'private' methods will be refactored to retrieve resources that have been persisted and/or cached
# and are relative to the given user_context

  def get_subject_context(self, subject_id: str) -> SubjectContext:
    return self.store_subject_context[subject_id]
  
  def save_subject_context(self, subject_context: SubjectContext):
    if (subject_context == None or subject_context.id == None):
      raise Exception("Subject Context and its id cannot be None")
    self.store_subject_context[subject_context.id] = subject_context

  # Store the chat_history in a dictionary with the user_context.id as the key
  def set_chat_history(self, subject_id: str, chat_history):
    self.store_chat_history[subject_id] = chat_history

  # Get the chat_history from the dictionary with the user_context.id as the key
  def get_chat_history(self, subject_id: str):
    return self.store_chat_history[subject_id]

  # Clear the chat_history for the user_context
  def clear_chat_history(self, subject_id: str):
    self.store_chat_history[subject_id] = []

  def get_embedding_index(self, subject_id: str):
    return self.store_embedding_index[subject_id]
  
  def set_embedding_index(self, subject_id: str, index):
    self.store_embedding_index[subject_id] = index

  def get_small_chain(self, subject_id: str):
    return self.store_small_chain[subject_id]

  def set_small_chain(self, subject_id: str, small_chain):
    self.store_small_chain[subject_id] = small_chain

  def get_large_chain(self, subject_id: str):
    return self.store_large_chain[subject_id]

  def set_large_chain(self, subject_id: str, large_chain):
    self.store_large_chain[subject_id] = large_chain

  # ====================================================================================================

  def ask_conversational_question(self, subject_id, question: str, use_chat_history=True): 
    
    # If the question is empty, return an empty string
    if (question == ""):
      return ""
    
    subject_context: SubjectContext = self.get_subject_context(subject_id)
    if (subject_context == None):
      raise Exception("Subject Context not found for subject_id: " + subject_id)

    if (use_chat_history):
      chat_history = self.get_chat_history(subject_id)
    
    # If there is no chat history or we aren't using a chat history, augment the question with the PROMPT_CANDIDATE to give the answer the desired personality
    if ((use_chat_history == False) or (len(chat_history) == 0)):
      question = "Assume to role of " + subject_context.applicant_name + " who is a smart, friendly, humble, and truthful job candidate. ALWAYS provide your answer in the first person.\nQuestion: " + "\n" + question

    small_chain = self.get_small_chain(subject_id)
    answer = small_chain({"question": question, "chat_history": chat_history})['answer']
    
    if (use_chat_history):
      chat_history.append((question, answer))
      self.set_chat_history(subject_id, chat_history)
    
    return answer

  # ====================================================================================================

  def ask_simple_with_context(self, subject_id, question: str, use_chat_history=False):
    small_chain = self.get_small_chain(subject_id)
    if (use_chat_history):
      return small_chain({"question": question, "chat_history": self.get_chat_history(subject_id)})['answer']
    else: 
      return small_chain({"question": question, "chat_history": []})['answer']

  # ====================================================================================================

  def ask_complex_with_context(self, subject_id: str, question: str, use_chat_history=False):
    big_chain = self.get_large_chain(subject_id)
    if (use_chat_history):
      return big_chain({"question": question, "chat_history": self.get_chat_history(subject_id)})['answer']
    else:
      return big_chain({"question": question, "chat_history": []})['answer']

  # ====================================================================================================

  def query_context(self, subject_id, question):
    return self.get_embedding_index(subject_id).query(question)

  # ====================================================================================================
  
  def ask_without_context(self, prompt):
    response = openai.Completion.create(
      model="gpt-3.5-turbo-instruct",
      prompt=prompt,
      temperature=0,
      max_tokens=250
    )
    return response['choices'][0]['text']

  # ====================================================================================================

  def initialize_subject(self, subject_id: str, rebuild_index=False):

    subject_ctx = self.get_subject_context(subject_id)
    # if the subject_ctx is None, throw an error
    if (subject_ctx == None):
      raise Exception("Subject Context not found for subject_id: " + subject_id)
      
    # TODO: Refactor to not be hardcoded foler and allow for multiple folders based upon the subject_id
    personal_docs_folder = self.PERSONAL_DOCS_FOLDER + "/" + subject_id
    persist_folder = self.PERSIST_FOLDER + "/" + subject_id

    if rebuild_index:
      print("Rebuilding the personal docs index...")
      # if the persist_folder exists, delete it
      if os.path.exists(persist_folder):
        shutil.rmtree(persist_folder)
      # create a new index
      loader = DirectoryLoader(personal_docs_folder)
      index = VectorstoreIndexCreator(
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0), 
        vectorstore_kwargs={"persist_directory":persist_folder}).from_loaders([loader])
    else:
      print("Reusing vectorstore from " + persist_folder + " directory...\n")
      vectorstore = Chroma(persist_directory=persist_folder, embedding_function=OpenAIEmbeddings())
      index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    
    # ====================================================================================================  
    # It is unfortunate that the langchain developers chose to use the unintuitive __call__ paradigm on their "chain" objects. This creates the confusing syntax that this chain object behaves like a function. 

    small_chain = ConversationalRetrievalChain.from_llm(
      llm=ChatOpenAI(model=self.GPT_4K_MODEL),
      retriever=index.vectorstore.as_retriever(search_kwargs={"k": 4}),
      verbose=False,
    )

    large_chain = ConversationalRetrievalChain.from_llm(
      llm=ChatOpenAI(model=self.GPT_16K_MODEL),
      retriever=index.vectorstore.as_retriever(search_kwargs={"k": 10}),
      verbose=True,
    )
    
    self.set_embedding_index(subject_id, index)
    self.set_small_chain(subject_id, small_chain)
    self.set_large_chain(subject_id, large_chain)
    self.clear_chat_history(subject_id)

  # ====================================================================================================
  # TODO: Refactor to use subject_id instead of user_context
  def get_matching_candidate_skills(self, subject_id: str) -> (list | str):
    subject_context = self.get_subject_context(subject_id)
    if (subject_context == None):
      raise Exception("Subject Context not found for subject_id: " + subject_id)
    
    if (subject_context.job_desc == None):
      raise Exception("Job description has not been set. Use the 'JD' command to set the job description.")

    job_qualification = self.ask_without_context("As an expert recruiter, identify and summarize up to 6 of the highest priority qualifications from the following Job Description given below. Provide your answer in JSON list format like the following example:\n\n [ \"Qualification Summary Text\", \"Qualification Summary Text\", \"Qualification Summary Text\" ]\n\nJob Description:\n" + subject_context.job_desc)
    print("\nTop Job Requirements:\n" + job_qualification)

    # Parse the top_requirements as JSON and iterate over the list of job_qualifications
    job_qualification = json.loads(job_qualification)
    most_relevant_skills = []
    for qualification in job_qualification:
      # Create a short paragraph on how the candidate's context meets each job qualification
      candidate_skill = self.query_context(subject_id, "You are an expert resume writer candidate described by the given context. Write text for a bullet point in a job application cover letter that is a concise summary that demonstrates your experiences and skills meet the following Job Qualification. Your answer must be from a first person perspective. Avoid directly quoting text from the following qualification word-for-word and do not claim to have skills or experience not described in the context.\n\nJob Qualification: " + qualification + "\n\n")
      print("\nJob qualification: " + qualification + "\nExperience: " + candidate_skill + "\n\n")
      # Add the candidate_skill to the list of most_relevant_skills
      most_relevant_skills.append(candidate_skill)

    print("\n" + subject_context.applicant_name + "'s most relevant experience to the most important requirement in the job description are:\n")
    # Create a bullet list of the most_relevant_skills and append to a string
    skills_list = ""
    for skill in most_relevant_skills:
      skills_list += " * " + skill + "\n"
    
    print("Candidate Skills:\n" + skills_list + "\n")

    # create a JSON list of the most_relevant_skills
    most_relevant_skills_json_list = json.dumps(most_relevant_skills)

    cleaned_json_skills = self.ask_complex_with_context(subject_id, "You are an recruitment expert and have been asked to copy-edit the following list of candidate skills to be included in a cover letter. You MUST provide your answer in JSON format where each list item is a string in a JSON list like the following format:\n [\"Item description\", \"Item description\", \"Item description\"]. Remove repeated skills and redundant statements from the input list items and condense each item in the following list without losing valuable Knowledge, Skills, and Abilities (KSA) or soft skills such as such as optimism, kindness, intellectual curiosity, strong work ethic, empathy, and integrity. Order the list in order of importance, and limit to a maximum of 6 items. Always include the first item as the number of years experience:\n\n" + most_relevant_skills_json_list)
    print("Cleaned Skills:\n" + cleaned_json_skills + "\n")

    # parse the jscon cleaned_json_skills into a list
    # handle exceptions parsing the json
    try:
      job_skills = json.loads(cleaned_json_skills)
    except: 
      print("Error parsing JSON. Trying to convert to JSON list of strings.")
      cleaned_json_skills = self.ask_without_context("Format the following into a JSON list of strings:\n" + cleaned_json_skills  + "\n")
      try:
        job_skills = json.loads(cleaned_json_skills)
      except:
        print("Unable to create cover letter. Error parsing JSON response from OpenAI: \n" + cleaned_json_skills + "\n")
        return []

    return (job_skills)

  # ====================================================================================================
  # TODO: Refactor to use subject_id instead of user_context
  def generate_cover_letter(self, subject_id: str):
    subject_context = self.get_subject_context(subject_id)
    if (subject_context == None):
      raise Exception("Subject Context not found for subject_id: " + subject_id)
    
    skill_list = self.get_matching_candidate_skills(subject_id)

    cover_letter = "Thank you for considering my application for the role of " + subject_context.job_title + " at " + subject_context.company_name + ". I believe the following skills and experience I have are a great fit:\n\n"
    for skill in skill_list:
      cover_letter += " * " + skill + "\n"
    
    cover_letter += "\n\n"

    mission_alignment = self.ask_complex_with_context(subject_id, "You are a diligent and smart job applicant. Write a very short closing statement in a cover letter to demonstrate how the company mission resonates with your interestes, career aspirations, or passions. Provide your answer by completing the following sentence in less than 25 words: I am excited about the opportunity to work at " + subject_context.company_name + " because I \n\n Base your answer upon the following job description:\n\n" + subject_context.job_desc + "\n")

    cover_letter += mission_alignment + "\n\n"

    cover_letter += "Thank you for your time and consideration,\n\n" + subject_context.applicant_name + "\n\n"
    return cover_letter


  # ====================================================================================================
  # TODO: Refactor to use subject_id instead of user_context
  def generate_resume(self, subject_id: str):

    subject_context = self.get_subject_context(subject_id)
    if (subject_context == None):
      raise Exception("Subject Context not found for subject_id: " + subject_id)

    query = "Write a resume for " + subject_context.applicant_name + " with the following sections: \n" + "Summary: summarising the candidate's valuable experience\nSummary of Skills and Experience: a short bullet list the candidates's relevant skills and experience to the job description\nWork Experience: Provide an entry for each company the candidate has worked at and a 3-5 item bullet list their relevant accomplishments\nEducation:\nThe resume must be tailored to the following job description:\n" + subject_context.job_desc + "\n"

    resume = self.ask_complex_with_context(query)
    return resume

  # ====================================================================================================
