import os
from pathlib import Path

import openai
from llama_hub.nougat_ocr import PDFNougatOCR

openai.api_key = os.getenv("OPENAI_API_KEY")

# load the Nougat model for academic OCR
reader = PDFNougatOCR()
print(reader)

# directory to PDFs for human behavior change project (HBCP)
documents_directory = "/Users/jerhadf/Desktop/elicit/HBCP/PDFs"

# get a list of all PDF files in the directory

pdf_path = Path("LLMs/data/whatalgorithmscantransformerslearn.pdf")
print(pdf_path)
reader.load_data(pdf_path)

# pdf_files = [
#     os.path.join(documents_directory, f)
#     for f in os.listdir(documents_directory)
#     if f.endswith(".pdf")
# ]
# print(pdf_files)

# load data for each PDF in the directory
# documents = [reader.load_data(pdf) for pdf in pdf_files]

# # questions
# questions = {
#     "title": "What is the title of this paper?",
#     "author": "Who is the primary author (first author) of this paper?",
#     "date": "What is the publication date of this paper? Format as YYYY-MM-DD.",
#     "study_arms": (
#         "What were all the study arms in this paper?"
#         "A study arm is a group of participants that receive a particular intervention."
#         "Return as a list of the names of study arms, using the names from the paper."
#         "For example, your list could be ['control', 'intervention', 'placebo']."
#     ),
#     "outcomes": "What were the quantitative outcomes of the intervention in this study?",
# }

# def write_qa_results_to_csv(documents, question_outcomes):
#     # create an agent and ask the same question for each document, saving results to file
#     with open("qa_results.csv", "w", newline="") as csvfile:
#         # define the field names for the csv file
#         fieldnames = ["document", "question", "answer"]
#         # create a csv writer object
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         # write the header row to the csv file
#         writer.writeheader()
#         # iterate over each document
#         for document in documents:
#             # create an OpenAI agent
#             agent = OpenAIAgent.from_defaults()
#             # get the answer from the agent
#             answer = agent.chat()
#             # write the document, question, and answer to the csv file
#             writer.writerow(
#                 {"document": document, "question": question_outcomes, "answer": answer}
#             )

# write_qa_results_to_csv(documents, question_outcomes)

# # Define a Pydantic model for Q&A
# class Question(BaseModel):
#     question: str = Field(..., description="The question to answer.")


# # Define a prompt template for the OpenAI agent
# prompt_template_str = "Extract the following data from the text: {input_str}"

# # Create an OpenAIPydanticProgram
# program = OpenAIPydanticProgram.from_defaults(
#     output_cls=MyData,
#     prompt_template_str=prompt_template_str,
#     verbose=True,
# )

# # Now you can use this program to extract data from the agent's responses
# output = program(input_str=question_outcomes)


# arxiv_tool = ArxivToolSpec()

# agent = OpenAIAgent.from_tools(
#     arxiv_tool.to_tool_list(),
#     verbose=True,
# )

# print(agent.chat("Whats going on with the superconductor lk-99"))
