import textwrap
import plotly.graph_objects as go
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from llmConstants import chat

### LLM ###
LLM = chat
  
### Prompt template ###
PROMPT_TEMPLATE = """
    You are a psychology researcher extracting information from research papers.
    Answer the question using only the provided context below. Follow the additional instructions given (if any).
    Context: ### {context} ###
    Qn: ### {question} ###
    Additional instructions: ### {instructions} ###
    If you do not know the answer, output "unsure". 
  """


### RAG chain ###

def get_llm_response(db, query, article_title, num_chunks_retrieved=3, additional_inst=""):
  """
    Uses a RetrievalQA chain to feed both the prompt and article chunks to the LLM to obtain a response to the given query. 
    Outputs response results and the source documents referenced by the LLM.
  """
  # Initialise prompt template with 3 variables --> context, question, and instructions
  prompt = PromptTemplate(template=PROMPT_TEMPLATE,
                          input_variables=["context", "question"],
                          partial_variables={"instructions": additional_inst})

  # Define RAG chain
  qa_chain = RetrievalQA.from_chain_type(
      llm=LLM,
      # Only chunks from a particular article are retrieved. Chunk search method and number of chunks retrieved can be adjusted.
      retriever=db.as_retriever(
          search_type="similarity", # similarity search used to retrieve relevant chunks
          search_kwargs={'k': num_chunks_retrieved,
                        'filter': {"fileName": article_title}
          }),
      chain_type="stuff",
      chain_type_kwargs={"prompt": prompt},
      return_source_documents=True
  )
  # Make LLM call and retrieve results and source documents in a tuple
  result = qa_chain({"query": query})
  return (result["result"], result["source_documents"])


### Helper functions ###

def get_page_nums(page_num_str):
  """
    Takes in a string indicating page numbers in the form of "x to y".
    Outputs a list containing unique page number(s).
  """
  if len(page_num_str) > 0:
    split_separator = " to "
    page_num_list = page_num_str.split(split_separator)
    # Convert to integer and sort
    page_num_list = [int(page_num) for page_num in page_num_list]
    return sorted(list(set(page_num_list)))
  else:
    return []
   
def parse_source_docs(source_docs_list):
  """
    Takes in a list of langchain Document objects. 
    Outputs a tuple containing a list of document contents and string of page numbers parsed from source documents.
  """
  # Initialise holder lists
  doc_contents_list = []
  page_nums_list = []

  for source_doc in source_docs_list:
    # Extract page content and page number from source documents
    doc_content = source_doc.page_content
    page_num_str = source_doc.metadata.get("pageNum", "")
    doc_contents_list.append(doc_content)
    # Add list of page number(s) from this particular document to main accumulated list
    page_nums_list += get_page_nums(page_num_str)

  # Join the list of page numbers into string
  page_nums_list_unique = sorted(list(set(page_nums_list)))
  page_nums_str = ", ".join([str(page_num) for page_num in page_nums_list_unique])
  return (doc_contents_list, page_nums_str)

def add_line_breaks(text, num_char_in_line):
  """
    Add line breaks in text for easier viewing of output on dashboard table display.
  """
  # Define wrapping
  wrapper = textwrap.TextWrapper(width=num_char_in_line)
  # Add break tags to text wrap points
  new_text = "<br>".join(wrapper.wrap(text.strip()))
  return new_text


### Output functions ###

def get_pdf_analysis_table(pdf_analysis_df):
  """
    Generate Plotly table to display pdf analysis output
  """
  df = pdf_analysis_df.copy()[["article", "answer", "page_ref"]]
  # Add line breaks to wrap text
  df["answer"] = df["answer"].apply(lambda x: add_line_breaks(x, 80))

  # Define visual parameters
  layout = go.Layout(
    margin=go.layout.Margin(
      l=0, #left margin
      r=0, #right margin
      b=0, #bottom margin
      t=0, #top margin
      pad=0
    )
  )
  
  fig = go.Figure(data=[go.Table(
    columnorder = [1,2,3],
    columnwidth = [150,400,30],
    header = dict(
      values = ['<b>Article Name</b>', '<b>Answer</b>', '<b>Page Ref</b>'],
      fill_color='#203864',
      align=['left', 'left', 'center'],
      font=dict(color='white', size=14),
      height=40
    ),
    cells=dict(
      values=[df.article, df.answer, df.page_ref],
      fill_color=["white", "white", "white"],
      align=['left', 'left', "center"],
      font_size=14
      ))
  ], layout=layout)
  return fig