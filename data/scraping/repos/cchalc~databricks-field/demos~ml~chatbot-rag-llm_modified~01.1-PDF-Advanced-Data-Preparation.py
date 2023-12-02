# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1.1/ Extra: Ingesting and preparing PDF for LLM Chatbot RAG
# MAGIC
# MAGIC ## In this example, we will focus on ingesting pdf documents as source for our retrieval process. 
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-1.png?raw=true" style="float: right; width: 500px; margin-left: 10px">
# MAGIC
# MAGIC This is an alternative version of [01-Data-Preparation]($./01-Data-Preparation) notebook.
# MAGIC
# MAGIC For this example, we will use Databricks ebook PDFs from [Databricks resources page](https://www.databricks.com/resources):
# MAGIC
# MAGIC Here are all the detailed steps:
# MAGIC
# MAGIC - Use autoloader to load the binary PDF as our first table. 
# MAGIC - Use ocrmypdf library to extract text from image-based PDFs (ORC).
# MAGIC - Use pymupdf and Langchain libraries to parse the text content of the PDFs.
# MAGIC - Use Langchain to split the texts into chuncks.
# MAGIC - Save our text chunk in a Delta Lake table, ready for Vector Search indexation.
# MAGIC
# MAGIC
# MAGIC Lakehouse AI not only provides state of the art solutions to accelerate your AI and LLM projects, but also to accelerate data ingestion and preparation at scale.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Install required external libraries 
# MAGIC %pip install transformers==4.30.2 ocrmypdf==15.1.0 pymupdf==1.23.3 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=cjc $db=chatbot $reset_all_data=false

# COMMAND ----------

# DBTITLE 1,Install the ORC in our cluster
# OCRmyPDF requires GhostScript & tesseract to be installed on all the nodes in the cluster. 
# This will run "sudo apt-get install -y ghostscript tesseract-ocr" in all your cluster.
# For production mode, prefer using a cluster init-script. See _resources/01-init for more details
install_ocr_on_nodes()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Ingesting Databricks ebook PDFs and extracting their pages
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-2.png?raw=true" style="float: right" width="400px">
# MAGIC
# MAGIC First, let's ingest our PDFs as a Delta Lake table with path urls and content in binary format. 
# MAGIC
# MAGIC We'll use [Databricks Autoloader](https://docs.databricks.com/en/ingestion/auto-loader/index.html) to incrementally ingeset new files, making it easy to incrementally consume billions of files from the data lake in various data formats. Autoloader can easily ingests our unstructured PDF data in binary format.
# MAGIC
# MAGIC *Note: For the purposes of this demo, we use DBFS for both storing the PDF files and storing checkpoints for Autoloader. However, the recommended way is to use [Databricks Volumes](https://docs.databricks.com/en/data-governance/unity-catalog/create-volumes.html)*

# COMMAND ----------

# MAGIC %fs ls /dbdemos/product/llm/databricks-doc

# COMMAND ----------

## only run if files do not exist
## datasets - https://github.com/databricks-demos/dbdemos-dataset/tree/main/llm/databricks-pdf-documentation 
# folder =  "/dbdemos/product/llm/databricks-doc"
# download_file_from_git('/dbfs'+folder+'/pdf_files', "databricks-demos", "dbdemos-dataset", "/llm/databricks-pdf-documentation")

# COMMAND ----------

# DBTITLE 1,Our pdf files are available in our Volume (or DBFS)
# List our raw PDF docs
folder =  "/dbdemos/product/llm/databricks-doc"
display(dbutils.fs.ls(folder+"/pdf_files"))

# COMMAND ----------

# DBTITLE 1,Ingesting PDF files as binary format using Databricks Autoloader

df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .option("pathGlobfilter", "*.pdf")
        .load(folder+'/pdf_files'))

# Write the data as a Delta table
(df.writeStream
  .trigger(once=True)
  .option("checkpointLocation", folder+'/checkpoints/raw_docs')
  .table('pdf_raw').awaitTermination())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM cloud_files_state('/dbdemos/product/llm/databricks-doc/checkpoints/raw_docs');

# COMMAND ----------

# MAGIC %sql SELECT * FROM pdf_raw LIMIT 2

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-pdf-3.png?raw=true" style="float: right" width="400px">
# MAGIC
# MAGIC ## Extracting our PDF content as text chunk
# MAGIC
# MAGIC We need to convert the pdf documents bytes as text, and extract chunks from their content.
# MAGIC
# MAGIC
# MAGIC This part can be tricky as pdf are hard to work with and can be saved as images, for which we'll need an OCR to extract the text.
# MAGIC
# MAGIC We'll then split the pdf by page, and ensure each page remains below the max token per document we defined (500). Remember that your chunk size depends of your use-case and how you'll run your final prompt.
# MAGIC <br style="clear: both">
# MAGIC
# MAGIC ### Splitting our big documentation page in smaller chunks
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/chunk-window-size.png?raw=true" style="float: right" width="700px">
# MAGIC
# MAGIC In this demo, some PDF can be really big, with a lot of text. We'll split them by page, and use Langchain [recursive_text_splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter), and ensure that each chunk isn't bigger than 500 tokens. 
# MAGIC
# MAGIC The chunk size and chunk overlap depend on the use case and the PDF files. Remember that your prompt+answer should stay below your model max window size (4000 for llama2). 
# MAGIC
# MAGIC  For more details, review the previous [01-Data-Preparation](01-Data-Preparation) notebook. 
# MAGIC
# MAGIC <br/>
# MAGIC <br style="clear: both">
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>Information</strong><br/>
# MAGIC   Remember that the following steps are specific to your dataset. This is a critical part to building a successful RAG assistant.
# MAGIC   <br/> Always take time to review the chunks created and ensure they make sense, containing relevant informations.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Create our documentation tables containing chunks
# %sql
# CREATE TABLE IF NOT EXISTS databricks_pdf_documentation  (id BIGINT GENERATED BY DEFAULT AS IDENTITY, content STRING, url STRING, page INT, total_page INT, ocr_status STRING);
# ALTER TABLE databricks_pdf_documentation SET OWNER TO `account users`;

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS databricks_pdf_documentation  (id BIGINT GENERATED BY DEFAULT AS IDENTITY, content STRING, path STRING, page INT, total_page INT, ocr_status STRING);
# MAGIC ALTER TABLE databricks_pdf_documentation SET OWNER TO `account users`;

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's start by extracting text from our PDF, including from image-based PDF using OCR:

# COMMAND ----------

# DBTITLE 1,Transform pdf as text
import io
import ocrmypdf
from ocrmypdf.exceptions import PriorOcrFoundError, EncryptedPdfError
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Transform the raw pdf bytes as a set of documents (1 per pdf page)
def extract_text_from_pdf(raw_content):
  try:
    #OCR the image-based PDFs as text
    ocred_raw_content = io.BytesIO()
    ocrmypdf.ocr(input_file=io.BytesIO(raw_content), output_file=ocred_raw_content, progress_bar=False,
                 deskew=True, optimize=0, output_type='pdf', fast_web_view=0, skip_big=10)
    ocred_raw_content.seek(0)
    ocr_status = 'ocred'
  except PriorOcrFoundError:
    #pdf was already a text
    ocred_raw_content = raw_content
    ocr_status = 'already text'
  except EncryptedPdfError:
    ocr_status = 'encrypted'
  
  #Parse the PDFs' texts using Langchain - InMemoryPDFLoader is defined in _resource/00-init
  return ocr_status, InMemoryPDFLoader(ocred_raw_content).load()

# COMMAND ----------

# DBTITLE 1,Let's try our function and extract the text from the lakehouse_for_retail pdf
#Let's try our function
with open("/dbfs/dbdemos/product/llm/databricks-doc/pdf_files/lakehouse_for_retail-082922.pdf", mode="rb") as pdf:
  ocr_status, doc = extract_text_from_pdf(pdf.read())  
  print(doc)

# COMMAND ----------

# MAGIC %md
# MAGIC This looks great. We'll now wrap it with a text_splitter to avoid having too big pages, and create a Pandas UDF function to easily scale that across multiple nodes.

# COMMAND ----------

from transformers import LlamaTokenizerFast
import ocrmypdf
from ocrmypdf.exceptions import PriorOcrFoundError, EncryptedPdfError
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load Llama2 tokenizer
tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
# Split our documents in smaller chunks if needed
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=500, chunk_overlap=50)

# Reduce the arrow batch size as our PDF can be big in memory (default is 10000)
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 50)

@pandas_udf("array<struct<content:string, page:int, total_page:int, ocr_status:string>>")
def parse_and_split(serie: pd.Series) -> pd.Series:
  #Remove non ascii from the docs and multiple spaces
  def remove_ascii(txt):
    txt = re.sub(r'[^\x00-\x7F]+',' ', txt)
    txt = re.sub(r'\n+', '\n', txt)
    return re.sub(r' +', ' ', txt)
  chunks = []
  for pdf_bytes in serie:
    ocr_status, doc = extract_text_from_pdf(pdf_bytes)
    #Split the pages to respect the maximum of 500 tokens per page
    splits = text_splitter.split_documents(doc)
    chunks.append([{'content': remove_ascii(c.page_content), 
                    'page': c.metadata['page'], 
                    'total_page': c.metadata['total_pages'], 
                    'ocr_status': ocr_status} for c in splits if len(c.page_content)>10])
  return pd.Series(chunks)

# COMMAND ----------

(spark.readStream.table('pdf_raw')
      .withColumn("chunk", F.explode(parse_and_split("content")))
      .select("chunk.*", "path")
  .writeStream
    .trigger(availableNow=True)
    .option("checkpointLocation", folder+'/checkpoints/pdf_chunks')
    .table('databricks_pdf_documentation').awaitTermination())

# COMMAND ----------

# MAGIC %md ERROR MESSAGE
# MAGIC ```
# MAGIC StreamingQueryException: [STREAM_FAILED] Query [id = 71b5d7aa-bc94-46e3-984e-244712931c89, runId = 8631265a-cc66-405a-bdfa-2d57fb726231] terminated with exception: A schema mismatch detected when writing to the Delta table (Table ID: 228e6c88-617c-4c70-b628-af8a4a6a08cc).
# MAGIC To enable schema migration using DataFrameWriter or DataStreamWriter, please set:
# MAGIC '.option("mergeSchema", "true")'.
# MAGIC For other operations, set the session configuration
# MAGIC spark.databricks.delta.schema.autoMerge.enabled to "true". See the documentation
# MAGIC specific to the operation for details.
# MAGIC
# MAGIC Table schema:
# MAGIC root
# MAGIC -- id: long (nullable = true)
# MAGIC -- content: string (nullable = true)
# MAGIC -- url: string (nullable = true)
# MAGIC -- page: integer (nullable = true)
# MAGIC -- total_page: integer (nullable = true)
# MAGIC -- ocr_status: string (nullable = true)
# MAGIC
# MAGIC
# MAGIC Data schema:
# MAGIC root
# MAGIC -- content: string (nullable = true)
# MAGIC -- page: integer (nullable = true)
# MAGIC -- total_page: integer (nullable = true)
# MAGIC -- ocr_status: string (nullable = true)
# MAGIC -- path: string (nullable = true)
# MAGIC ```

# COMMAND ----------

display(spark.table("databricks_pdf_documentation"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Our dataset is now ready! Let's create our Vector Search Index.
# MAGIC
# MAGIC Our dataset is now ready, and saved as a Delta Lake table.
# MAGIC
# MAGIC We could easily deploy this part as a production-grade job, leveraging Delta Live Table capabilities to incrementally consume and cleanup document updates.
# MAGIC
# MAGIC Remember, this is the real power of the Lakehouse: one unified platform for data preparation, analysis and AI.
# MAGIC
# MAGIC Next: Open the [02-Creating-Vector-Index]($./02-Creating-Vector-Index) notebook and create our embedding endpoint and index.
