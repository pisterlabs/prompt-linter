# (A) LOAD SETTINGS & MODULES
import a_settings as set
import c_oto_rodo as oto
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from flask import Flask, Response, request
# import jwt # @TODO - ENABLE THIS TO OPEN FOR REGISTERED USERS ONLY

# (B) CHAIN
chain = RetrievalQA.from_chain_type(
  llm = oto.llm,
  retriever = Chroma(
    persist_directory = set.path_db,
    embedding_function = HuggingFaceInstructEmbeddings(**set.embed_args)
  ).as_retriever(),
  chain_type_kwargs = {
    "prompt": PromptTemplate (
      template = set.prompt_template,
      input_variables = ["question", "context"]
    )
  },
  ** set.chain_args
)

""" @TODO - ENABLE THIS TO OPEN FOR REGISTERED USERS ONLY
# (C) VERIFY USER
def jwtVerify(cookies):
  try:
    token = jwt.decode(
      jwt = cookies.get("cbsess"),
      key = set.jwt_secret,
      audience = set.http_host,
      algorithms = [set.jwt_algo]
    )
    # DO WHATEVER YOU WANT WITH THE DECODED USER TOKEN
    # print(token)
    return True
  except Exception as error:
    # print(error)
    return False
"""

# (D) FLASK
app = Flask(__name__)
@app.route("/", methods = ["POST"])
def bot():
  # (D1) CORS
  if "HTTP_ORIGIN" in request.environ and request.environ["HTTP_ORIGIN"] in set.http_allow:
    # (D1-1) ALLOW ONLY REGISTERED USERS
    """ @TODO - ENABLE THIS TO OPEN FOR REGISTERED USERS ONLY
    if jwtVerify(request.cookies) is False:
      return Response("Not Allowed", status = 405)
    """

    # (D1-2) ANSWER THE QUESTION
    data = dict(request.form)
    if "query" in data:
      ans = chain(data["query"])
      ans = ans["result"]
    else:
      ans = "Where's the question, yo?"
    response = Response(ans, status = 200)
    response.headers.add("Access-Control-Allow-Origin", request.environ["HTTP_ORIGIN"])
    response.headers.add("Access-Control-Allow-Credentials", "true")

  # (D2) ORIGIN NOT ALLOWED
  else:
    response = Response("Not Allowed", status = 405)
  return response

# (E) GO!
if __name__ == "__main__":
  app.run(
    host = set.http_host,
    port = set.http_port
  )